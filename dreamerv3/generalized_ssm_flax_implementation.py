import jax
import jax.numpy as jnp
from flax import linen as nn
from functools import partial
from jax.nn.initializers import lecun_normal, normal

# S5  
from discretization import discretize_zoh, discretize_bilinear
from scan import apply_ssm
from initializers import init_VinvB, init_CV, mimo_log_step_initializer, trunc_standard_normal
from initializers import make_DPLR_HiPPO
# S4
from initializers import log_step_initializer
from discretization import discrete_DPLR
from kernels import kernel_DPLR, causal_convolution
from scan import scan_SSM



###############################
# Instantiates a single S5 layer
###############################
def S5LayerInit(
  blocks, init_block_size, H, P, C_init, discretization, dt_min, dt_max, conj_sym, clip_eigs, bidirectional
):
  # H=d_model, P=ssm_size
  return partial(
    S5Layer, blocks=blocks, init_block_size=init_block_size, H=H, P=P, C_init=C_init, discretization=discretization,
    dt_min=dt_min, dt_max=dt_max, conj_sym=conj_sym, clip_eigs=clip_eigs, bidirectional=bidirectional, 
  )

###############################
# Flax implementation of S5 Layer
###############################
class S5Layer(nn.Module):
  # Lambda_re_init: jnp.DeviceArray # (P, )
  # Lambda_im_init: jnp.DeviceArray # (P, )
  # V: jnp.DeviceArray  # (P, P)
  # Vinv: jnp.DeviceArray # (P, P)
  blocks: int
  init_block_size: int

  H: int  # input features
  P: int  # state dim
  C_init: str
  discretization: str
  dt_min: float
  dt_max: float
  conj_sym: bool=True
  clip_eigs: bool=False
  bidirectional: bool=False
  step_rescale: float=1.0


  #########################################################
  # Initialize params once, Perform discretization each step
  #########################################################
  def setup(self):
    # Initialize DPLR HiPPO matrix
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(self.init_block_size)
    block_size = self.init_block_size//2 if self.conj_sym else self.init_block_size # conj. pairs halve the state space
    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    # Put each HiPPO on each block diagonal
    Lambda = (Lambda * jnp.ones((self.blocks, block_size))).ravel()
    V = jax.scipy.linalg.block_diag(*[V]*self.blocks)
    Vinv = jax.scipy.linalg.block_diag(*[Vc]*self.blocks)
    self.Lambda_re_init, self.Lambda_im_init = Lambda.real, Lambda.imag
    self.V, self.Vinv = V, Vinv


    local_P = 2*self.P if self.conj_sym else self.P
    self.Lambda_re = self.param('Lambda_re', lambda rng, shape: self.Lambda_re_init, (None,))
    self.Lambda_im = self.param('Lambda_im', lambda rng, shape: self.Lambda_im_init, (None,))
    # self.Lambda_re = self.param('Lambda_re', lambda rng, shape: self.Lambda_re_init, (None,))
    # self.Lambda_im = self.param('Lambda_im', lambda rng, shape: self.Lambda_im_init, (None,))
    if self.clip_eigs:
      self.Lambda = jnp.clip(self.Lambda_re, None, -1e-4) + 1j*self.Lambda_im
    else:
      self.Lambda = self.Lambda_re + 1j*self.Lambda_im
    B_init = lecun_normal()
    B_shape = (local_P, self.H)
    self.B = self.param('B', lambda rng, shape: init_VinvB(B_init, rng, shape, self.Vinv), B_shape)
    B_tilde = self.B[..., 0] + 1j*self.B[..., 1]

    if self.C_init in ['trunc_standard_normal']:
      C_init = trunc_standard_normal
      C_shape = (self.H, local_P, 2)
    elif self.C_init in ['lecun_normal']:
      C_init = lecun_normal()
      C_shape = (self.H, local_P, 2)
    elif self.C_init in ['complex_normal']:
      C_init = jax.nn.initializers.normal(stddev=0.5**0.5)
    else:
      raise NotImplementedError(f'C_init method {self.C_init} not implemented')
    if self.C_init in ['complex_normal']:
      if self.bidirectional:
        C = self.param('C', C_init, (self.H, 2*self.P, 2))
        self.C_tilde = C[..., 0] + 1j*C[..., 1]
      else:
        C = self.param('C', C_init, (self.H, self.P, 2))
        self.C_tilde = C[..., 0] + 1j*C[..., 1]
    else:
      if self.bidirectional:
        self.C1 = self.param('C1', lambda rng, shape: init_CV(C_init, rng, shape, self.V), C_shape)
        self.C2 = self.param('C2', lambda rng, shape: init_CV(C_init, rng, shape, self.V), C_shape)
        C1 = self.C1[..., 0] + 1j*self.C1[..., 1]
        C2 = self.C2[..., 0] + 1j*self.C2[..., 1]
        self.C_tilde = jnp.concatenate((C1, C2), axis=-1)
      else:
        self.C = self.param('C', lambda rng, shape: init_CV(C_init, rng, shape, self.V), C_shape)
        self.C_tilde = self.C[..., 0] + 1j*self.C[..., 1]
      self.D = self.param('D', jax.nn.initializers.normal(stddev=1.0), (self.H,))
      # self.log_step = self.param('log_step', init_log_steps, (self.P, self.dt_min, self.dt_max))
      self.log_step = self.param('log_step', mimo_log_step_initializer, (self.P, self.dt_min, self.dt_max))

      step = self.step_rescale * jnp.exp(self.log_step[:, 0])
      # Discretization
      if self.discretization in ['zoh']:
        self.Lambda_bar,  self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)
      elif self.discretization in ['bilinear']:
        self.Lambda_bar,  self.B_bar = discretize_bilinear(self.Lambda, B_tilde, step)
      else:
        raise NotImplementedError(f'Discretization method {self.discretization} not implemented')

    # RNN cache to store the internal states
    # self.x_k_1 = self.variable('cache', 'cache_x_k', jnp.zeros, (self.P,), jnp.complex64)

  def step(self, input, prev_state):
    x_k = self.Lambda_bar @ prev_state + self.B_bar @ input
    y_k = (self.C_tilde @ x_k).real + self.D * input
    return y_k, x_k


  def __call__(self, input_sequence):
    ys, state = apply_ssm(self.Lambda_bar, self.B_bar, self.C_tilde, input_sequence, self.conj_sym, self.bidirectional)
    # if self.is_mutable_collection('cache'):
    #   self.x_k_1.value = state
    Du = jax.vmap(lambda u: self.D*u)(input_sequence)
    out = ys + Du
    return out, state





###############################
# Instantiates a single S4 layer
###############################
def S4LayerInit(
  Lambda_re_init, Lambda_im_init, P_init, B_init, N, l_max, imagine, clip_eigs 
):
  return partial(
    S4Layer, Lambda_re_init=Lambda_re_init, Lambda_im_init=Lambda_im_init,
    P_init=P_init, B_init=B_init, N=N, l_max=l_max, imagine=imagine, clip_eigs=clip_eigs
)


###############################
# Flax implementation of S4 Layer
###############################
class S4Layer(nn.Module):
  Lambda_re_init: jnp.DeviceArray
  Lambda_im_init: jnp.DeviceArray
  P_init: jnp.DeviceArray
  B_init: jnp.DeviceArray

  N: int
  l_max: int
  imagine: bool=False
  clip_eigs: bool=False

  def setup(self):

    self.Lambda_re = self.param('Lambda_re', lambda rng, shape: self.Lambda_re_init, (None,))
    self.Lambda_im = self.param('Lambda_im', lambda rng, shape: self.Lambda_im_init, (None,))

    if self.clip_eigs:
      self.Lambda = jnp.clip(self.Lambda_re, None, -1e-4) + 1j*self.Lambda_im
    else:
      self.Lambda = self.Lambda_re + 1j*self.Lambda_im
    # P is the low-rank factor
    self.P = self.param('P', lambda rng, shape: self.P_init, (None,))
    self.B = self.param('B', lambda rng, shape: self.B_init, (None,))
    
    self.C = self.param('C', jax.nn.initializers.normal(stddev=0.5**0.5), (self.N, 2))
    self.C = self.C[...,0] + 1j*self.C[...,1]
    self.D = self.param('D', jax.nn.initializers.ones, (1,))
    self.step = jnp.exp(self.param('log_step', log_step_initializer(), (1,)))

    # Used for efficient parallelization during world model training
    self.K = kernel_DPLR(
      self.Lambda, self.P, self.P, self.B, self.C, self.step, self.l_max
    )
    # RNN step is used for agent training (world model evaluation)
    self.A, self.B, self.C = discrete_DPLR(
      self.Lambda, self.P, self.P, self.B, self.C, self.step, self.l_max
    )
    # RNN cache to store the internal states
    self.x_k_1 = self.variable('cache', 'cache_x_k', jnp.zeros(self.N,), jnp.complex64)

  def __call__(self, u):
    if not self.imagine:
      return causal_convolution(u, self.K) + self.D*u
    else:
      x_k, y_s = scan_SSM(self.A, self.B, self.C, u[:jnp.newaxis], self.x_k_1.value)
      if self.is_mutable_collection('cache'):
        self.x_k_1.value = x_k
      return y_s.reshape(-1).real + self.D*u




  



###############################################
# Full Sequence Layer that includes the SSM, Norm,
# Dropout, and Nonlinearity [Implemented in Flax]
##############################################
class GeneralSequenceLayer(nn.Module):
  ssm: nn.Module
  dropout: float
  d_model: int
  activation: str='gelu'
  training: bool=True
  prenorm: bool=False
  batchnorm: bool=False
  bn_momentum: float=0.9
  step_rescale: float=1.0

  def setup(self):
    # self.seq = self.ssm(step_rescale=self.step_rescale)
    # self.seq = nj.FlaxModule(self.ssm, step_rescale=self.step_rescale, name='ssm')
    self.seq = self.ssm(step_rescale=self.step_rescale)
    if self.activation in ['full_glu']:
      self.out1 = nn.Dense(self.d_model)
      self.out2 = nn.Dense(self.d_model)
    elif self.activation in ['half_glu1', 'half_glu2']:
      self.out2 = nn.Dense(self.d_model)
    if self.batchnorm:
      self.norm = nn.BatchNorm(use_running_average=not self.training, momentum=self.bn_momentum, axis_name='batch')
    else:
      self.norm = nn.LayerNorm()
    self.drop = nn.Dropout(
      self.dropout,
      broadcast_dims=[0],
      deterministic=not self.training
    )


  def __call__(self, x, state, scan):
    # Takes in (L, d_model) and outputs (L, d_model)
    skip = x
    if self.prenorm:
      x = self.norm(x)
    if scan is not None: 
      x, state = self.seq(x)
    else:
      x, state = self.seq.step(x, state)
    if self.activation in ['full_glu']:
      x = self.drop(nn.gelu(x))
      x = self.out1(x) * jax.nn.sigmoid(self.out2(x))
      x = self.drop(x)
    elif self.activation in ['half_glu1']:
      x = self.drop(nn.gelu(x))
      x = x * jax.nn.sigmoid(self.out2(x))
      x = self.drop(x)
    elif self.activation in ['half_glu2']:
      # only apply GELU to the gate input
      x1 = self.drop(nn.gelu(x))
      x = x * jax.nn.sigmoid(self.out2(x1))
      x = self.drop(x)
    elif self.activation in ['gelu']:
      x = self.drop(nn.gelu(x))
    else:
      raise NotImplementedError(f'Activation {self.activation} not implemented')
    x = skip + x
    if not self.prenorm:
      x = self.norm(x)
    # cache_val = self.seq.x_k_1.value
    return x, state
    # return x


########################################
# Masked meanpool used for pooling output
########################################
def masked_meanpool(x, lengths):
  L = x.shape[0]
  mask = jnp.arange(L)<lengths
  return jnp.sum(mask[...,None]*x, axis=0)/lengths



#########################
# Stacked Encoder Model
#########################
class GeneralSequenceModel(nn.Module):
  ssm: nn.Module
  d_output: int
  d_model: int
  n_layers: int
  activation: str='gelu'
  dropout: float=0.0
  training: bool=True
  mode: str=''
  prenorm: bool=False
  batchnorm: bool=False
  bn_momentum: float=0.9
  step_rescale: float=1.0

  def setup(self):
    # Initializes linear encoder and stack of S5 layers
    self.encoder = nn.Dense(self.d_model)
    self.layers = [
      GeneralSequenceLayer(
        ssm=self.ssm,
        dropout=self.dropout,
        d_model=self.d_model,
        activation=self.activation,
        training=self.training,
        prenorm=self.prenorm,
        batchnorm=self.batchnorm,
        bn_momentum=self.bn_momentum,
        step_rescale=self.step_rescale
      ) for _ in range(self.n_layers)
    ]
    self.decoder = nn.Dense(self.d_output)

  def __call__(self, x):
    # In: (L, d_input), Out: (L, d_model)
    if self.padded:
      x, length = x
    x = self.encoder(x)
    for l in self.layers:
      x = l(x)
    if self.mode in ['pool']:
      if self.padded:
        x = masked_meanpool(x, length)
      else:
        x = jnp.mean(x, axis=0)
    elif self.mode in ['last']:
      if self.padded:
        raise NotImplementedError(f'Mode must be in pool for self.padded=True')
      else:
        x = x[-1]
    else:
      x = x 
    x = self.decoder(x)
    x = nn.log_softmax(x, axis=-1)
    return x
  

############################################
# Final batched version to use in the pipeline
############################################
BatchedSequenceModel= nn.vmap(
  GeneralSequenceModel,
  in_axes=(0, 0),
  out_axes=0,
  variable_axes={'params':None, 'dropout':None, 'batch_stats':None, 'cache':0, 'prime':None},
  split_rngs={'params': False, 'dropout': True},
  axis_name='batch'
)