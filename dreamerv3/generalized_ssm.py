import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from functools import partial
from tensorflow_probability.substrates import jax as tfp

from . import jaxutils
from . import ninjax as nj
from . import nets


from initializers import make_DPLR_HiPPO, init_VinvB, trunc_standard_normal, init_CV, mimo_log_step_initializer
from jax.nn.initializers import lecun_normal, normal
from discretization import discretize_zoh, discretize_bilinear
from scan import apply_ssm


from s5 import S5LayerInit
from s4 import S4LayerInit
from dss import DSSLayerInit

f32 = jnp.float32
tfd = tfp.distributions
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute
Linear = nets.Linear



###############################
# Ninjax implementation of S5 Layer
###############################
class NinjaxS5Layer(nj.Module):


  #########################################################
  # Initialize params once, Perform discretization each step
  #########################################################
  def __init__(self, blocks:int, init_block_size:int, H:int, P:int,
    C_init:str, discretization:str, dt_min:float, dt_max:float, conj_sym:bool=True, clip_eigs:bool=False, bidirectional:bool=False, step_rescale:float=1.0
  ):
    self.blocks = blocks
    self.init_block_size = init_block_size
    self.H = H
    self.P = P
    self.C_init = C_init
    self.discretization = discretization
    self.dt_min = dt_min
    self.dt_max = dt_max
    self.conj_sym = conj_sym
    self.clip_eigs = clip_eigs
    self.bidirectional = bidirectional
    self.step_rescale = step_rescale
    

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

    self.Lambda_re, self.Lambda_im = Lambda.real, Lambda.imag
    self.V, self.Vinv = V, Vinv
    if self.clip_eigs:
      self.Lambda = jnp.clip(self.Lambda_re, None, -1e-4) + 1j*self.Lambda_im
    else:
      self.Lambda = self.Lambda_re + 1j*self.Lambda_im

    local_P = 2*self.P if self.conj_sym else self.P
    B_init = lecun_normal()
    B_shape = (local_P, self.H)
    self.B = init_VinvB(B_init, nj.rng(), B_shape, self.Vinv)
    # self.B = self.get('B', lambda rng, shape: init_VinvB(B_init, rng, shape, self.Vinv), B_shape)
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
        # C = self.get('C', C_init, (self.H, 2*self.P, 2))
        C = C_init((self.H, 2*self.P, 2), name='C')
        self.C_tilde = C[..., 0] + 1j*C[..., 1]
      else:
        C = C_init((self.H, self.P, 2), name='C')
        self.C_tilde = C[..., 0] + 1j*C[..., 1]
    else:
      if self.bidirectional:
        self.C1 = init_CV(C_init, nj.rng(), C_shape, self.V)
        self.C2 = init_CV(C_init, nj.rng(), C_shape, self.V)
        C1 = self.C1[..., 0] + 1j*self.C1[..., 1]
        C2 = self.C2[..., 0] + 1j*self.C2[..., 1]
        self.C_tilde = jnp.concatenate((C1, C2), axis=-1)
      else:
        self.C = init_CV(C_init, nj.rng(), C_shape, self.V)
        self.C_tilde = self.C[..., 0] + 1j*self.C[..., 1]
    # self.D = self.get('D', jax.nn.initializers.normal(stddev=1.0), (self.H,))
    self.D = normal(stddev=1.0)(nj.rng(), (self.H,))

    # self.log_step = self.get('log_step', mimo_log_step_initializer, (self.P, self.dt_min, self.dt_max))
    self.log_step = mimo_log_step_initializer(nj.rng(), (self.P, self.dt_min, self.dt_max))
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


  def ssm_forward(self, input_sequence):
    # batched_ssm = jax.vmap(apply_ssm, in_axes=(0, 0, 0, 0, None, None), out_axes=0)
    ys, state = apply_ssm(self.Lambda_bar, self.B_bar, self.C_tilde, input_sequence, self.conj_sym, self.bidirectional)

    # if self.is_mutable_collection('cache'):
    #   self.x_k_1.value = state
    Du = jax.vmap(lambda u: self.D*u)(input_sequence)
    out = ys + Du
    return out, state

  def __call__(self, batched_input_sequence):
    # outs = []
    # states = []
    # for batch in batched_input_sequence:
    #   out, state = self.ssm_forward(batch)
    #   outs.append(out)
    #   states.append(state)
    # return jnp.stack(outs), jnp.stack(states)
  
    def ssm_sequence(self, input_sequence):
      return apply_ssm(self.Lambda_bar, self.B_bar, self.C_tilde, input_sequence, self.conj_sym, self.bidirectional)

    ys, state = jax.vmap(
      lambda u: apply_ssm(self.Lambda_bar, self.B_bar, self.C_tilde, u, self.conj_sym, self.bidirectional),
      in_axes=0, out_axes=0)(batched_input_sequence) 

    # if self.is_mutable_collection('cache'):
    #   self.x_k_1.value = state
    Du = jax.vmap(
      lambda u: self.D*u,
      in_axes=0, out_axes=0)(batched_input_sequence)

    out = ys + Du
    return out, state


  def step(self, input, prev_state):
    x_k = self.Lambda_bar @ prev_state + self.B_bar @ input
    y_k = (self.C_tilde @ x_k).real + self.D * input
    return y_k, x_k


  







def build_ssm(config):
  if config.ssm =='s5':
    raise NotImplementedError
  elif config.ssm=='s4':
    raise NotImplementedError
  elif config.ssm=='s4d':
    raise NotImplementedError
  elif config.ssm=='dss':
    raise NotImplementedError
  elif config.ssm=='liquid':
    raise NotImplementedError
  elif config.ssm=='lru':
    raise NotImplementedError
  elif config.ssm=='mega':
    raise NotImplementedError
  else:
    raise NotImplementedError
  return ssm  

########################################
# Masked meanpool used for pooling output
########################################
def masked_meanpool(x, lengths):
  L = x.shape[0]
  mask = jnp.arange(L)<lengths
  return jnp.sum(mask[...,None]*x, axis=0)/lengths



class FlaxUnfrozen(nj.Module):

  def __init__(self, ctor, *args, **kwargs):
    self.module = ctor(*args, **kwargs)
    # self.batchnorm = kwargs['batchnorm']

  def __call__(self, *args, **kwargs):
    state = self.get('state', self.module.init, nj.rng(), *args, **kwargs)
    # if self.batchnorm:
    #   params = state['params'].unfreeze()
    #   batch_stats = state['batch_stats']
    # else:
    params = state['params'].unfreeze()
    params = {'params': params}
    return self.module.apply(params, *args, **kwargs)



class NJGeneralSequenceLayer(nj.Module):

  def __init__(self, ssm: nj.Module, dropout: float, d_model: int,
      activation: str='gelu', training: bool=True, prenorm: bool=False,
      batchnorm: bool=False, bn_momentum: float=0.9, step_rescale: float=1.0, **layer_args):
    self.seq = ssm
    self.dropout = dropout
    self.d_model = d_model
    self.activation = activation
    self.training = training
    self.prenorm = prenorm
    self.batchnorm = batchnorm
    self.bn_momentum = bn_momentum
    self.step_rescale = step_rescale

    self.layer_args = layer_args
    
    # self.seq = FlaxUnfrozen(self.ssm, step_rescale=self.step_rescale, name='ssm')
    # if self.activation in ['full_glu']:
    #   self.out1 = nn.Dense(self.d_model)
    #   self.out2 = nn.Dense(self.d_model)
    # elif self.activation in ['half_glu1', 'half_glu2']:
    #   self.out2 = nn.Dense(self.d_model)
    # if self.batchnorm:
    #   self.norm = nn.BatchNorm(use_running_average=not self.training, momentum=self.bn_momentum, axis_name='batch')
    # else:
    #   # self.norm = nj.FlaxModule(nn.LayerNorm, name='norm')
    #   self.norm = FlaxUnfrozen(nn.LayerNorm, name='norm')
    # self.drop = nn.Dropout(
    #   self.dropout,
    #   broadcast_dims=[0],
    #   deterministic=not self.training
    # )


  def __call__(self, x, state, scan):
    # Takes in (L, d_model) and outputs (L, d_model)

    skip = x
    if self.prenorm:
      # x = self.norm(x)
      x = self.get('norm', nets.Norm, 'layer')(x)
    if scan is not None: 
      # x, state = self.seq(x)
      # from s5 import NinjaxS5Layer
      init = jax.nn.initializers.variance_scaling(1, 'fan_avg', 'uniform')
      weights = self.get('weights', init, nj.rng(), (64, 32))   # This nj.rng() works but not in NinjaxS5Layer
      x, state = self.get('ssm', NinjaxS5Layer, **self.layer_args)(x)
      # x, state = self.get('ssm', self.seq).scan(x)
      # x, state = self.seq(x, state)
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
      # x = self.drop(nn.gelu(x))
      x = nn.gelu(x)
    else:
      raise NotImplementedError(f'Activation {self.activation} not implemented')
    x = skip + x
    if not self.prenorm:
      # x = self.norm(x)
      x = self.get('norm', nets.Norm, 'layer')(x)
    # cache_val = self.seq.x_k_1.value
    return x, state



###############################################
# Full Sequence Layer that includes the SSM, Norm,
# Dropout, and Nonlinearity
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
    self.seq = FlaxUnfrozen(self.ssm, step_rescale=self.step_rescale, name='ssm')
    if self.activation in ['full_glu']:
      self.out1 = nn.Dense(self.d_model)
      self.out2 = nn.Dense(self.d_model)
    elif self.activation in ['half_glu1', 'half_glu2']:
      self.out2 = nn.Dense(self.d_model)
    if self.batchnorm:
      self.norm = nn.BatchNorm(use_running_average=not self.training, momentum=self.bn_momentum, axis_name='batch')
    else:
      # self.norm = nj.FlaxModule(nn.LayerNorm, name='norm')
      self.norm = FlaxUnfrozen(nn.LayerNorm, name='norm')
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


##############################
# Stacked Deep SSM Layers
# Implemented in ninjax
##############################
class StackedSSM(nj.Module):
    
    def __init__(
      self, init_fn, n_layers, dropout, d_model, act, prenorm, batchnorm, bn_momentum, **layer_args
    ):
      seq_layer = GeneralSequenceLayer(
        ssm=init_fn, dropout=dropout, d_model=d_model, activation=act,
        prenorm=prenorm, batchnorm=batchnorm, bn_momentum=bn_momentum,
      )
      self.n_layers = n_layers
      # ssm layer attributes
      self.init_fn = init_fn
      self.dropout = dropout
      self.d_model = d_model
      self.act = act
      self.prenorm = prenorm
      self.batchnorm = batchnorm
      self.bn_momentum = bn_momentum
      self.layer_args = layer_args

    def __call__(self, x, state=None, mode='scan'):
      # format it as batch for vmap
      scan = jnp.ones((x.shape[0]), dtype=jnp.int32) if mode=='scan' else None
      for l in range(self.n_layers):
        # x, state = BatchedSequenceLayer(
        #   ssm=self.init_fn, dropout=self.dropout, d_model=self.d_model, activation=self.act,
        #   prenorm=self.prenorm, batchnorm=self.batchnorm, bn_momentum=self.bn_momentum,
        #   name=f'layer_{l}'
        # )(x, state, scan)
        # x, state = GeneralSequenceLayer(
        #   ssm=self.init_fn, dropout=self.dropout, d_model=self.d_model, activation=self.act,
        #   prenorm=self.prenorm, batchnorm=self.batchnorm, bn_momentum=self.bn_momentum
        # )(x, state, scan)
        x, state = self.get(f'layer_{l}', NJGeneralSequenceLayer,
          ssm=self.init_fn, dropout=self.dropout, d_model=self.d_model, activation=self.act,
          prenorm=self.prenorm, batchnorm=self.batchnorm, bn_momentum=self.bn_momentum, **self.layer_args
        )(x, state, scan)
      return x, state



#############################################
# Stacked Encoder Model as implemented by
#############################################
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

# BatchedSequenceLayer = nn.vmap(
#   GeneralSequenceLayer,
#   in_axes=0,
#   out_axes=0,
#   variable_axes={'params':None, 'dropout':None, 'batch_stats':None, 'cache':0, 'prime':None},
#   split_rngs={'params': False, 'dropout': True},
#   axis_name='batch'
# )
BatchedSequenceLayer = nn.vmap(
  NJGeneralSequenceLayer,
  in_axes=0,
  out_axes=0,
  variable_axes={'params':None, 'dropout':None, 'cache':0, 'prime':None},
  split_rngs={'params': False, 'dropout': True},
  axis_name='batch'
)

BatchedSSM = jax.vmap(
  StackedSSM,
  in_axes=0,
  out_axes=0,
  # variable_axes={'params':None, 'dropout':None, 'cache':0, 'prime':None},
  # split_rngs={'params': False, 'dropout': True},
  # axis_name='batch'
)



#################################################
# General-RSSM Model using Deep State-Space Layers
# Implemented with ninjax
#################################################
class General_RSSM(nj.Module):
  def __init__(
    self, ssm_size, blocks, d_model, n_layers, ssm_post_act, discretization,
    dt_min, dt_max, conj_sym, clip_eigs, bidirectional, C_init,
    dropout, prenorm, batchnorm, bn_momentum, 
    deter=1024, stoch=32, classes=32, unroll=False,
    initial='learned', unimix=0.01, action_clip=1.0, **kw
  ):
    self._deter = deter
    self._stoch = stoch
    self._classes = classes
    self._unroll = unroll
    self._initial = initial
    self._unimix = unimix
    self._action_clip = action_clip
    self._kw = kw

    # HiPPO Matrix set-up args
    self.ssm = 's5'
    self.blocks = blocks
    self.ssm_size = ssm_size
    self.init_block_size = int(self.ssm_size/self.blocks)
    self.conj_sym = conj_sym
    self.ssm_size = self.ssm_size//2 if self.conj_sym else self.ssm_size

    # Discretization and initialization args
    self.C_init = C_init
    self.discretization = discretization
    self.dt_min = dt_min
    self.dt_max = dt_max
    self.clip_eigs = clip_eigs
    self.bidirectional = bidirectional

    # General Sequence Layer args
    self.n_layers = n_layers
    self.dropout = dropout
    self.d_model = d_model
    self.act = ssm_post_act
    self.prenorm = prenorm
    self.batchnorm = batchnorm
    self.bn_momentum = bn_momentum

    # Calling this function instantiates a single layer
    SSM_MODELS = {'s4': S4LayerInit, 'dss': DSSLayerInit, 's5': S5LayerInit }
    init_fn = SSM_MODELS[self.ssm](
      blocks=self.blocks, init_block_size=self.init_block_size, H=self.d_model, P=self.ssm_size, C_init=self.C_init, discretization=self.discretization,
      dt_min=self.dt_min, dt_max=self.dt_max, conj_sym=self.conj_sym, clip_eigs=self.clip_eigs, bidirectional=self.bidirectional, 
    )
    self.layer_args = {'blocks': self.blocks, 'init_block_size': self.init_block_size, 'H': self.d_model, 'P': self.ssm_size, 'C_init': self.C_init,
      'discretization': self.discretization, 'dt_min': self.dt_min, 'dt_max': self.dt_max, 'conj_sym': self.conj_sym,'clip_eigs': self.clip_eigs,'bidirectional': self.bidirectional}
    self.ssm_args = {'init_fn': init_fn, 'n_layers': self.n_layers, 'dropout': self.dropout, 'd_model': self.d_model,
      'act': self.act, 'prenorm': self.prenorm, 'batchnorm': self.batchnorm, 'bn_momentum': self.bn_momentum}



  def initial(self, bs):

    # Define the initial state
    if self._classes:
      state = dict(
          deter=jnp.zeros([bs, self._deter], f32),  # deterministic output
          # hidden=jnp.zeros([bs, self.ssm_size], jnp.complex64), # hidden state
          hidden_re=jnp.zeros([bs, self.ssm_size], f32), # hidden state
          hidden_im=jnp.zeros([bs, self.ssm_size], f32), # hidden state
          logit=jnp.zeros([bs, self._stoch, self._classes], f32), # predictions
          stoch=jnp.zeros([bs, self._stoch, self._classes], f32)) # stochastic output
    else:
      state = dict(
          deter=jnp.zeros([bs, self._deter], f32),
          # hidden=jnp.zeros([bs, self.ssm_size], jnp.complex64),
          hidden_re=jnp.zeros([bs, self.ssm_size], f32),
          hidden_im=jnp.zeros([bs, self.ssm_size], f32),
          mean=jnp.zeros([bs, self._stoch], f32),
          std=jnp.ones([bs, self._stoch], f32),
          stoch=jnp.zeros([bs, self._stoch], f32))
    if self._initial == 'zeros':
      return cast(state)
    elif self._initial == 'learned':
      # deter = self.get('initial', jnp.zeros, state['deter'][0].shape, f32)
      deter = self.get('initial', jnp.zeros, state['deter'][0].shape, jnp.complex64)
      state['deter'] = jnp.repeat(jnp.tanh(deter)[None], bs, 0)
      if self.ssm=='':
        # the stochastic component comes from the hidden state
        state['stoch'] = self.get_stoch(cast(state['deter']))
      else:
        # the stochatic component comes from our output
        dummy_out = jnp.zeros([bs, self.d_model], f32)
        state['stoch'] = self.get_stoch(cast(dummy_out))
      return cast(state)
    else:
      raise NotImplementedError(self._initial)


  def observe(self, embed, action, is_first, state=None):
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    inputs = swap(action), swap(embed), swap(is_first)

    # Make sure that everything is right format and within bounds
    seq_len = len(jax.tree_util.tree_leaves(inputs)[0])
    expand_to_seq = lambda x: jnp.repeat(x[:, None], seq_len, 1)
    prev_state = state

    is_first = cast(is_first)
    prev_action = cast(action)
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))

    # Expand our init variables to match  sequence length
    prev_state = jax.tree_util.tree_map(expand_to_seq, (prev_state))
    init_seq = jax.tree_util.tree_map(expand_to_seq, (self.initial(len(is_first))))
    # Make given (L, B) sequence
    prev_action, prev_state = jax.tree_util.tree_map(
      lambda x: self._mask_sequence(x, 1.0 - is_first), (prev_action, prev_state))
    # match type of prev state components
    prev_state = jax.tree_util.tree_map(
        lambda x, y: x + self._mask_sequence(y, is_first), prev_state, init_seq) 

    prev_stoch = prev_state['stoch']  # img step checkpoint
    if self._classes:
      shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
      prev_stoch = prev_stoch.reshape(shape)
    if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
      shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
      prev_action = prev_action.reshape(shape)
    # Prior takes in previous output and input action
    x = jnp.concatenate([prev_stoch, prev_action], -1)

    # Model dim encoder
    x = self.get('img_in', Linear, **self._kw)(x)
    # Sequence Model batched scan
    full_bz = x.shape[0]*x.shape[1]
    dummy_in = jnp.zeros((full_bz, x.shape[-2], x.shape[-1]), f32)
    expanded_x = dummy_in.at[:x.shape[0]].set(x)
    expanded_out, expanded_deter = self.get('ssm', StackedSSM, **self.ssm_args, **self.layer_args)(expanded_x, state=None)
    # expanded_out, expanded_deter = self.get('ssm', BatchedSSM, **self.ssm_args, **self.layer_args)(expanded_x, state=None)
    out = expanded_out[:x.shape[0]]
    deter = expanded_deter[:x.shape[0]]

    # 1. Calculate the prior
    x = self.get('img_out', Linear, **self._kw)(out)
    stats = self._stats('img_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    prior = cast({'stoch': stoch, 'deter': out, 'hidden_re': deter.real, 'hidden_im': deter.imag, **stats})

    # 2. Calculate the posterior
    x = jnp.concatenate([out, embed], -1)
    x = self.get('obs_out', Linear, **self._kw)(x)
    stats = self._stats('obs_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    # post = {'stoch': stoch, 'deter': prior['deter'], 'hidden': prior['hidden'], **stats}
    post = cast({'stoch': stoch, 'deter': prior['deter'], 'hidden_re': prior['hidden_re'], 'hidden_im': prior['hidden_im'], **stats})


    raise Exception('end of obs')

    return post, prior
  

  def imagine(self, action, state=None):
    raise Exception('imagine')
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    state = self.initial(action.shape[0]) if state is None else state
    assert isinstance(state, dict), state

    prev_stoch = state['stoch']  # img step checkpoint
    prev_action = action
    if self._classes:
      shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
      prev_stoch = prev_stoch.reshape(shape)
    if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
      shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
      prev_action = prev_action.reshape(shape)
    # Prior takes in previous output and input action
    x = jnp.concatenate([prev_stoch, prev_action], -1)

    # Model dim encoder
    x = self.get('img_in', Linear, **self._kw)(x)

    # Sequence Model batched scan
    full_bz = x.shape[0]*x.shape[1]
    dummy_in = jnp.zeros((full_bz, x.shape[-2], x.shape[-1]), f32)
    expanded_x = dummy_in.at[:x.shape[0]].set(x)
    expanded_out, expanded_deter = self.get('ssm', StackedSSM, **self.ssm_args)(expanded_x, state=None)
    out = expanded_out[:x.shape[0]]
    deter = expanded_deter[:x.shape[0]]

    # 1. Calculate the prior
    x = self.get('img_out', Linear, **self._kw)(out)
    stats = self._stats('img_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    prior = {'stoch': stoch, 'deter': out, 'hidden_re': deter.real, 'hidden_im': deter.imag, **stats}
    return cast(prior)
  

  def obs_step(self, prev_state, prev_action, embed, is_first):
    print(f'prev state shape {prev_state["stoch"].shape} prev action shape {prev_action.shape} embed shape {embed.shape} is first shape {is_first.shape}')
    raise Exception('obs')
    is_first = cast(is_first)
    prev_action = cast(prev_action)
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))

    # Expand our init variables to match  sequence length
    prev_state = jax.tree_util.tree_map(expand_to_seq, (prev_state))
    init_seq = jax.tree_util.tree_map(expand_to_seq, (self.initial(len(is_first))))
    # Make given (L, B) sequence
    prev_action, prev_state = jax.tree_util.tree_map(
      lambda x: self._mask_sequence(x, 1.0 - is_first), (prev_action, prev_state))
    prev_state = jax.tree_util.tree_map(
        lambda x, y: x + self._mask_sequence(y, is_first), prev_state, init_seq) 

    return cast(post), cast(prior)
  

  def img_step(self, prev_state, prev_action):
    prev_stoch = prev_state['stoch']
    prev_action = cast(prev_action)
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))
    if self._classes:
      shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
      prev_stoch = prev_stoch.reshape(shape)
    if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
      shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
      prev_action = prev_action.reshape(shape)
    # Prior takes in previous output and input action
    x = jnp.concatenate([prev_stoch, prev_action], -1)

    # Model dim encoder
    x = self.get('img_in', Linear, **self._kw)(x)
    deter = prev_state['hidden_re'] + 1j*prev_state['hidden_im']
    # Inference step with SSM
    out, deter = self.get('ssm', StackedSSM, **self.ssm_args)(x=x, state=deter, mode='step')
    # 1. Calculate the prior
    x = self.get('img_out', Linear, **self._kw)(out)
    stats = self._stats('img_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    prior = {'stoch': stoch, 'deter': out, 'hidden_re': deter.real, 'hidden_im': deter.imag, **stats}
    return cast(prior)



  def get_dist(self, state, argmax=False):
    if self._classes:
      logit = state['logit'].astype(f32)
      return tfd.Independent(jaxutils.OneHotDist(logit), 1)
    else:
      mean = state['mean'].astype(f32)
      std = state['std'].astype(f32)
      return tfd.MultivariateNormalDiag(mean, std)


  def get_stoch(self, deter):
    x = self.get('img_out', Linear, **self._kw)(deter)
    stats = self._stats('img_stats', x)
    dist = self.get_dist(stats)
    return cast(dist.mode())
  

  def _stats(self, name, x):
    if self._classes:
      x = self.get(name, Linear, self._stoch * self._classes)(x)
      logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
      if self._unimix:
        probs = jax.nn.softmax(logit, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        logit = jnp.log(probs)
      stats = {'logit': logit}
      return stats
    else:
      x = self.get(name, Linear, 2 * self._stoch)(x)
      mean, std = jnp.split(x, 2, -1)
      std = 2 * jax.nn.sigmoid(std / 2) + 0.1
      return {'mean': mean, 'std': std}

  def _mask(self, value, mask):
    return jnp.einsum('b...,b->b...', value, mask.astype(value.dtype))

  def _mask_sequence(self, value, mask):
    return jnp.einsum('b l ..., b l -> b l ...', value, mask.astype(value.dtype))

  def dyn_loss(self, post, prior, impl='kl', free=1.0):
    if impl == 'kl':
      loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
    elif impl == 'logprob':
      loss = -self.get_dist(prior).log_prob(sg(post['stoch']))
    else:
      raise NotImplementedError(impl)
    if free:
      loss = jnp.maximum(loss, free)
    return loss

  def rep_loss(self, post, prior, impl='kl', free=1.0):
    if impl == 'kl':
      loss = self.get_dist(post).kl_divergence(self.get_dist(sg(prior)))
    elif impl == 'uniform':
      uniform = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), prior)
      loss = self.get_dist(post).kl_divergence(self.get_dist(uniform))
    elif impl == 'entropy':
      loss = -self.get_dist(post).entropy()
    elif impl == 'none':
      loss = jnp.zeros(post['deter'].shape[:-1])
    else:
      raise NotImplementedError(impl)
    if free:
      loss = jnp.maximum(loss, free)
    return loss
  