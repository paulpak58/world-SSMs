
import jax
import jax.numpy as jnp
from flax import linen as nn
from functools import partial
from jax.nn.initializers import lecun_normal, normal

from . import ninjax as nj

from discretization import discretize_zoh, discretize_bilinear
from scan import apply_ssm
from initializers import init_VinvB, init_CV, mimo_log_step_initializer, trunc_standard_normal
from initializers import make_DPLR_HiPPO



###############################
# Instantiates a single S5 layer
###############################
def S5LayerInit(
  blocks, init_block_size, H, P, C_init, discretization, dt_min, dt_max, conj_sym, clip_eigs, bidirectional,
  name='ssm'
):
  # H=d_model, P=ssm_size
  return partial(
    NinjaxS5Layer, blocks=blocks, init_block_size=init_block_size, H=H, P=P, C_init=C_init, discretization=discretization,
    dt_min=dt_min, dt_max=dt_max, conj_sym=conj_sym, clip_eigs=clip_eigs, bidirectional=bidirectional,
    name=name)



###############################
# Ninjax implementation of S5 Layer
###############################
class NinjaxS5Layer(nj.Module):

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

    # Initialize state matrix Lambda
    self.Lambda_re = self.get('Lambda_re', lambda rng, shape: Lambda.real, nj.rng(), (None,))
    self.Lambda_im = self.get('Lambda_im', lambda rng, shape: Lambda.imag, nj.rng(), (None,))
    self.V, self.Vinv = V, Vinv
    if self.clip_eigs:
      self.Lambda = jnp.clip(self.Lambda_re, None, -1e-4) + 1j*self.Lambda_im
    else:
      self.Lambda = self.Lambda_re + 1j*self.Lambda_im

    # Initialize input-to-state matrix B
    local_P = 2*self.P if self.conj_sym else self.P
    B_init = lecun_normal()
    B_shape = (local_P, self.H)
    self.B = self.get('B', lambda rng, shape: init_VinvB(B_init, rng, shape, self.Vinv), nj.rng(), B_shape)
    B_tilde = self.B[..., 0] + 1j*self.B[..., 1]

    # Initialize state-to-output matrix C
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
        C = self.get('C', C_init, nj.rng(), (self.H, 2*self.P, 2))
        self.C_tilde = C[..., 0] + 1j*C[..., 1]
      else:
        C = self.get('C', C_init, nj.rng(), (self.H, self.P, 2))
        self.C_tilde = C[..., 0] + 1j*C[..., 1]
    else:
      if self.bidirectional:
        self.C1 = self.get('C1', lambda rng, shape: init_CV(C_init, rng, shape, self.V), nj.rng(), C_shape)
        self.C2 = self.get('C2', lambda rng, shape: init_CV(C_init, rng, shape, self.V), nj.rng(), C_shape)
        C1 = self.C1[..., 0] + 1j*self.C1[..., 1]
        C2 = self.C2[..., 0] + 1j*self.C2[..., 1]
        self.C_tilde = jnp.concatenate((C1, C2), axis=-1)
      else:
        self.C = self.get('C', lambda rng, shape: init_CV(C_init, rng, shape, self.V), nj.rng(), C_shape)
        self.C_tilde = self.C[..., 0] + 1j*self.C[..., 1]
    
    # Initialize feedthrough matrix
    self.D = self.get('D', normal(stddev=1.0), nj.rng(), (self.H,))

    self.log_step = self.get('log_step', mimo_log_step_initializer, nj.rng(), (self.P, self.dt_min, self.dt_max))
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
    # if self.is_mutable_collection('cache'):
    #   self.x_k_1.value = state
    ys, state = jax.vmap(
      lambda u: apply_ssm(self.Lambda_bar, self.B_bar, self.C_tilde, u, self.conj_sym, self.bidirectional),
      in_axes=0, out_axes=0)(batched_input_sequence) 
    Du = jax.vmap(
      lambda u: self.D*u,
      in_axes=0, out_axes=0)(batched_input_sequence)
    out = ys + Du
    return out, state

  def step(self, input, prev_state):
    x_k = jax.vmap(
      lambda prev_state, u: self.Lambda_bar @ prev_state + self.B_bar @ u,
      in_axes=0, out_axes=0)(prev_state, input)
    y_k = jax.vmap(
      lambda x_k, u: (self.C_tilde @ x_k).real + self.D * u,
      in_axes=0, out_axes=0)(x_k, input)
    # y_k = (self.C_tilde @ x_k).real + self.D * input
    return y_k, x_k
