import jax
import jax.numpy as jnp
from flax import linen as nn
from functools import partial
from . import ninjax as nj 

from initializers import log_step_initializer, make_DPLR_HiPPO
from discretization import discrete_DPLR
from kernels import kernel_DPLR, causal_convolution
from scan import scan_SSM


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
# Ninjax implementation of S4 Layer
###############################
class S4Layer(nj.Module):

  def __init__(self, N:int, l_max, clip_eigs:bool=True):

    Lambda, P, B, _, _ = make_DPLR_HiPPO(N)
    # Initialize state matrix Lambda
    self.Lambda_re = self.get('Lambda_re', lambda rng, shape: Lambda.real, nj.rng(), (None,))
    self.Lambda_im = self.get('Lambda_im', lambda rng, shape: Lambda.imag, nj.rng(), (None,))
    if self.clip_eigs:
      self.Lambda = jnp.clip(self.Lambda_re, None, -1e-4) + 1j*self.Lambda_im
    else:
      self.Lambda = self.Lambda_re + 1j*self.Lambda_im
    # P is the low-rank factor
    self.P = self.get('P', lambda rng, shape: P, nj.rng(), (None,))
    self.B = self.get('B', lambda rng, shape: B, nj.rng(), (None,))

    # need to recompose C in call since jax optimizers don't work well with complex numbers
    self.C = self.get('C', jax.nn.initializers.normal(stddev=0.5**0.5), (N, 2))
    # self.C = self.C[...,0] + 1j*self.C[...,1]
    self.D = self.get('D', jax.nn.initializers.ones, (1,))
    self.step = jnp.exp(self.get('log_step', log_step_initializer(), (1,)))


    # Used for efficient parallelization during world model training
    self.K = kernel_DPLR(
      self.Lambda, self.P, self.P, self.B, self.C, self.step, self.l_max
    )
    # RNN step is used for agent training (world model evaluation)
    self.A, self.B, self.C = discrete_DPLR(
      self.Lambda, self.P, self.P, self.B, self.C, self.step, self.l_max
    )

  def __call__(self, batched_input_sequence, state, train=True):
    if train:
      y = jax.vmap(
        lambda u: causal_convolution(u, self.K) + self.D*u,
        in_axes=0, out_axes=0)(batched_input_sequence)
    else:
      x_k = jax.vmap(
        lambda prev_state, u: self.Lambda_bar @ prev_state + self.B_bar @ u,
        in_axes=0, out_axes=0)(prev_state, batched_input_sequence)
      y_k = jax.vmap(
        lambda x_k, u: (self.C_tilde @ x_k).real + self.D * u,
        in_axes=0, out_axes=0)(x_k, batched_input_sequence)

      x_k, y_s = scan_SSM(self.A, self.B, self.C, input_sequence[:jnp.newaxis], state)
      y = y_s.reshape(-1).real + self.D*input_sequence




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
