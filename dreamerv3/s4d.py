
import jax
import jax.numpy as jnp
from flax import linen as nn
from functools import partial

from initializers import log_step_initializer
from kernels import s4d_kernel_zoh, causal_convolution
from discretization import discrete_s4d_ssm
from scan import scan_SSM


###############################
# Instantiates a single S4D layer
###############################
def S4DLayerInit(
  A_re_init, A_im_init, N, l_max, imagine, clip_eigs, scaling
):
  return partial(
    S4DLayer, A_re_init=A_re_init, A_im_init=A_im_init, N=N,
    l_max=l_max, imagine=imagine, clip_eigs=clip_eigs, scaling=scaling
  )


###############################
# Flax implementation of S4D Layer
###############################
class S4DLayer(nn.Module):
  A_re_init: jnp.DeviceArray
  A_im_init: jnp.DeviceArray

  N: int
  l_max: int
  imagine: bool=False
  clip_eigs: bool=False
  scaling: str='hippo'
  lr = {
    'A_re': 0.1,
    'A_im': 0.1,
    'B_re': 0.1,
    'B_im': 0.1,
    'log_step': 0.1
  }

  def setup(self):
    if self.scaling=='hippo':
      self.A_re = self.param('A_re', self.A_re_init, (None,))
      self.A_im = self.param('A_im', self.A_im_init, (None,))
    elif self.scaling=='linear':
      self.A_re = self.param('A_re', jax.nn.initializers.constant(-0.5), (self.N,))
      def arange_initializer(scale):
        return lambda key, shape: scale*jnp.ones(shape) * jnp.arange(shape[-1])
      self.A_im = self.param('A_im', arange_initializer(jnp.pi), (self.N,))
    else:
      raise NotImplementedError
    if self.clip_eigs:
      self.A = jnp.clip(self.A_re, None, -1e-4) + 1j*self.A_im
    else:
      self.A = self.A_re + 1j*self.A_im
    self.B_re = self.param('B_re', jax.nn.initializers.ones, (self.N))
    self.B_im = self.param('B_im', jax.nn.initializers.zeros, (self.N))
    self.B = self.B_re + 1j*self.B_im
    self.C = self.param('C', jax.nn.initializers.normal(stddev=0.5**0.5), (self.N, 2))
    self.C = self.C[...,0] + 1j*self.C[...,1]
    self.D = self.param('D', jax.nn.initializers.ones, (1,))
    self.step = jnp.exp(self.param('log_step', log_step_initializer(), (1,)))

    self.K = s4d_kernel_zoh(
      self.C, self.A, self.l_max, self.step
    )
    self.A, self.B, self.C = discrete_s4d_ssm(
      self.C, self.A, self.l_max, self.step
    )
    # RNN Cache
    self.x_k_1 = self.variable("cache", "cache_x_k", jnp.zeros, (self.N,), jnp.complex64)

  def __call__(self, u):
    if not self.imagine:
      return causal_convolution(u, self.K) + self.D*u
    else:
      x_k, y_s = scan_SSM(self.A, self.B, self.C, u[:jnp.newaxis], self.x_k_1.value)
      if self.is_mutable_collection('cache'):
        self.x_k_1.value = x_k
      return y_s.reshape(-1).real + self.D*u

