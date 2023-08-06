import jax
import jax.numpy as jnp
from flax import linen as nn
from functools import partial

from initializers import log_step_initializer
from discretization import discrete_dss_ssm
from kernels import dss_kernel, causal_convolution
from scan import scan_SSM


###############################
# Flax implementation of DSS Layer
###############################
class DSSLayer(nn.Module):
  Lambda_re_init: jnp.DeviceArray
  Lambda_im_init: jnp.DeviceArray
  P_init: jnp.DeviceArray
  B_init: jnp.DeviceArray

  N: int
  l_max: int
  imagine: bool=False # True is rnn-mode

  def setup(self):
    self.Lambda_re = self.param('Lambda_re', lambda rng, shape: self.Lambda_re_init, (None,))
    self.Lambda_im = self.param('Lambda_im', lambda rng, shape: self.Lambda_im_init, (None,))
    self.Lambda = self.Lambda_re + 1j*self.Lambda_im
    self.W = self.param('W', jax.nn.initializers.normal(stddev=0.5**0.5), (1, self.N, 2))
    self.W = self.W[...,0] + 1j*self.W[...,1]
    self.D = self.param('D', nn.initializers.ones, (1,))
    self.step = jnp.exp(self.param('log_step', log_step_initializer, (1,)))

    self.K = dss_kernel(
      self.W, self.Lambda, self.l_max, self.step
    )
    self.Lambda, self.B, self.C = discrete_dss_ssm(
      self.W, self.Lambda, self.l_max, self.step
    )
    # RNN Cache
    self.x_k = self.variable('cache', 'cache_x_k', jnp.zeros, (self.N,), jnp.complex64)

    def __call__(self, u):
      if not self.decode:
        return causal_convolution(u, self.K) + self.D*u
      else:
        x_k, y_s = scan_SSM(*self.ssm, u[:, None], self.x_k.value)
        if self.is_mutable_collection('cache'):
          self.x_k.value = x_k
        return y_s.reshape(-1).real + self.D*u
