import jax
import jax.numpy as jnp
from functools import partial  
from flax import linen as nn

from initializers import log_step_initializer
from dreamerv3.discretization import discretize_zoh, discretize_bilinear, discrete_DPLR
from convolutions import kernel_DPLR, causal_convolution
from scan import scan_SSM


def S4LayerInit(N):
  _, Lambda, p, q, V = _make_S4_NPLR_HiPPO(N)
  Vc = V.conj().T
  p = Vc @ p
  q = Vc @ q.conj()
  A = jnp.diag(Lambda) - p[:,jnp.newaxis] @ q[:, jnp.newaxis].conj().T
  return partial(S4Layer, N=N, A=A, Lambda=Lambda, p=p, q=q, Vc=Vc)



class SSMLayer(nn.Module):
  N: int
  l_max: int
  decode: bool=False

  def setup(self):
    self.A = self.param('A', jax.nn.initializers.lecun_normal(), (self.N, self.N))
    self.B = self.param('B', jax.nn.initializers.lecun_normal(), (self.N, 1))
    self.C = self.param('C', jax.nn.initializers.lecun_normal(), (1, self.N))
    self.D = self.param('D', jax.nn.initializers.ones(), (1,))

    self.log_step = self.param('log_step', log_step_initializer(), (1,))
    step = jnp.exp(self.log_step)


    if self.discretization in ['zoh']:
      self.A, self.B = discretize_zoh(self.A, self.B, step)
    elif self.discretization in ['bilinear']:
      self.A, self.B = discretize_bilinear(self.A, self.B, step)
    else:
      raise NotImplementedError(f'Discretization method {self.discretization} not implemented')

    self.K = K_conv(self.A, self.B, self.C, self.l_max)


class S4Layer(nn.Module):
  Lambda_re_init: jnp.DeviceArray
  Lambda_im_init: jnp.DeviceArray
  P_init: jnp.DeviceArray
  B_init: jnp.DeviceArray

  N: int
  l_max: int
  imagine: bool=False

  clip_eigs: bool=False

  lr = {
    'Lambda_re': 0.1,
    'Lambda_im': 0.1,
    'P': 0.1,
    'B': 0.1,
    'log_step': 0.1
  }

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

        



class S4Layer(nn.Module):
  A: jnp.DeviceArray
  Vc: jnp.DeviceArray
  p: jnp.DeviceArray
  q: jnp.DeviceArray
  Lambda: jnp.DeviceArray
  N: int
  l_max: int
  decode: bool=False

  def setup(self):
    self.Ct = self.param('C1', jax.nn.initializers.lecun_normal(), (1, self.N, 2))
    self.Ct = self.Ct[...,0] + self.Ct[...,1]*1
    self.B = self.param('B', jax.nn.initializers.lecun_normal(), (self.N, 1))
    self.B = self.Vc @ self.B
    self.step = jnp.exp(self.param('log_step'), log_step_initializer(), (1,))
    if not self.decode:
      K_gen = K_gen_DPLR(
        self.Lambda,
        self.p,
        self.q,
        self.B,
        self.Ct,
        self.step[0],
        unmat=self.l_max>1000
      )
      self.K = conv_from_gen(K_gen, self.l_max)
    else:
      def init_discrete():
        return discrete_DPLR(
          self.Lambda,
          self.p,
          self.q,
          self.B,
          self.Ct,
          self.step[0],
          self.l_max
        )
      ssm_var = self.variable('prime', 'ssm', init_discrete)
      if self.is_mutable_collection('prime'):
        ssm_var.value = init_discrete()
      self.ssm = ssm_var.value

      self.x_k_1 = self.variable('cache', 'cache_x_k', jnp.zeros, (self.N,), jnp.complex64)

  def __call__(self, x):
    # This is identical to SSM Layer
    if not self.decode:
      # CNN Mode
      return non_circular_convolution(u, self.K) + self.D * u
    else:
      # RNN Mode
      x_k, y_s = scan_SSM(*self.ssm, u[:, np.newaxis], self.x_k_1.value)
      if self.is_mutable_collection("cache"):
          self.x_k_1.value = x_k
      return y_s.reshape(-1).real + self.D * u

