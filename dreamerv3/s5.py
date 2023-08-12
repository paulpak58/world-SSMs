
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import lecun_normal
from functools import partial
import ninjax as nj


from discretization import discretize_zoh, discretize_bilinear
from scan import apply_ssm
from initializers import init_VinvB, init_CV, mimo_log_step_initializer, trunc_standard_normal


###############################
# Instantiates a single S5 layer
###############################
def S5LayerInit(
  H, P, Lambda_re_init, Lambda_im_init, V, Vinv, C_init, discretization, dt_min, dt_max, conj_sym, clip_eigs, bidirectional
):
  # H=d_model, P=ssm_size
  return partial(
    S5Layer, H=H, P=P, Lambda_re_init=Lambda_re_init, Lambda_im_init=Lambda_im_init,
    V=V, Vinv=Vinv, C_init=C_init, discretization=discretization, dt_min=dt_min, dt_max=dt_max,\
    conj_sym=conj_sym, clip_eigs=clip_eigs, bidirectional=bidirectional
  )


###############################
# Flax implementation of S5 Layer
###############################
class S5Layer(nn.Module):
  Lambda_re_init: jnp.DeviceArray # (P, )
  Lambda_im_init: jnp.DeviceArray # (P, )
  V: jnp.DeviceArray  # (P, P)
  Vinv: jnp.DeviceArray # (P, P)

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
    local_P = 2*self.P if self.conj_sym else self.P
    self.Lambda_re = self.param('Lambda_re', lambda rng, shape: self.Lambda_re_init, (None,))
    self.Lambda_im = self.param('Lambda_im', lambda rng, shape: self.Lambda_im_init, (None,))
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
    self.x_k_1 = self.variable('cache', 'cache_x_k', jnp.zeros, (self.P,), jnp.complex64)


  def step(self, input, prev_state):
    x_k = self.Lambda_bar @ prev_state + self.B_bar @ input
    y_k = self.C_tilde @ x_k + self.D * input
    return y_k, x_k


  def __call__(self, input_sequence):
    ys, state = apply_ssm(self.Lambda_bar, self.B_bar, self.C_tilde, input_sequence, self.conj_sym, self.bidirectional)
    if self.is_mutable_collection('cache'):
      self.x_k_1.value = state
    Du = jax.vmap(lambda u: self.D*u)(input_sequence)
    out = ys + Du
    return out, state
  