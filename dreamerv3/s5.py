import jax
import jax.numpy as jnp
from flax import linen as nn
from functools import partial
from jax.nn.initializers import lecun_normal, normal

from . import ninjax as nj

from .discretization import discretize_zoh, discretize_bilinear
from scan import apply_ssm, apply_ssm_state
from initializers import init_VinvB, init_CV, mimo_log_step_initializer, trunc_standard_normal
from initializers import make_DPLR_HiPPO


###############################
# Instantiates a single S5 layer
###############################
def S5LayerInit(
  blocks, init_block_size, Lambda_re_init, Lambda_im_init, V, Vinv,
  H, P, C_init, discretization, dt_min, dt_max, conj_sym, clip_eigs, bidirectional,
  name='ssm'
):
  # H=d_model, P=ssm_size
  return partial(
    S5, blocks=blocks, init_block_size=init_block_size, Lambda_re_init=Lambda_re_init, Lambda_im_init=Lambda_im_init, V=V, Vinv=Vinv,
    H=H, P=P, C_init=C_init, discretization=discretization, dt_min=dt_min, dt_max=dt_max, conj_sym=conj_sym, clip_eigs=clip_eigs, bidirectional=bidirectional,
    name=name)



###############################
# Ninjax implementation of S5 Layer
###############################
class S5(nj.Module):

  def __init__(self, blocks:int, init_block_size:int, 
    Lambda_re_init:jax.Array, Lambda_im_init:jax.Array, V:jax.Array, Vinv:jax.Array, H:int, P:int,
    C_init:str, discretization:str, dt_min:float, dt_max:float, conj_sym:bool=True, clip_eigs:bool=False, bidirectional:bool=False, step_rescale:float=1.0
  ):
    # self.blocks = blocks
    # self.init_block_size = init_block_size
    self.Lambda_re_init = Lambda_re_init
    self.Lambda_im_init = Lambda_im_init
    self.V = V
    self.Vinv = Vinv

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
    
  def __call__(self, batched_input_sequence, init_state):

    Lambda_re = self.get('Lambda_re', lambda rng, shape: self.Lambda_re_init, nj.rng(), (None,))
    Lambda_im = self.get('Lambda_im', lambda rng, shape: self.Lambda_im_init, nj.rng(), (None,))
    if self.clip_eigs:
      Lambda = jnp.clip(Lambda_re, None, -1e-4) + 1j*Lambda_im
    else:
      Lambda = Lambda_re + 1j*Lambda_im

    # Initialize input-to-state matrix B
    local_P = 2*self.P if self.conj_sym else self.P
    B = self.get('B', lambda rng, shape: init_VinvB(lecun_normal(), rng, shape, self.Vinv), nj.rng(), (local_P, self.H))
    B_tilde = B[..., 0] + 1j*B[..., 1]

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
        C_tilde = C[..., 0] + 1j*C[..., 1]
      else:
        C = self.get('C', C_init, nj.rng(), (self.H, self.P, 2))
        C_tilde = C[..., 0] + 1j*C[..., 1]
    else:
      if self.bidirectional:
        C1 = self.get('C1', lambda rng, shape: init_CV(C_init, rng, shape, self.V), nj.rng(), C_shape)
        C2 = self.get('C2', lambda rng, shape: init_CV(C_init, rng, shape, self.V), nj.rng(), C_shape)
        C1 = self.C1[..., 0] + 1j*self.C1[..., 1]
        C2 = self.C2[..., 0] + 1j*self.C2[..., 1]
        C_tilde = jnp.concatenate((C1, C2), axis=-1)
      else:
        C = self.get('C', lambda rng, shape: init_CV(C_init, rng, shape, self.V), nj.rng(), C_shape)
        C_tilde = C[..., 0] + 1j*C[..., 1]
    # Initialize feedthrough matrix
    D = self.get('D', normal(stddev=1.0), nj.rng(), (self.H,))

    self.log_step = self.get('log_step', mimo_log_step_initializer, nj.rng(), (self.P, self.dt_min, self.dt_max))
    step = self.step_rescale * jnp.exp(self.log_step[:, 0])

    # Discretization
    if self.discretization in ['zoh']:
      Lambda_bar,  B_bar = discretize_zoh(Lambda, B_tilde, step)
    elif self.discretization in ['bilinear']:
      Lambda_bar,  B_bar = discretize_bilinear(Lambda, B_tilde, step)
    else:
      raise NotImplementedError(f'Discretization method {self.discretization} not implemented')

    state = jnp.ones((batched_input_sequence.shape[0], batched_input_sequence.shape[1], Lambda_bar.shape[0]), dtype=jnp.complex64)
    ys, xs = jax.vmap(
      lambda x, u: apply_ssm_state(x, u, Lambda_bar, B_bar, C_tilde, self.conj_sym, self.bidirectional),
      in_axes=(0,0), out_axes=(0))(state, batched_input_sequence)
    Du = jax.vmap(
      lambda u: D*u,
      in_axes=0, out_axes=0)(batched_input_sequence)
    out, state = ys + Du, xs

    return out, state



  def step(self, input, prev_state):

    # Lambda_re = self.get('Lambda_re', lambda rng, shape: self.Lambda_re_init, nj.rng(), (None,))
    # Lambda_im = self.get('Lambda_im', lambda rng, shape: self.Lambda_im_init, nj.rng(), (None,))
    Lambda_re = self.get('Lambda_re')
    Lambda_im = self.get('Lambda_im')
    B_tilde = self.get('B')[..., 0] + 1j*self.get('B')[..., 1]
    step = self.step_rescale * jnp.exp(self.get('log_step')[:, 0])
    if self.clip_eigs:
      Lambda = jnp.clip(Lambda_re, None, -1e-4) + 1j*Lambda_im
    else:
      Lambda = Lambda_re + 1j*Lambda_im
    if self.discretization in ['zoh']:
      Lambda_bar,  B_bar = discretize_zoh(Lambda, B_tilde, step)
    elif self.discretization in ['bilinear']:
      Lambda_bar,  B_bar = discretize_bilinear(Lambda, B_tilde, step)
    else:
      raise NotImplementedError(f'Discretization method {self.discretization} not implemented')
    C_tilde = self.get('C')[..., 0] + 1j*self.get('C')[..., 1]
    D = self.get('D')

    # x_k = jax.vmap(
    #   lambda prev_state, u: Lambda_bar @ prev_state + B_bar @ u)(prev_state, input)
    # y_k = jax.vmap(
    #   lambda x_k, u: (C_tilde @ x_k).real + D * u,
    #   in_axes=0, out_axes=0)(x_k, input)
    # y_k = (C_tilde @ x_k).real + D * input

    x_k = Lambda_bar * prev_state + jax.vmap(lambda u: B_bar @ u)(input)
    y_k = jax.vmap(lambda x: (C_tilde @ x).real)(x_k) + D*input
    return y_k, x_k
