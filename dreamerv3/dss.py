import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import lecun_normal
from functools import partial
import ninjax as nj

from ssm_utils import _make_S4_NPLR_HiPPO, make_DPLR_HiPPO, trunc_standard_normal, \
  discretize_zoh, discretize_bilinear
from initializers import init_VinvB, init_CV, init_log_steps

rng = jax.random.PRNGKey(15)



# Stable softmax over complex inputs
def complex_softmax(x, eps=1e-7):
  def reciprocal(x):
      return x.conj()/(x * x.conj() + eps)
  e = jnp.exp(x - x[jnp.argmax(x.real)])
  return e*reciprocal(jnp.sum(e))

def dss_kernel(W, Lambda, L, step):
  P = (step * Lambda)[:, None] * np.arange(L)
  # Taking row softmax is a lot easier with vmapping over all of P
  S = jax.vmap(complex_softmax)(P)
  return ((W / Lambda) @ S).ravel().real

def dss_ssm(W, Lambda, L, step):
  N = Lambda.shape[0]
  Abar = jnp.diag(jnp.exp(Lambda * step))
  b = jax.vmap(lambda l: 1 / (l * (jnp.exp(l * jnp.arange(L) * step)).sum()))
  Bbar = b(Lambda).reshape(N, 1)
  Cbar = W.reshape(1, N)
  return Abar, Bbar, Cbar

def log_step_initializer(dt_min=0.001, dt_max=0.1):
  def init(key, shape):
    return jax.random.uniform(key,shape) * \
      (jnp.log(dt_max)-jnp.log(dt_min)) + jnp.log(dt_min)
  return init


def discrete_DPLR(Lambda, p, q, B, Ct, step, L):
  N = Lambda.shape[0]
  A = jnp.diag(Lambda) - p[:, jnp.newaxis] @ q[:, jnp.newaxis].conj().T
  I = jnp.eye(N)

  # Forward Euler
  A0 = (2.0/step)*I + A

  # Backward Euler
  D = jnp.diag(1.0 / ((2.0/step) - Lambda))
  qc = q.conj().T.reshape(1, -1)
  p2 = p.reshape(-1, 1)
  A1 = D - (D @ p2 * (1.0 / (1 + (qc @ D @ p2))) * qc @ D)

  # A bar and B bar
  Ab = A1 @ A0
  Bb = 2*A1 @ B

  # Recover Cbar from Ct
  Cb = Ct @ jnp.linalg.inv(I - jnp.linalg.matrix_power(Ab, L)).conj()
  return Ab, Bb, Cb.conj()



def scan_ssm(A, B, C, u, x0):
  # step similar to RNN step
  def step(x_k_1, u_k):
    x_k = A @ x_k_1 + B @ u_k
    y_k = C @ x_k
    return x_k, y_k
  return jax.lax.scan(step, x0, u)
  '''
  I like the for-loop skeletion in https://github.com/ChanBong/s4-jax

  result = x[0]
  x_k, y_k = 0, 0
  for i in range(u):
    x_k = A @ x[i] + B @ u[i]
    y_k = C @ x[i]
    result.append(x_k)
  final = x_k
  return final, result
  '''
 

def S4LayerInit(N):
  _, Lambda, p, q, V = _make_S4_NPLR_HiPPO(N)
  Vc = V.conj().T
  p = Vc @ p
  q = Vc @ q.conj()
  A = jnp.diag(Lambda) - p[:,jnp.newaxis] @ q[:, jnp.newaxis].conj().T
  return partial(S4Layer, N=N, A=A, Lambda=Lambda, p=p, q=q, Vc=Vc)


def DSSLayerInit(N):
  _, Lambda, _, _, _= _make_S4_NPLR_HiPPO(2*N)
  Lambda = Lambda[jnp.nonzero(Lambda.imag>0, size=N)]
  return partial(DSSLayer, N=N, Lambda=Lambda)


def S5LayerInit(N, d_model, block_size, ssm_size, blocks, conj_sym):
  Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)
  if conj_sym:
    block_size = block_size//2
    ssm_size = ssm_size//2
  Lambda = Lambda[:block_size]
  V = V[:, :block_size]
  Vc = V.conj().T

  # Reshape matrices
  Lambda = (Lambda*jnp.ones((blocks,block_size))).ravel()
  V = jax.scipy.linalg.block_diag(*([V]*blocks))
  Vinv = jax.scipy.linalg.block_diag(*([Vc]*blocks))

  return partial(
    S5Layer, H=d_model, P=ssm_size, Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
    V=V, Vinv=Vinv, C_init=C_init, discretization=discretization, dt_min=dt_min, dt_max=dt_max,\
    conj_sym=conj_sym, clip_eigs=clip_eigs, bidirectional=bidirectional, step=#TODO
  )

class S5Layer(nj.Module):

  def __init__(
    self, H, P, Lambda_re_init, Lambda_im_init, V, Vinv, C_init, discretization, dt_min, dt_max,
    conj_sym=True, clip_eigs=False, bidirectional=False, step_rescale=1.0
  ):
    self.H = H
    self.P = P
    self.Lambda_re_init = Lambda_re_init
    self.Lambda_im_init = Lambda_im_init
    self.V = V
    self.Vinv = Vinv

    self.C_init = C_init
    self.discretization = discretization
    self.dt_min = dt_min
    self.dt_max = dt_max
    self.conj_sym = conj_sym
    self.clip_eigs = clip_eigs
    self.bidirectional = bidirectional
    self.step_rescale = step_rescale

  ##############################
  # Initialize complex matrices
  ##############################
  def forward_scan(self):
    # account for when a real B and C are sampled and then multiplied by half-sized Vinv
    local_P = 2*self.P if self.conj_sym else self.P

    # Initialize diagonal state to state matrix Lambda
    Lambda_re = self.get('Lambda_real', self.Lambda_re_init, (None, ))
    Lambda_imag = self.get('Lambda_im', self.Lambda_im_init, (None, ))
    if self.clip_eigs:
      Lambda = jnp.clip(Lambda_re, None, -1e-4) + 1j*Lambda_imag
    else:
      Lambda = Lambda_re + 1j*Lambda_imag

    # Input to state matrix (B)
    B = self.get('B', init_VinvB, nj.rng(), (local_P, self.H), self.Vinv)
    B_tilde = B[..., 0] + 1j*B[..., 1]

    # State to output matrix (C)
    if self.C_init=='trunc_standard_normal' or self.C_init=='lecun_normal':
      C_init = trunc_standard_normal if self.C_init=='trunc_standard_normal' else jax.nn.initializers.lecun_normal()
      C_shape = (self.H, local_P, 2)
      if self.bidirectional:
        C1 = self.get('C1', init_CV, C_init, nj.rng(), C_shape, self.V)
        C1 = self.get('C2', init_CV, C_init, nj.rng(), C_shape, self.V)
        C1 = C1[..., 0] + 1j * C1[..., 1]
        C2 = C2[..., 0] + 1j * C2[..., 1]
        C_tilde = jnp.concatenate((C1, C2), axis=-1)
      else:
        C = self.get('C', init_CV, C_init, nj.rng(), C_shape, self.V)
        C_tilde = C[..., 0] + 1j*C[..., 1]
    elif self.C_init=='complex_normal':
      C_init = jax.nn.initializers.normal(stddev=0.5**0.5)
      if self.bidirectional:
        C = self.get('C', C_init, nj.rng(), (self.H, 2*self.P, 2))
      else:
        C = self.get('C', C_init, nj.rng(), (self.H, self.P, 2))
      C_tilde = C[..., 0] + 1j*C[..., 1]
    else:
      raise NotImplementedError(f'C_init method {self.C_init} not implemented')
    
    # Feedthrough matrix (D)
    D = self.get('D', jax.nn.initializers.normal(stddev=1.0), nj.rng(), (self.H,))

    # Learnable discretization timescale value
    log_step = self.get('log_step', init_log_steps, nj.rng(), (self.P, self.dt_min, self.dt_max))
    step = self.step_rescale * jnp.exp(log_step[:, 0])
    if self.discretization=='zoh':
      Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
    elif self.discretization=='bilinear':
      Lambda_bar, B_bar = discretize_bilinear(Lambda, B_tilde, step)
    else:
      raise NotImplementedError(f'Discretization method {self.discretization} not implemented')
    

    Lambda_elements = self.Lambda_bar * jnp.ones((input_sequence.shape[0], self.Lambda_bar.shape[0]))
    # B_bar: Discretized input matrix (P, H)
    Bu_elements = jax.vmap(lambda u: self.B_bar @ u)(input_sequence)
    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))
    if self.conj_sym:
        ys = jax.vmap(lambda x: 2*(C_tilde @ x).real)(xs)
    else:
        ys = jax.vmap(lambda x: (C_tilde @ x).real)(xs) # (L,H)
    Du = jax.vmap(lambda u: self.D * u)(input_sequence)
    return ys + Du, xs
    

 
    
class S4Layer(nj.Module):
  def __init__(
        self, N, A, Lambda, p, q, Vc)
  ):
    self.N = N
    self.A = A
    self.Lambda = Lambda
    self.p = p
    self.q = q
    self.Vc = Vc

  def forward_scan(self):
    Ct = self.get('Ct', jax.nn.initializers.lecun_normal(), (1, self.N, 2))
    C = Ct[..., 0] + 1j*Ct[..., 1]

    B = self.get('B', jax.nn.initializers.lecun_normal(), (self.N, 1))
    Bv = self.Vc @ B

    step = self.get('step', log_step_initializer(), (1,))
    exp_step = jnp.exp(step)

    D = self.get('D', jax.nn.initializers.uniform(), (1,))

    def init_discrete():
      return discrete_DPLR(
        self.Lambda, self.p, self.q, Bv, Ct, exp_step, self.L
      )

    # ssm = self.get('ssm', init_discrete())
    states = self.get('states', jax.nn.initializers.zeros(), (self.N,), dtype=jnp.complex64)

    x_k, y_s = scan_ssm()
  

class DSSLayer(nj.Module):
  def __init__(self, N, l_max, decode=False):
    self.N = N
    self.l_max = l_max
    self.decode = decode

  def initialize(self):
    W = self.get('W', lecun_normal(), (1, self.N, 2))
    D = self.get('D', jax.nn.initializers.ones, (1,))
    step = self.get('step', log_step_initializer, (1,))
    
  def forward_scan(self):
    W = self.get('W')
    W = self.W[..., 0] + 1j*self.W[..., 1]
    if not self.decode:
      raise NotImplementedError
    else:
      x_k, y_s = scan_ssm()
      out = y_s.reshape(-1).real + self.get('D')*u
