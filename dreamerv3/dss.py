import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import lecun_normal
from functools import partial
import ninjax as nj

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
  


class S4Layer(nj.Module):
  def __init__(self, N):
    self.N = N

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
    ssm = self.get('ssm', init_discrete())
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
