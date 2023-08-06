import jax
import jax.numpy as jnp


###########################################
# Cauchy matrix multiplication: (n), (l), (n) -> (l)
##########################################
@jax.jit
def cauchy(v, omega, lambd):
  cauchy_dot = lambda _omega: (v/(_omega-lambd)).sum()
  return jax.vmap(cauchy_dot)(omega)

######################################
# DPLR kernel from the AnnotatedS4 (S4)
######################################
def kernel_DPLR(Lambda, P, Q, B, C, step, L):
  # Evaluate at roots of unity
  # Generating function is (-)z-transform, so we evaluate at (-)root
  Omega_L = jnp.exp((-2j * jnp.pi) * (jnp.arange(L) / L))

  aterm = (C.conj(), Q.conj())
  bterm = (B, P)

  g = (2.0 / step) * ((1.0 - Omega_L) / (1.0 + Omega_L))
  c = 2.0 / (1.0 + Omega_L)

  # Reduction to core Cauchy kernel
  k00 = cauchy(aterm[0] * bterm[0], g, Lambda)
  k01 = cauchy(aterm[0] * bterm[1], g, Lambda)
  k10 = cauchy(aterm[1] * bterm[0], g, Lambda)
  k11 = cauchy(aterm[1] * bterm[1], g, Lambda)
  atRoots = c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)
  out = jnp.fft.ifft(atRoots, L).reshape(L)
  return out.real


##########################################################
# Non-circular convolution using convoluion theorem and FFTs  
##########################################################
def causal_convolution(u, K, nofft=False):
  if nofft:
    return jax.scipy.signal.convolve(u, K, mode='full')[:u.shape[0]]
  else:
    assert K.shape[0]==u.shape[0]
    ud = jnp.fft.rfft(jnp.pad(u, (0, K.shape[0])))
    Kd = jnp.fft.rfft(jnp.pad(K, (0, u.shape[0])))
    out = ud*Kd
    return jnp.fft.irfft(out)[:u.shape[0]]


######################################################
# Discretization and vandermonde product optimized [S4D]
######################################################
def s4d_kernel_zoh(C, A, L, step):
  kernel_l = lambda l: (C*(jnp.exp(step*A)-1)/A * jnp.exp(l*step*A)).sum()
  return jax.vmap(kernel_l)(jnp.arange(L)).real


##################################
# Stable softmax over complex inputs
##################################
def complex_softmax(x, eps=1e-7):
  def reciprocal(x):
      return x.conj()/(x * x.conj() + eps)
  e = jnp.exp(x - x[jnp.argmax(x.real)])
  return e*reciprocal(jnp.sum(e))


#######################################
# DSS kernel with complex softmax helper
#######################################
def dss_kernel(W, Lambda, L, step):
  P = (step * Lambda)[:, None] * jnp.arange(L)
  # Taking row softmax is a lot easier with vmapping over all of P
  S = jax.vmap(complex_softmax)(P)
  return ((W / Lambda) @ S).ravel().real