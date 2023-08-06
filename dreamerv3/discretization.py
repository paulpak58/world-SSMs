# Revised and inspired from https://github.com/lindermanlab/S5/blob/liquid/s5/ssm.py
# Revision author: Paul Pak

import jax
import jax.numpy as jnp


#############################################
# Discretize diagonalized, continuous-time SSM
# using bilinear transform
#############################################
def discretize_bilinear(Lambda, B_tilde, Delta):
  # Lambda: Diagonal state matrix (P,)
  Identity = jnp.ones(Lambda.shape[0])
  # B_tilde: Input matrix (P, H), Delta: Time step (P, )
  BL = 1/(Identity-(Delta/2.0)*Lambda)
  # Discretized diagonal state matrix (P, )
  Lambda_bar = (Identity+(Delta/2.0)*Lambda)*BL
  # Discretized input matrix (P, H)
  B_bar = (BL*Delta)[..., None]*B_tilde
  return Lambda_bar, B_bar


#############################################
# Discretize diagonalized, continuous-time SSM
# using zero-order hold
#############################################
def discretize_zoh(Lambda, B_tilde, Delta):
  # Lambda: Diagonal state matrix (P,)
  Identity = jnp.ones(Lambda.shape[0])
  # Lambda_bar: Discretized diagonal state matrix (P, )
  Lambda_bar = jnp.exp(Delta*Lambda)
  # B_bar: Discretized input matrix (P, H)
  B_bar = (1/Lambda*(Lambda_bar-Identity))[..., None]*B_tilde
  return Lambda_bar, B_bar


#############################################################
# Discretization of the diagonal plus low-rank SSM
# Used for S4 which is different from S5, which drops the low-rank terms
#############################################################
def discrete_DPLR(Lambda, P, Q, B, C, step, L):
  # Convert parameters to matrices
  B = B[:, jnp.newaxis]
  Ct = C[jnp.newaxis, :]
  N = Lambda.shape[0]
  A = jnp.diag(Lambda) - P[:, jnp.newaxis] @ Q[:, jnp.newaxis].conj().T
  I = jnp.eye(N)
  # Forward Euler
  A0 = (2.0 / step) * I + A
  # Backward Euler
  D = jnp.diag(1.0 / ((2.0 / step) - Lambda))
  Qc = Q.conj().T.reshape(1, -1)
  P2 = P.reshape(-1, 1)
  A1 = D - (D @ P2 * (1.0 / (1 + (Qc @ D @ P2))) * Qc @ D)
  # A bar and B bar
  Ab = A1 @ A0
  Bb = 2 * A1 @ B
  # Recover Cbar from Ct
  Cb = Ct @ jnp.linalg.inv(I - jnp.linalg.matrix_power(Ab, L)).conj()
  return Ab, Bb, Cb.conj()


####################################################
# Discretized S4D with simplified bilinear and zoh transforms
####################################################
def discrete_s4d_ssm(C, A, l_max, step, mode='zoh'):
  N = A.shape[0]
  if mode=="bilinear":
      num, denom = 1 + .5 * step*A, 1 - .5 * step*A
      Abar = num/denom
      Bbar = step*jnp.ones(N)/denom
  elif mode == "zoh":
      Abar = jnp.exp(step*A)
      Bbar = (jnp.exp(step*A)-1)/A * jnp.ones(N)
  Abar = jnp.diag(Abar)
  Bbar = Bbar.reshape(N, 1)
  Cbar = C.reshape(1, N)
  return Abar, Bbar, Cbar


###########################
# Discretized DSS simplified
###########################
def discrete_dss_ssm(W, Lambda, L, step):
  N = Lambda.shape[0]
  Abar = jnp.diag(jnp.exp(Lambda * step))
  b = jax.vmap(lambda l: 1 / (l * (jnp.exp(l * jnp.arange(L) * step)).sum()))
  Bbar = b(Lambda).reshape(N, 1)
  Cbar = W.reshape(1, N)
  return Abar, Bbar, Cbar
