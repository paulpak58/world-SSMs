import jax
import jax.numpy as jnp


############################
# Creates HiPPO-LegS matrix
############################
def _make_HiPPO(N):
  # N is the state size
  P = jnp.sqrt(1 + 2*jnp.arange(N))
  A = P[:, jnp.newaxis]*P[jnp.newaxis, :]
  A = jnp.tril(A) - jnp.diag(jnp.arange(N))
  # (N, N) HiPPO LegS matrix
  return -A


############################################
# Creates NPLR Representation of HiPPO-LegS matrix
############################################
def _make_NPLR_HiPPO(N):
  # N is the state size
  hippo = _make_HiPPO(N)
  P = jnp.sqrt(jnp.arange(N) + 0.5)
  B = jnp.sqrt(2*jnp.arange(N) + 1.0)
  # (N, N) HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B
  return hippo, P, B

# Verify that this returns the same outputs as the implementation above
def _make_S4_NPLR_HiPPO(N):
   nhippo = _make_HiPPO(N)
   # Add in rank 1 term
   p = 0.5*jnp.sqrt(2*jnp.arange(1,N+1)+1.0)
   q = 2*p
   S = nhippo + p[:,jnp.newaxis] * q[jnp.newaxis,:]
   # Diagonalize to S to V\LambdaV^*
   Lambda, V = jax.jit(jnp.linalg.eig, backend='cpu')(S)
   return nhippo, Lambda, p, q, V


############################################
# Creates DPLR Representation of HiPPO-LegS matrix
############################################
def make_DPLR_HiPPO(N):
  # N is the state size
  A, P, B = _make_NPLR_HiPPO(N)
  S = A + P[:, jnp.newaxis]*P[jnp.newaxis, :]
  S_diag = jnp.diagonal(S)
  Lambda_real = jnp.mean(S_diag) + jnp.ones_like(S_diag)
  # Diagonalize S to V Lambda V^*
  Lambda_imag, V = jnp.linalg.eigh(S*-1j)
  P = V.conj().T@P
  B_orig = B
  B = V.conj().T@B
  # Eigenvalues Lambda, low-rank factor P, conjugated HiPPO input matrix B, eigenvectors V, original HiPPO input matrix B
  return Lambda_real + 1j*Lambda_imag, P, B, V, B_orig


############################
# SSM Initilization helper
############################
def trunc_standard_normal(key, shape):
    H, P, _ = shape
    Cs = []
    for i in range(H):
        key, skey = random.split(key)
        # sample C matrix
        C = lecun_normal()(skey, (1, P, 2))                                         
        Cs.append(C)
    # (H, P, 2)
    return jnp.array(Cs)[:, 0]




#################################################
# Initialize C_tilde=CV. First sample C, then compute CV
#################################################
def init_CV(init_fun, rng, shape, V):
    C_ = init_fun(rng, shape)
    C = C_[..., 0] + 1j * C_[..., 1]
    CV = C @ V
    # C_tilde (complex64) shape (H, P, 2)
    return jnp.concatenate((CV.real[..., None], CV.imag[..., None]), axis=-1)


########################
# Initialize VinvB=VinvB
########################
def init_VinvB(rng, shape, Vinv):
  # Desired shape of matrix (P, H)
  B = jax.nn.initializers.lecun_normal(rng, shape)
  Vinv = Vinv@B
  return jnp.concatenate([Vinv.real[..., None], Vinv.imag[..., None]], axis=-1)   # (P, H, 2)


#################################################
# S5 discretization for multi-dimensional input u
#################################################
def mimo_log_step_initializer(key, input):
  H, dt_min, dt_max = input
  log_steps = []
  for _ in range(H):
      key, skey = jax.random.split(key)
      log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(skey, shape=(1,))
      log_steps.append(log_step)
  return jnp.array(log_steps) # (H,)

#######################################
# S4/DSS discretization in the SISO case
# The step size is learned in log space
#######################################
def log_step_initializer(dt_min=0.001, dt_max=0.1):
  def init(key, shape):
    return jax.random.uniform(key,shape) * \
      (jnp.log(dt_max)-jnp.log(dt_min)) + jnp.log(dt_min)
  return init




###################################################
# Lambda real component from DPLR HiPPO-LegS matrix
##################################################
def lambda_re_initializer(shape):
  blocks, block_size = shape
  Lambda, _, _, _, _ = make_DPLR_HiPPO(block_size)
  Lambda = Lambda[:block_size]
  Lambda = (Lambda*jnp.ones((blocks, block_size))).ravel()
  return Lambda.real


###################################################
# Lambda imaginary component from DPLR HiPPO-LegS matrix
##################################################
def lambda_imag_initializer(shape):
  blocks, block_size = shape
  Lambda, _, _, _, _ = make_DPLR_HiPPO(block_size)
  Lambda = Lambda[:block_size]
  Lambda = (Lambda*jnp.ones((blocks, block_size))).ravel()
  return Lambda.imag
