import jax
import jax.numpy as jnp
from s5_utils import make_DPLR_HiPPO, trunc_standard_normal


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


####################################
# Discretization Initilization helper
###################################
def init_log_steps(key, input):
  H, dt_min, dt_max = input
  dt_min = 0.001 if dt_min is None else dt_min
  dt_max = 0.1 if dt_max is None else dt_max
  log_steps = []
  for i in range(H):
      key, skey = jax.random.split(key)
      log_step = jax.random.uniform(skey, (1,))**(jnp.log(dt_max) - jnp.log(dt_min)) + jnp.log(dt_min)
      log_steps.append(log_step)
  # (H,)
  return jnp.array(log_steps)


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