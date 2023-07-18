import jax
import jax.numpy as jnp
from s5_utils import make_DPLR_HiPPO, trunc_standard_normal


########################
# Initialize C_tilde=CV
########################
def CV_initializer(C_init, rng, shape, V):
  if C_init=='trunc_standard_normal':
    C_func = trunc_standard_normal(rng, shape)
  elif C_init=='lecun_normal':
    C_func = jax.nn.initializers.lecun_normal(rng, shape)
  elif C_init=='complex_normal':
    C_func = jax.nn.initializers.normal(stddev=0.5**0.5)(rng, shape)
  else:
    raise NotImplementedError(f'C_init method {C_init} not implemented')

  if C_init=='complex_normal':
    return C_func
  else:
    # Desired shape of matrix (H, P)
    C = C_func[..., 0] + 1j*C_func[..., 1]
    # V composes the eigenvectors of the diagonalized state matrix
    CV = C@V
    return jnp.concatenate([CV.real[..., None], CV.imag[..., None]], axis=-1) # (H, P, 2)


########################
# Initialize VinvB=VinvB
########################
def VinvB_initializer(rng, shape, Vinv):
  # Desired shape of matrix (P, H)
  B = jax.nn.initializers.lecun_normal(rng, shape)
  Vinv = Vinv@B
  return jnp.concatenate([Vinv.real[..., None], Vinv.imag[..., None]], axis=-1)   # (P, H, 2)


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