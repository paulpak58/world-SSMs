import math
import jax
import jax.numpy as jnp
from flax import linen as nn
from functools import partial
import itertools

from initializers import log_step_initializer
from discretization import discrete_DPLR
from kernels import kernel_DPLR, causal_convolution
from scan import scan_SSM


###############################
# Instantiates a single S4 layer
###############################
def LiquidS4LayerInit(
):
  raise NotImplementedError



######################################
# Flax implementation of Liquid-S4 Layer
######################################
class LiquidS4Layer(nn.Module):
  Lambda_re_init: jnp.DeviceArray
  Lambda_im_init: jnp.DeviceArray
  P_init: jnp.DeviceArray
  B_init: jnp.DeviceArray

  N: int
  l_max: int
  imagine: bool=False
  clip_eigs: bool=False
  liquid: int=0
  allcombs: bool=True
  allcombs_idx_cache: None

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
    
    # All combinations index cache for liquid correlations
    # self.all_combs_index_cache = self.variable('cache', 'cache_all_combs_index', jnp.zeros((self.N, self.N, self.N, self.N)), jnp.int32)
    

  def __call__(self, u):
    if not self.imagine:
      y_s = causal_convolution(u, self.K) + self.D*u
    else:
      x_k, y_s = scan_SSM(self.A, self.B, self.C, u[:jnp.newaxis], self.x_k_1.value)
      if self.is_mutable_collection('cache'):
        self.x_k_1.value = x_k
      y_s = y_s.reshape(-1).real + self.D*u

    seq_len = int(y_s.shape[0])
    if self.allcombs_idx_cache is None:
      self.allcombs_idx_cache = []
      for p in range(2, self.liquid+2):
        selected_count = 1
        for n in range(2, seq_len):
          count = math.comb(n,p)
          if count>=seq_len:
            selected_count = n
            break
        idxs = list(itertools.combinations(range(seq_len-selected_count, seq_len), p))
        # Select the right amount to match the sequence length
        if len(idxs) != seq_len:
          idxs = idxs[-seq_len:]
        self.allcombs_idx_cache.append(jnp.array(idxs))

    us = u
    for i in range(self.liquid):
      if self.all_combs:
        p, idxs = self.allcombs_idx_cache[i]
        us = u[..., idxs[:, 0]]
        for j in range(1,p):
          us = us*u[..., idxs[:, j]]
        if us.size(-1) != u.size(-1):
          us = jnp.pad(us, ((0,0), (0, u.size(-1)-us.size(-1))), mode='constant') # TODO: check padding
      else:
        us_shift = jnp.pad(us[..., :-1], ((0,0),(1,0)), mode='constant')
        us = us*us_shift
      y_s = y_s 
