import re
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from . import jaxutils
from . import ninjax as nj
from . import nets
# from nets import Linear


f32 = jnp.float32
tfd = tfp.distributions
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute


import warnings
warnings.filterwarnings("ignore", category=jnp.ComplexWarning)


#############################
# HiPPO Initializer for S5
#############################
def _make_HiPPO(N):
  # N is state size, Returns NxN LegS matrix
  P = np.sqrt(1 + 2 * np.arange(N))
  A = P[:, np.newaxis] * P[np.newaxis, :]
  A = np.tril(A) - np.diag(np.arange(N))
  return -A

############################################
# Creates NPLR Representation of HiPPO-LegS matrix
############################################
def _make_NPLR_HiPPO(N):
  # N is the state size
  hippo = make_HiPPO(N)
  P = jnp.sqrt(jnp.arange(N) + 0.5)
  B = jnp.sqrt(2*jnp.arange(N) + 1.0)
  # (N, N) HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B
  return hippo, P, B

############################################
# Creates DPLR Representation of HiPPO-LegS matrix
############################################
def _make_DPLR_HiPPO(self,N):
  # N is the state size
  A, P, B = make_NPLR_HiPPO(N)
  S = A + P[:, jnp.newaxis]*P[jnp.newaxis, :]
  S_diag = jnp.diagonal(S)

  Lambda_real = self.get('Lambda_real', jnp.zeros(), shape=(N,))
  Lambda_real = jnp.mean(S_diag) + jnp.ones_like(S_diag)
  self.put('Lambda_real', Lambda_real)

  # Diagonalize S to V Lambda V^*, Returns eigenvalues and eigenvectors
  Lambda_imag = self.get('Lambda_imag', jnp.zeros(), shape=(N,))
  self.get('V', jnp.zeros(), shape=(N,N))
  Lambda_imag, V = jnp.linalg.eigh(S*-1j)
  self.put('Lambda_imag', Lambda_imag)
  self.put('V', V)
      

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
        key, skey = jax.random.split(key)
        # sample C matrix
        C = jax.nn.initializers.lecun_normal()(skey, (1, P, 2))                                         
        Cs.append(C)
    # (H, P, 2)
    return jnp.array(Cs)[:, 0]


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


#############################################
# Binary operator for parallel scan of linear recurrence
###############################################
@jax.vmap
def binary_operator(element_i, element_j):
    # [(P,),(P,)], [(P,),(P,)]
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    # (A_out, Bu_out)
    return a_i*a_j, a_j*bu_i + bu_j





def s5_scan(self, input_sequence):
  local_P = 2*self.P if self._conj_sym else self.P

  # A: initialize diagonal state to state matrix Lambda (eigenvalues)
  Lambda_re = self.get('Lambda_real')
  Lambda_imag = self.get('Lambda_imag')
  if self.clip_eigs:
    Lambda = jnp.clip(Lambda_re, None, -1e-4) + 1j*Lambda_imag
  else:
    Lambda = Lambda_re + 1j*Lambda_imag

  # B: input to state matrix
  B = self.get('B', jax.nn.initializers.lecun_normal(), nj.rng(), (local_P, self.H))
  B_tilde = B[..., 0] + 1j*B[..., 1]

  # C: state to output matrix
  if self.C_init=='trunc_standard_normal':
    C = self.get('C', trunc_standard_normal, nj.rng(), (self.H, local_P, 2))
  elif self.C_init=='lecun_normal':
    C = self.get('C', jax.nn.initializers.lecun_normal(), nj.rng(), (self.H, local_P, 2))
  elif self.C_init=='complex_normal':
    C = self.get('C', jax.nn.initializers.normal(stddev=0.5**0.5), nj.rng(), (self.H, self.P, 2)) 
  else:
    raise NotImplementedError(f'C_init method {self.C_init} not implemented')
  C_tilde = C[..., 0] + 1j*C[..., 1]

  # D: feedback matrix
  D = self.get('D', jax.nn.initializers.normal(stddev=1.0), nj.rng(), (self.H,))

  # Learnable discretization timescale value
  log_step = self.get('log_step', init_log_steps, (self.P, self.dt_min, self.dt_max))
  step = self.step_rescale * jnp.exp(self.log_step[:, 0])

  if self.discretization in ['zoh']:
    Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
  elif self.discretization in ['bilinear']:
    self.Lambda_bar, self.B_bar = discretize_bilinear(self.Lambda, B_tilde, step)
  else:
    raise NotImplementedError(f'Discretization method {self.discretization} not implemented')
  self.get('Lambda_bar', )
  self.put('Lambda_bar', Lambda_bar)
  self.put('B_bar', B_bar)

  ##############
  # Discretized diagonal state matrix (P,P), input_sequence: (L, H)
  Lambda_bar = self.get('Lambda_bar')
  B_bar = self.get('B_bar')
  C = self.get('C')
  Lambda_elements = Lambda_bar * jnp.ones((input_sequence.shape[0], Lambda_bar.shape[0]))
  # B_bar: Discretized input matrix (P, H), C_tilde: Output matrix (H, P) or (2H, P)
  Bu_elements = jax.vmap(lambda u: B_bar@u)(input_sequence)
  _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))

  C_tilde = C[..., 0] + 1j*C[..., 1]
  if self._conj_sym:
      return jax.vmap(lambda x: 2*(C_tilde@x).real)(xs)
  else:
      return jax.vmap(lambda x: (C_tilde@x).real)(xs) # (L,H)


class S5_WM(nj.Module):

  def __init__(
      self, deter=1024, stoch=32, classes=32, unroll=False, initial='learned',
      unimix=0.01, action_clip=1.0, hidden='lru', conj_sym=True, ssm_size_base=256,
      blocks=8, d_model=128, **kw):
    self._deter = deter
    self._stoch = stoch
    self._classes = classes
    self._unroll = unroll
    self._initial = initial
    self._unimix = unimix
    self._action_clip = action_clip
    self._kw = kw
    self._hidden = hidden
    # S5 specific parameters
    self._conj_sym = conj_sym
    block_size = int(ssm_size_base/blocks)
    if self._conj_sym:
      self.block_size = block_size//2
      self.ssm_size = ssm_size_base//2
    else:
      self.block_size = block_size
      self.ssm_size = ssm_size_base
    self.d_model = d_model


  ########################################
  # Initial consistent with original RSSM
  ########################################
  def initial(self, bs):
    if self._classes:
      state = dict(
          deter=jnp.zeros([bs, self._deter], f32),
          logit=jnp.zeros([bs, self._stoch, self._classes], f32),
          stoch=jnp.zeros([bs, self._stoch, self._classes], f32))
    else:
      state = dict(
          deter=jnp.zeros([bs, self._deter], f32),
          mean=jnp.zeros([bs, self._stoch], f32),
          std=jnp.ones([bs, self._stoch], f32),
          stoch=jnp.zeros([bs, self._stoch], f32))
    if self._initial == 'zeros':
      return cast(state)
    elif self._initial == 'learned':
      deter = self.get('initial', jnp.zeros, state['deter'][0].shape, f32)
      state['deter'] = jnp.repeat(jnp.tanh(deter)[None], bs, 0)
      state['stoch'] = self.get_stoch(cast(state['deter']))
      return cast(state)
    else:
      raise NotImplementedError(self._initial)


  ##############################
  # Initialize complex matrices
  ##############################
  def initialize_matrices(self, bs):
    H = self.d_model
    P = self.ssm_size
    local_P = 2*self.P if self._conj_sym else self.ssm_size

    # A: initialize diagonal state to state matrix Lambda (eigenvalues)
    Lambda_re = self.get('Lambda_real')
    Lambda_imag = self.get('Lambda_imag')
    if self.clip_eigs:
      Lambda = jnp.clip(Lambda_re, None, -1e-4) + 1j*Lambda_imag
    else:
      Lambda = Lambda_re + 1j*Lambda_imag

    # B: input to state matrix
    B = self.get('B', jax.nn.initializers.lecun_normal(), nj.rng(), (local_P, self.H))
    B_tilde = B[..., 0] + 1j*B[..., 1]

    # C: state to output matrix
    if self.C_init=='trunc_standard_normal':
      C = self.get('C', trunc_standard_normal, nj.rng(), (self.H, local_P, 2))
    elif self.C_init=='lecun_normal':
      C = self.get('C', jax.nn.initializers.lecun_normal(), nj.rng(), (self.H, local_P, 2))
    elif self.C_init=='complex_normal':
      C = self.get('C', jax.nn.initializers.normal(stddev=0.5**0.5), nj.rng(), (self.H, self.P, 2)) 
    else:
      raise NotImplementedError(f'C_init method {self.C_init} not implemented')
    C_tilde = C[..., 0] + 1j*C[..., 1]

    # D: feedback matrix
    D = self.get('D', jax.nn.initializers.normal(stddev=1.0), nj.rng(), (self.H,))

    # Learnable discretization timescale value
    log_step = self.get('log_step', init_log_steps, (self.P, self.dt_min, self.dt_max))
    step = self.step_rescale * jnp.exp(self.log_step[:, 0])

    if self.discretization in ['zoh']:
      Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
    elif self.discretization in ['bilinear']:
      self.Lambda_bar, self.B_bar = discretize_bilinear(self.Lambda, B_tilde, step)
    else:
      raise NotImplementedError(f'Discretization method {self.discretization} not implemented')
    self.get('Lambda_bar', )
    self.put('Lambda_bar', Lambda_bar)
    self.put('B_bar', B_bar)
  



  #################################
  # Full observation parallel scan
  #################################
  def observe(self, embed, action, is_first, state=None):
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    step = lambda prev, inputs: self.obs_step(prev[0], *inputs)
    inputs = swap(action), swap(embed), swap(is_first)
    start = state, state
    # print(f'swapped actions {inputs[0].shape}')
    # print(f'swapped embed {inputs[1].shape}')
    # print(f'swapped is_first {inputs[2].shape}')
    # print(f'start state {start[0]["deter"].shape}')
    # post, prior = jaxutils.scan(step, inputs, start, self._unroll)
    post, prior = jaxutils.scan(step, inputs, start, self._unroll, modify=True)
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior


  def observe(self, embed, action, is_first, state=None):






  #################################
  # Full imagination parallel scan
  #################################
  def imagine(self, action, state=None):
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    state = self.initial(action.shape[0]) if state is None else state
    assert isinstance(state, dict), state
    action = swap(action)
    # prior = jaxutils.scan(self.img_step, action, state, self._unroll)
    prior = jaxutils.scan(self.img_step, action, state, self._unroll, modify=True)
    prior = {k: swap(v) for k, v in prior.items()}
    # lru_kw = {'N': self._deter, 'H': action.shape[-1]}
    # y, states = self.get('lru', LRU, **lru_kw)._forward_scan(action)
    return prior
  

  ######################################################################
  # Observation step uses the imagination step to retrieve the prior
  # and concatenates it with the current latent embedding to compute posterior
  ######################################################################
  def obs_step(self, prev_state, prev_action, embed, is_first):
    # This is the same action and stochastic preprocesssing as Dreamer work
    is_first = cast(is_first)
    prev_action = cast(prev_action)
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(prev_action)))
    prev_state, prev_action = jax.tree_util.tree_map(
        lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
    prev_state = jax.tree_util.tree_map(
        lambda x, y: x + self._mask(y, is_first),
        prev_state, self.initial(len(is_first)))
    # [DreamerV3 Sec. Actor-Critic] Actor-critic operates on model states and benefits from Markovian representation 
    prior = self.img_step(prev_state, prev_action)

    # Embed is the next observation in the latent space
    x = jnp.concatenate([prior['deter'], embed], -1)

    # [DreamerV3 Eq. 3]
    x = self.get('obs_out', nets.Linear, **self._kw)(x)
    prev_stoch = prev_state['stoch']
    if self._classes:
      shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
      prev_stoch = prev_stoch.reshape(shape)
    u = jnp.concatenate([prev_stoch, prev_action], -1)
    u = self.get('img_in')(u)    # defined in img_step()

    post_deters = jnp.array([])
    for i in range(x.shape[0]):
      post_deter = self.get('lru').compute_out(input=u[i], prev_state=x[i])
      post_deters = jnp.append(post_deters, post_deter)
    post_deters = jnp.reshape(post_deters, (x.shape[0], -1))

    stats = self._stats('obs_stats', post_deters)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    # print(f'prior deter shape {prior["deter"].shape}')
    # print(f'post deter shape {post["deter"].shape}')
    return cast(post), cast(prior)


  ############################################################################
  # Imagination step takes in the previous hidden state which is composed of both a
  # deterministic and stochastic component and outputs out the next hidden state
  ############################################################################
  def img_step(self, prev_state, prev_action):
      # print(f'prev_state {prev_state["deter"].shape}')
      # print(f'prev_action {prev_action.shape}')
      # print(f' prev_state stoch {prev_state["stoch"].shape}')

      # This is the same action and stochastic preprocesssing as Dreamer work
      prev_stoch = prev_state['stoch']
      prev_action = cast(prev_action)
      if self._action_clip > 0.0:
        prev_action *= sg(self._action_clip / jnp.maximum(self._action_clip, jnp.abs(prev_action)))
      if self._classes:
        shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
        prev_stoch = prev_stoch.reshape(shape)
      if len(prev_action.shape)>len(prev_stoch.shape):
        shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
        prev_action = prev_action.reshape(shape)
      # Previous stochastic state and the action compose the full input
      u = jnp.concatenate([prev_stoch, prev_action], -1)
      u = self.get('img_in', nets.Linear, **self._kw)(u)

      # Dynamics predictor: Linear Recurrent Unit [N: state dim, H: model dim]
      ys, deters = jnp.array([]), jnp.array([])
      # ys = jnp.zeros((prev_action.shape[0], u.shape[-1]))
      # deters = jnp.zeros((prev_action.shape[0], self._deter))
      lru_kw = {'N': self._deter, 'H': u.shape[-1]}
      for i in range(prev_action.shape[0]):
        y, deter = self.get('lru', LRU, **lru_kw)(input=u[i],prev_state=prev_state['deter'][i])
        # ys = ys.at[i].set(y)
        # deters = deters.at[i].set(deter)
        ys = jnp.append(ys, y)
        deters = jnp.append(deters, deter)
      ys = jnp.reshape(ys, (prev_action.shape[0], -1))
      deters = jnp.reshape(deters, (prev_action.shape[0], -1))
      # print(f'ys {ys.shape}')
      # print(f'deters {deters.shape}')

      # Linear layer for stochastic number of classes
      # ys = self.get('img_out', nets.Linear, **self._kw)(ys)

      # Compute stochastic output
      stats = self._stats('img_stats', ys)
      dist = self.get_dist(stats)
      # [DreamerV3 Eq. 3]
      stoch = dist.sample(seed=nj.rng())
      # [DreamerV3 Sec. 2]: model state is concatenation of deter hidden state and sample of stoch state
      prior = {'stoch': stoch, 'deter': deters, **stats}

      return cast(prior)


  ####################################################
  # Discretize continuous action space into classes or
  # use a multivariate gaussian for continuous actions
  ####################################################
  def get_dist(self, state, argmax=False):
    if self._classes:
      logit = state['logit'].astype(f32)
      return tfd.Independent(jaxutils.OneHotDist(logit), 1)
    else:
      mean = state['mean'].astype(f32)
      std = state['std'].astype(f32)
      return tfd.MultivariateNormalDiag(mean, std)


  ####################################################
  # Stochastic component which is retrieved by passing
  # the hidden state through a dense layer and sampling
  ####################################################
  def get_stoch(self, deter):
    x = self.get('img_out', nets.Linear, **self._kw)(deter)
    stats = self._stats('img_stats', x)
    dist = self.get_dist(stats)
    return cast(dist.mode())
  


  def _stats(self, name, x):
    if self._classes:
      x = self.get(name, nets.Linear, self._stoch * self._classes)(x)
      logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
      if self._unimix:
        probs = jax.nn.softmax(logit, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        logit = jnp.log(probs)
      stats = {'logit': logit}
      return stats
    else:
      x = self.get(name, nets.Linear, 2 * self._stoch)(x)
      mean, std = jnp.split(x, 2, -1)
      std = 2 * jax.nn.sigmoid(std / 2) + 0.1
      return {'mean': mean, 'std': std}


  def _mask(self, value, mask):
    return jnp.einsum('b...,b->b...', value, mask.astype(value.dtype))


  def dyn_loss(self, post, prior, impl='kl', free=1.0):
    if impl == 'kl':
      loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
    elif impl == 'logprob':
      loss = -self.get_dist(prior).log_prob(sg(post['stoch']))
    else:
      raise NotImplementedError(impl)
    if free:
      loss = jnp.maximum(loss, free)
    return loss


  def rep_loss(self, post, prior, impl='kl', free=1.0):
    if impl == 'kl':
      loss = self.get_dist(post).kl_divergence(self.get_dist(sg(prior)))
    elif impl == 'uniform':
      uniform = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), prior)
      loss = self.get_dist(post).kl_divergence(self.get_dist(uniform))
    elif impl == 'entropy':
      loss = -self.get_dist(post).entropy()
    elif impl == 'none':
      loss = jnp.zeros(post['deter'].shape[:-1])
    else:
      raise NotImplementedError(impl)
    if free:
      loss = jnp.maximum(loss, free)
    return loss



class Nu_Log_Initializer:

  def __init__(self, r_min, r_max):
    self.r_min = r_min
    self.r_max = r_max
  
  def __call__(self, shape):
    u1 = np.random.uniform(size=shape)
    return np.log(-0.5*np.log(u1*(self.r_max**2-self.r_min**2) + self.r_min**2))
  

class Theta_Log_Initializer:

  def __init__(self, max_phase):
    self.max_phase = max_phase

  def __call__(self, shape):
    u2 = np.random.uniform(size=shape)
    return np.log(self.max_phase*u2)
    

class Gamma_Log_Initializer:

  def __init__(self, nu_log, theta_log):
    self.nu_log = nu_log
    self.theta_log = theta_log

  def __call__(self, shape):
    # [Sec 3.3] Lambda is a complex value matrix uniformly distributed on ring between [r_min, r_max]
    diag_lambda = np.exp(-np.exp(self.nu_log) + 1j*np.exp(self.theta_log))
    return np.log(np.sqrt(1-np.abs(diag_lambda)**2))

class B_Initializer:

  def __init__(self, scale=2.0):
    self.scale = scale

  def __call__(self, shape):
    # Shape: (N,H)
    return np.random.normal(size=shape)/np.sqrt(self.scale*shape[1])

class C_Initializer:

  def __init__(self, scale=2.0):
    self.scale = scale

  def __call__(self, shape):
    # Shape: (H,N)
    return np.random.normal(size=shape)/np.sqrt(self.scale*shape[1])



class LRU(nj.Module):
  ''' Linear Recurrent Unit '''

  ########################################
  # Initialize complex value matrices
  ########################################
  def __init__(self, N, H, r_min=0, r_max=1, max_phase=6.28):
    self.N = N
    self.H = H
    self.r_min = r_min
    self.r_max = r_max
    self.max_phase = max_phase
    self.u1 = np.random.uniform(size=(N,))
    self.u2 = np.random.uniform(size=(N,))


  ###################################
  # Helper for linear recurrent scan
  ###################################
  def _binary_operator_diag(self, element_i, element_j):
    ''' Binary operator for parallel scan of linear recurrence '''
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    return a_i*a_j, a_j*bu_i + bu_j


  # def _init_params(self):
  #   self.get('nu_log', jnp.zeros, (self.N,), jnp.float32)
  #   self.get('theta_log', jnp.zeros, (self.N,), jnp.float32)
  #   self.get('B_re', jnp.zeros, (self.N,self.H), jnp.float32)
  #   self.get('B_im', jnp.zeros, (self.N,self.H), jnp.float32)
  #   self.get('C_re', jnp.zeros, (self.H,self.N), jnp.float32)
  #   self.get('C_im', jnp.zeros, (self.H,self.N), jnp.float32)
  #   self.get('D', jnp.zeros, (self.H,), jnp.float32)
  #   self.get('gamma_log', jnp.zeros, (self.N,), jnp.float32)
  #   self.put('nu_log', np.log(-0.5*np.log(self.u1*(self.r_max**2-self.r_min**2) + self.r_min**2)))
  #   self.put('theta_log', np.log(self.max_phase*self.u2))
  #   # Glorot initialized input/output projection matrices
  #   self.put('B_re', np.random.normal(size=(self.N,self.H))/np.sqrt(2*self.H))
  #   self.put('B_im', np.random.normal(size=(self.N,self.H))/np.sqrt(2*self.H))
  #   self.put('C_re', np.random.normal(size=(self.H,self.N))/np.sqrt(self.N))
  #   self.put('C_im', np.random.normal(size=(self.H,self.N))/np.sqrt(self.N))
  #   self.put('D', np.random.normal(size=(self.H,)))
  #   # [Sec 3.3] Lambda is a complex value matrix uniformly distributed on ring between [r_min, r_max] 
  #   diag_lambda = jnp.exp(-jnp.exp(self.get('nu_log')) + 1j*jnp.exp(self.get('theta_log')))
  #   # [Eq.7] Normalization factor that gets multiplied element-wise with Bu_{k}
  #   self.put('gamma_log', jnp.log(jnp.sqrt(1-jnp.abs(diag_lambda)**2)))

  #   self.init_flag = 1

    # self.nu_log = np.log(-0.5*np.log(self.u1*(r_max**2-r_min**2) + r_min**2))
    # self.theta_log = np.log(max_phase*self.u2)
    # # Gloret initialized input/output projection matrices
    # self.B_re = np.random.normal(size=(N,H))/np.sqrt(2*H)
    # self.B_im = np.random.normal(size=(N,H))/np.sqrt(2*H)
    # self.C_re = np.random.normal(size=(H,N))/np.sqrt(N)
    # self.C_im = np.random.normal(size=(H,N))/np.sqrt(N)
    # self.D = np.random.normal(size=(H,))
    # # [Sec 3.3] Lambda is a complex value matrix uniformly distributed on ring between [r_min, r_max]
    # diag_lambda = np.exp(-np.exp(self.nu_log) + 1j*np.exp(self.theta_log))
    # # [Eq.7] Normalization factor that gets multiplied element-wise with Bu_{k}
    # self.gamma_log = np.log(np.sqrt(1-np.abs(diag_lambda)**2))


  def compute_out(self, input, prev_state):
    C_re = self.get('C_re')
    C_im = self.get('C_im')
    D = self.get('D')
    C = C_re + 1j*C_im
    y = (jnp.dot(C, prev_state).real + D*input).astype(jnp.float32)
    return y


  ########################################
  # Single step of the LRU
  ########################################
  def __call__(self, input, prev_state):


    # [lemma 3.2] learnable parameters for diagonal lambda
    nu_log = self.get('nu_log', nu_log_initializer(self.r_min, self.r_max), (self.n,))
    theta_log = self.get('theta_log', theta_log_initializer(self.max_phase), (self.n,))
    # glorot initialized input/output projection matrices
    # b_re = self.get('b_re', b_initializer, (self.n,self.h))
    # b_im = self.get('b_im', b_initializer, (self.n,self.h))
    # c_re = self.get('c_re', c_initializer, (self.h,self.n))
    # c_im = self.get('c_im', c_initializer, (self.h,self.n))
    b_re = self.get('b_re', jax.nn.initializers.glorot_normal(), nj.rng(), (self.n,self.h))
    b_im = self.get('b_im', jax.nn.initializers.glorot_normal(), nj.rng(), (self.n,self.h))
    c_re = self.get('c_re', jax.nn.initializers.glorot_normal(), nj.rng(), (self.h,self.n))
    c_im = self.get('c_im', jax.nn.initializers.glorot_normal(), nj.rng(), (self.h,self.n))
    d = self.get('d', jax.nn.initializers.normal(), nj.rng(), (self.h,))
    # [Eq.7] Normalization factor that gets multiplied element-wise with Bu_{k}
    gamma_log = self.get('gamma_log', Gamma_Log_Initializer(nu_log, theta_log), (self.N,))

    # Construct diagonal Lambda with stability enforced
    Lambda = jnp.exp(-jnp.exp(nu_log) + 1j*jnp.exp(theta_log))
    Lambda = jnp.repeat(Lambda[None, ...], input.shape[0], axis=0)
    # Construct B matrix with normalization
    gamma = jnp.expand_dims(jnp.exp(gamma_log), axis=-1)
    B_norm = (B_re + 1j*B_im)*gamma
    # Construct C matrix
    C = C_re + 1j*C_im

    # First hidden state term
    Lambda_elements = jnp.dot(Lambda, prev_state)

    # Second hidden state term
    # Bu_elements = jax.vmap(lambda u: B_norm @ u)(input)
    # Bu_elements = jax.vmap(lambda x,y: jnp.dot(x,y))(B_norm, input)
    Bu_elements = jnp.dot(B_norm, input)

    # Updated state composed from A (Lambda) and B
    state = Lambda_elements + Bu_elements

    # Output composed from C and D 
    # y = jax.vmap(lambda x,u: (C @ x).real + self.D*u)(state, input)
    # y = (C @ state).real + self.D*input
    y = jnp.dot(C, state).real + D*input

    # Take real component of state
    y = y.astype(jnp.float32)
    state = state.astype(jnp.float32)

    return y, state

