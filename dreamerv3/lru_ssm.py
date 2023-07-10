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



class LRU_SSM(nj.Module):

  def __init__(
      self, deter=1024, stoch=32, classes=32, unroll=False, initial='learned',
      unimix=0.01, action_clip=1.0, hidden='lru', **kw):
    self._deter = deter
    self._stoch = stoch
    self._classes = classes
    self._unroll = unroll
    self._initial = initial
    self._unimix = unimix
    self._action_clip = action_clip
    self._kw = kw
    self._hidden = hidden

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
    post, prior = jaxutils.scan(step, inputs, start, self._unroll)
    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    return post, prior

  #################################
  # Full imagination parallel scan
  #################################
  def imagine(self, action, state=None):
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    state = self.initial(action.shape[0]) if state is None else state
    assert isinstance(state, dict), state
    action = swap(action)
    # prior = jaxutils.scan(self.img_step, action, state, self._unroll)
    lru_kw = {'bs': action.shape[0], 'deter': self._deter}
    y, states = self.get('lru', LRU, **lru_kw)._forward_scan(action)
    state['deter'] = states

    prior = {k: swap(v) for k, v in prior.items()}
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
    stats = self._stats('obs_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    return cast(post), cast(prior)


  ############################################################################
  # Imagination step takes in the previous hidden state which is composed of both a
  # deterministic and stochastic component and outputs out the next hidden state
  ############################################################################
  def img_step(self, prev_state, prev_action):
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
      # Dynamics predictor: Linear Recurrent Unit
      lru_kw = {'bs': prev_action.shape[0], 'deter': self._deter}
      y, deter = self.get('lru', LRU, **lru_kw)(input=u, prev_state=prev_state['deter'])
      # Linear layer for stochastic number of classes
      y = self.get('img_out', nets.Linear, **self._kw)(y)
      # Compute stochastic output
      stats = self._stats('img_stats', y)
      dist = self.get_dist(stats)
      # [DreamerV3 Eq. 3]
      stoch = dist.sample(seed=nj.rng())
      # [DreamerV3 Sec. 2]: model state is concatenation of deter hidden state and sample of stoch state
      prior = {'stoch': stoch, 'deter': deter, **stats}

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



class LRU(nj.Module):
  ''' Linear Recurrent Unit '''

  ########################################
  # Initialize complex value matrices
  ########################################
  def __init__(self, bs, deter, r_min=0, r_max=1, max_phase=6.28):
    u1 = np.random.uniform(size=(bs,))
    u2 = np.random.uniform(size=(bs,))
    # [Lemma 3.2] Learnable parameters for diagonal Lambda
    self.nu_log = np.log(-0.5*np.log(u1*(r_max**2-r_min**2) + r_min**2))
    self.theta_log = np.log(max_phase*u2)
    # Gloret initialized input/output projection matrices
    self.B_re = np.random.normal(size=(bs,deter))/np.sqrt(2*deter)
    self.B_im = np.random.normal(size=(bs,deter))/np.sqrt(2*deter)
    self.C_re = np.random.normal(size=(deter,bs))/np.sqrt(bs)
    self.C_im = np.random.normal(size=(deter,bs))/np.sqrt(bs)
    self.D = np.random.normal(size=(deter,))
    # [Sec 3.3] Lambda is a complex value matrix uniformly distributed on ring between [r_min, r_max]
    self.diag_lambda = np.exp(-np.exp(self.nu_log) + 1j*np.exp(self.theta_log))
    # [Eq.7] Normalization factor that gets multiplied element-wise with Bu_{k}
    self.gamma_log = np.log(np.sqrt(1-np.abs(self.diag_lambda)**2))


  ###################################
  # Helper for linear recurrent scan
  ###################################
  def _binary_operator_diag(self, element_i, element_j):
    ''' Binary operator for parallel scan of linear recurrence '''
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    return a_i*a_j, a_j*bu_i + bu_j


  ########################################
  # Full parallel scan across sequence
  ########################################
  def _forward_scan(self, input):
    # Construct diagonal Lambda with stability enforced
    Lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j*jnp.exp(self.theta_log))
    # Construct B matrix with normalization
    gamma = jnp.expand_dims(jnp.exp(self.gamma_log), axis=-1)
    B_norm = (self.B_re + 1j*self.B_im)*gamma
    # Construct C matrix
    C = self.C_re + 1j*self.C_im

    # First hidden state term
    Lambda_elements = jnp.repeat(Lambda[None, ...], input.shape[0], axis=0)
    # Second hidden state term
    Bu_elements = jax.vmap(lambda u: B_norm @ u)(input)
    # Updated state composed from A (Lambda) and B
    elements = (Lambda_elements, Bu_elements)
    _, states = jax.lax.associative_scan(self._binary_operator_diag, elements)
    # Output composed from C and D 
    y = jax.vmap(lambda x,u: (C @ x).real + self.D*u)(states, input)

    return y, states


  ########################################
  # Single step of the LRU
  ########################################
  def __call__(self, input, prev_state):
    # Construct diagonal Lambda with stability enforced
    Lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j*jnp.exp(self.theta_log))
    Lambda = jnp.repeat(Lambda[None, ...], input.shape[0], axis=0)
    # Construct B matrix with normalization
    gamma = jnp.expand_dims(jnp.exp(self.gamma_log), axis=-1)
    B_norm = (self.B_re + 1j*self.B_im)*gamma
    # Construct C matrix
    C = self.C_re + 1j*self.C_im

    # First hidden state term
    # Lambda_elements = jax.vmap(lambda x: Lambda @ x)(prev_state)
    Lambda_elements = jnp.array(Lambda @ prev_state)

    # Second hidden state term
    #Bu_elements = jax.vmap(lambda u: B_norm @ u)(input)
    print(f'B_norm: {B_norm.shape}, input: {input.shape}')
    Bu_elements = jnp.array(B_norm @ input)

    # Updated state composed from A (Lambda) and B
    #Bu_elements = jnp.repeat(Bu_elements, Lambda_elements.shape[1], axis=1)
    print(f'Lambda_elements: {Lambda_elements.shape}, Bu_elements: {Bu_elements.shape}')
    state = Lambda_elements + Bu_elements
    # Output composed from C and D 
    y = jax.vmap(lambda x,u: (C @ x).real + self.D*u)(state, input)

    return y, state

