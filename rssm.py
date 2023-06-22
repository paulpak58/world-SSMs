
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
from . import ninjax as nj
from . import jaxutils

# Applies stop gradient to every leaf node in the input tree
sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)

class Linear(nj.Module):

  def __init__(
      self, units, act='none', norm='none', bias=True, outscale=1.0,
      outnorm=False, winit='uniform', fan='avg'):
    self._units = tuple(units) if hasattr(units, '__len__') else (units,)
    self._act = get_act(act)
    self._norm = norm
    self._bias = bias and norm == 'none'
    self._outscale = outscale
    self._outnorm = outnorm
    self._winit = winit
    self._fan = fan

  def __call__(self, x):
    shape = (x.shape[-1], np.prod(self._units))
    kernel = self.get('kernel', Initializer(
        self._winit, self._outscale, fan=self._fan), shape)
    kernel = jaxutils.cast_to_compute(kernel)
    x = x @ kernel
    if self._bias:
      bias = self.get('bias', jnp.zeros, np.prod(self._units), np.float32)
      bias = jaxutils.cast_to_compute(bias)
      x += bias
    if len(self._units) > 1:
      x = x.reshape(x.shape[:-1] + self._units)
    x = self.get('norm', Norm, self._norm)(x)
    x = self._act(x)
    return x




class RSSM(nj.Module):

  def __init__(
      self, deter=1024, stoch=32, classes=32, unroll=False, initial='learned',
      unimix=0.01, action_clip=1.0, **kw):
    self._deter = deter
    self._stoch = stoch
    self._classes = classes
    self._unroll = unroll
    self._initial = initial
    self._unimix = unimix
    self._action_clip = action_clip
    self._kw = kw

  def initial(self, bs):
    if self._classes:
      state = dict(
          deter=jnp.zeros([bs, self._deter], jnp.float32),
          logit=jnp.zeros([bs, self._stoch, self._classes], jnp.float32),
          stoch=jnp.zeros([bs, self._stoch, self._classes], jnp.float32))
    else:
      state = dict(
          deter=jnp.zeros([bs, self._deter], jnp.float32),
          mean=jnp.zeros([bs, self._stoch], jnp.float32),
          std=jnp.ones([bs, self._stoch], jnp.float32),
          stoch=jnp.zeros([bs, self._stoch], jnp.float32))
    if self._initial == 'zeros':
      return jaxutils.cast_to_compute(state)
    elif self._initial == 'learned':
      deter = self.get('initial', jnp.zeros, state['deter'][0].shape, jnp.float32)
      state['deter'] = jnp.repeat(jnp.tanh(deter)[None], bs, 0)
      state['stoch'] = self.get_stoch(jaxutils.cast_to_compute(state['deter']))
      return jaxutils.cast_to_compute(state)
    else:
      raise NotImplementedError(self._initial)




  ##################################
  # Returns probability distribution
  ##################################
  def get_dist(self, state, argmax=False):
    # If the distribution is categorical
    if self._classes:
      logit = state['logit'].astype(jnp.float32)
      return tfp.distributions.Independent(jaxutils.OneHotDist(logit), 1)
    # Else, represent the distribution as Gaussian
    else:
      mean = state['mean'].astype(jnp.float32)
      std = state['std'].astype(jnp.float32)
      return tfp.distributions.MultivariateNormalDiag(mean, std)
    

  ######################################
  # Single observation step that returns
  # the prior and posterior distributions
  ######################################
  def obs_step(self, prev_state, prev_action, embed, is_first):

    # Set up tree map to convert data types
    cast_to_float32 = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32), x)
    is_first = cast_to_float32(is_first)
    prev_action = cast_to_float32(prev_action)

    # The action is clipped by the maximum action value
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))
  
    # Apply masks to the previous state and action so that model only focuses on the current state
    mask = lambda x,m: jnp.einsum('b...,b->b...', x, m.astype(x.dtype))
    prev_state, prev_action = jax.tree_util.tree_map(lambda x: mask(x, 1.0 - is_first), (prev_state, prev_action))
    prev_state = jax.tree_util.tree_map(lambda x, y: x + mask(y, is_first), prev_state, self.initial(len(is_first)))
    
    # Retrieve the prior distribution predicted by the world model
    prior = self.img_step(prev_state, prev_action)

    # Deterministic prior represents the mean or central tendency
    # Stochastic component represents uncertainty
    x = jnp.concatenate([prior['deter'], embed], -1)
    x = self.get('obs_out', Linear, **self._kw)(x)
    stats = self._stats('obs_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())

    # Posterior distribution is composed of stochastic state and deterministic prior
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}

    return cast_to_float32(post), cast_to_float32(prior)

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

  def img_step(self, prev_state, prev_action):
    prev_stoch = prev_state['stoch']
    prev_action = jaxutils.cast_to_compute(prev_action)
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))
    if self._classes:
      shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
      prev_stoch = prev_stoch.reshape(shape)
    if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
      shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
      prev_action = prev_action.reshape(shape)
    x = jnp.concatenate([prev_stoch, prev_action], -1)
    x = self.get('img_in', Linear, **self._kw)(x)
    x, deter = self._gru(x, prev_state['deter'])
    x = self.get('img_out', Linear, **self._kw)(x)
    stats = self._stats('img_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    prior = {'stoch': stoch, 'deter': deter, **stats}
    return jaxutils.cast_to_compute(prior)

  def imagine(self, action, state=None):
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    state = self.initial(action.shape[0]) if state is None else state
    assert isinstance(state, dict), state
    action = swap(action)
    prior = jaxutils.scan(self.img_step, action, state, self._unroll)
    prior = {k: swap(v) for k, v in prior.items()}
    return prior

  def get_stoch(self, deter):
    x = self.get('img_out', Linear, **self._kw)(deter)
    stats = self._stats('img_stats', x)
    dist = self.get_dist(stats)
    return jaxutils.cast_to_compute(dist.mode())


  ################################
  # Gated Recurrent Unit
  ################################
  def _gru(self, x, deter):
    # Attach the last hidden state to the input
    x = jnp.concatenate([deter, x], -1)

    # Dictionary used to linearly transform input to three gates
    kw = {**self._kw, 'act': 'none', 'units': 3 * self._deter}
    x = self.get('gru', Linear, **kw)(x)

    # Apportion the transformation into those three gates
    reset, cand, update = jnp.split(x, 3, -1)

    # How much of prevous hidden state to forget; bound between 0 and 1
    reset = jax.nn.sigmoid(reset)

    # Apply tanh and bound between -1 and 1
    cand = jnp.tanh(reset * cand)

    # Controls how much to update the next hidden state
    update = jax.nn.sigmoid(update - 1)
    deter = update * cand + (1 - update) * deter

    # deter is both the next hidden state and output
    return deter, deter

  def _stats(self, name, x):
    if self._classes:
      x = self.get(name, Linear, self._stoch * self._classes)(x)
      logit = x.reshape(x.shape[:-1] + (self._stoch, self._classes))
      if self._unimix:
        probs = jax.nn.softmax(logit, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        logit = jnp.log(probs)
      stats = {'logit': logit}
      return stats
    else:
      x = self.get(name, Linear, 2 * self._stoch)(x)
      mean, std = jnp.split(x, 2, -1)
      std = 2 * jax.nn.sigmoid(std / 2) + 0.1
      return {'mean': mean, 'std': std}



  ##################################
  # Dynamic loss computes loss between
  # posterior and prior distributions
  ##################################
  def dyn_loss(self, post, prior, impl='kl', free=1.0):
    if impl == 'kl':
      # categorical loss is computed as KL divergence between posterior and prior
      loss = self.get_dist(sg(post)).kl_divergence(self.get_dist(prior))
    elif impl == 'logprob':
      # negative log probability of the posterior is calculated w.r.t. the prior
      loss = -self.get_dist(prior).log_prob(sg(post['stoch']))
    else:
      raise NotImplementedError(impl)
    if free:
      loss = jnp.maximum(loss, free)
    return loss

  ###########################################################
  # Representation loss computes loss between posterior and a 
  # uniform distribution or computes the entropy of the posterior
  ###########################################################
  def rep_loss(self, post, prior, impl='kl', free=1.0):
    if impl == 'kl':
      # Note the stop gradients on the prior vs the posterior above
      loss = self.get_dist(post).kl_divergence(self.get_dist(sg(prior)))
    elif impl == 'uniform':
      # Regularize the posterior to be uniform
      uniform = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), prior)
      loss = self.get_dist(post).kl_divergence(self.get_dist(uniform))
    elif impl == 'entropy':
      # Encourages exploration
      loss = -self.get_dist(post).entropy()
    elif impl == 'none':
      loss = jnp.zeros(post['deter'].shape[:-1])
    else:
      raise NotImplementedError(impl)
    if free:
      loss = jnp.maximum(loss, free)
    return loss