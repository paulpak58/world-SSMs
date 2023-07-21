`import re
from typing import Any
import math
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import block_diag
from tensorflow_probability.substrates import jax as tfp
from . import jaxutils
from . import ninjax as nj
from . import nets
from . import s5_utils
from s5_utils import discretize_bilinear, discretize_zoh, init_log_steps, \
                     trunc_standard_normal, make_DPLR_HiPPO, binary_operator
from initializers import VinvB_initializer, CV_initializer, lambda_re_initializer, lambda_imag_initializer

f32 = jnp.float32
tfd = tfp.distributions
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute




class MultiHeadEMA(nj.Module):
  def __init__(
      self, d_model, d_state=2, bidirectional=False, l_max=None
  ):
    self.H = d_model
    self.N = d_state
    self.bidirectional = bidirectional
    self.l_max = l_max
    self.scale = math.sqrt(1.0/self.N)




class Mega_WM(nj.Module):

  def __init__(
      self, deter=1024, stoch=32, classes=32, unroll=False, initial='learned',
      unimix=0.01, action_clip=1.0, d_model, d_attin, d_attout, d_state, **kw):
    self._deter = deter
    self._stoch = stoch
    self._classes = classes
    self._unroll = unroll
    self._initial = initial
    self._unimix = unimix
    self._action_clip = action_clip
    self._kw = kw
    # Mega parameter
    self.d_model = d_model
    self.d_attin = d_attin
    self.d_attout = d_attout
    self.d_state = d_state



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
  # Initialize matrices
  ##############################
  def initialize_matrices(self):
    # delta & alpha (dt and A parameters)
    delta = self.get('delta', jax.nn.initializers.normal(stddev=0.2), (self.ssm_size,))
    alpha = self.get('alpha', jax.nn.initializers.normal(stddev=0.2), (self.ssm_size,))
    # Mega: beta [1,-1,1,-1,...] like implementation by Gu
    val = jax.nn.ones(self.N) #TODO: self.N
    if self.N>1:
      idx = jnp.array(list(range(1, self.N, 2)))
      val.index_fill_(0, idx, -1.0)
    beta = self.get('beta', jax.nn.initializers.normal(stddev=0.2), (self.ssm_size,))
    # gamma & omega (C and D parameters) of unit variance
    gamma = self.get('gamma', jax.nn.initializers.normal(stddev=0.1), (self.ssm_size,))
    omega = self.get('omega', jax.nn.initializers.normal(stddev=0.1), (self.ssm_size,))
    ####### Linear layers #######
    v_proj = self.get('v_proj', nets.Linear, (self.d_model, self.d_attout), bias=True)
    mx_proj = self.get('mx_proj', nets.Linear, (self.d_model, self.d_attin+self.d_attout+2*self.d_model) bias=True)
    h_proj = self.get('h_proj', nets.Linear, (self.d_attout, self.d_model) bias=True)


  def element_attention(self, q, k, padding_mask, attn_mask, before_attn_fn):
    slen = k.size(2)
    if padding_mask is not None:
      inverse_mask = 1.0-padding_mask.type_as(q)  # (B,K,C)
      lengths = inverse_mask.sum(dim=-1, keepdim=True)  # (B,K,1)
      lengths = lengths.clamp(min=1.0).unsqueeze(-1)  # (B,K,1,1)
    else:
      lengths = slen
      inverse_mask = None
    if attn_mask is not None:
      lengths = attn_mask.sum(dim=-1, keepdim=True) # Cx1
    # TODO


  def forward_scan(
    self,
    x,
    state = None,
    padding_mask = None,
    need_weights = False,
    attn_mask = None,
    before_attn_fn = False,
    **kwargs
  ):
    # Input shape: (B, L, D)
    if self.transposed: # TODO
      x = x.transpose(-1,-2)
    B,L,D = x.size()
    assert D==self.d_model

    residual = x
    if self.prenorm:  # TODO
      x = self.get('norm')(x)
    v_proj = self.get('v_proj')(x)
    v = self.get('activation')(v_proj)


    mx, _ = self.ssm(x, state=state, padding_mask=padding_mask) # (B,L,D)
      




  #####################################
  # S5 scan operates on (L,H) sequences
  #####################################
  def s5_scan(self, input_sequence):
    local_P = 2*self.ssm_size if self._conj_sym else self.ssm_size
    C = self.get('C', CV_initializer, self.C_init, nj.rng(), (self.H, local_P, 2), self.V)
    C_tilde = C[..., 0] + 1j*C[..., 1]
    D = self.get('D', jax.nn.initializers.normal(stddev=1.0), nj.rng(), (self.H,))
    Lambda_elements = self.Lambda_bar * jnp.ones((input_sequence.shape[0], self.Lambda_bar.shape[0]))
    # B_bar: Discretized input matrix (P, H)
    Bu_elements = jax.vmap(lambda u: self.B_bar @ u)(input_sequence)
    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))
    if self.conj_sym:
        ys = jax.vmap(lambda x: 2*(C_tilde @ x).real)(xs)
    else:
        ys = jax.vmap(lambda x: (C_tilde @ x).real)(xs) # (L,H)
    Du = jax.vmap(lambda u: self.D * u)(input_sequence)
    return ys + Du, xs

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
    print(f'swapped actions {inputs[0].shape}')
    print(f'swapped embed {inputs[1].shape}')
    print(f'swapped is_first {inputs[2].shape}')

    # post, prior = jaxutils.scan(step, inputs, start, self._unroll)
    post, prior = jaxutils.scan(step, inputs, start, self._unroll, modify=True)
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
    return loss``
`