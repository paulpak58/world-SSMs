import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from functools import partial
from tensorflow_probability.substrates import jax as tfp

from . import jaxutils
from . import ninjax as nj
from . import nets

# from . import s5_test
from . import s5
# from . import s4
# from . import dss


f32 = jnp.float32
tfd = tfp.distributions
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute
Linear = nets.Linear


SSM_MODELS = {
  # 's4': s4.S4LayerInit,
  # 'dss': dss.DSSLayerInit,
  's5': s5.S5LayerInit
}



def build_ssm(config):
  if config.ssm =='s5':
    raise NotImplementedError
  elif config.ssm=='s4':
    raise NotImplementedError
  elif config.ssm=='s4d':
    raise NotImplementedError
  elif config.ssm=='dss':
    raise NotImplementedError
  elif config.ssm=='liquid':
    raise NotImplementedError
  elif config.ssm=='lru':
    raise NotImplementedError
  elif config.ssm=='mega':
    raise NotImplementedError
  else:
    raise NotImplementedError
  return ssm  

########################################
# Masked meanpool used for pooling output
########################################
def masked_meanpool(x, lengths):
  L = x.shape[0]
  mask = jnp.arange(L)<lengths
  return jnp.sum(mask[...,None]*x, axis=0)/lengths



###############################################
# Full Sequence Layer that includes the SSM, Norm,
# Dropout, and Nonlinearity [Implemented in Ninjax]
##############################################
class GeneralSequenceLayer(nj.Module):

  def __init__(self, ssm: nj.Module, dropout: float, d_model: int,
      activation: str='gelu', training: bool=True, prenorm: bool=False,
      batchnorm: bool=False, bn_momentum: float=0.9, step_rescale: float=1.0):
    self.ssm = ssm
    self.activation = activation
    self.prenorm = prenorm

    if self.activation in ['full_glu']:
      self.out1 = nj.FlaxModule(nn.Dense, d_model, name='out1')
      self.out2 = nj.FlaxModule(nn.Dense, d_model, name='out2')
    elif self.activation in ['half_glu1', 'half_glu2']:
      self.out2 = nj.FlaxModule(nn.Dense, d_model, name='out2')
    if batchnorm:
      # Layer norm is defined inline but it also can be defined upfront as done with batchnorm here. We show both.
      self.norm = nj.FlaxModule(nn.BatchNorm, use_running_average=not training, momentum=bn_momentum, axis_name='batch', name='norm')
    self.drop = nj.FlaxModule(nn.Dropout, dropout, broadcast_dims=[0], deterministic=not training, name='drop')


  def __call__(self, x, state, mode='train'):
    # Takes in (L, d_model) and outputs (L, d_model)
    skip = x
    if self.prenorm:
      x = self.get('norm', nets.Norm, 'layer')(x)
    if mode=='train':
      # x, state = self.get('ssm', self.ssm)(x)
      # x, state = self.get('s5', s5.NinjaxS5Layer, **self.layer_args)(x, state=None, train=True)
      x, state = self.get('s5', self.ssm)(x, state, train=True)
    else:
      # x, state = self.get('ssm', self.ssm).step(x, state)
      x, state = self.get('s5', self.ssm)(x, state, train=False)
      # x, state = self.get('ssm', self.ssm)(x, state)
    if self.activation in ['full_glu']:
      x = self.drop(nn.gelu(x))
      x = self.out1(x) * jax.nn.sigmoid(self.out2(x))
      x = self.drop(x)
    elif self.activation in ['half_glu1']:
      x = self.drop(nn.gelu(x))
      x = x * jax.nn.sigmoid(self.out2(x))
      x = self.drop(x)
    elif self.activation in ['half_glu2']:
      # only apply GELU to the gate input
      x1 = self.drop(nn.gelu(x))
      x = x * jax.nn.sigmoid(self.out2(x1))
      x = self.drop(x)
    elif self.activation in ['gelu']:
      # x = self.drop(nn.gelu(x))
      x = nn.gelu(x)
    else:
      raise NotImplementedError(f'Activation {self.activation} not implemented')
    x = skip + x
    if not self.prenorm:
      x = self.get('norm', nets.Norm, 'layer')(x)
    # cache_val = self.seq.x_k_1.value
    return x, state





##############################
# Stacked Deep SSM Layers
# Implemented in ninjax
##############################
class StackedSSM(nj.Module):
    
    def __init__(
      self, init_fn, n_layers, dropout, d_model, act, prenorm, batchnorm, bn_momentum,
    ):
      self.n_layers = n_layers
      # ssm layer attributes
      self.init_fn = init_fn
      self.dropout = dropout
      self.d_model = d_model
      self.act = act
      self.prenorm = prenorm
      self.batchnorm = batchnorm
      self.bn_momentum = bn_momentum

    def __call__(self, x, state=None, mode='train'):
      # format it as batch for vmap
      # scan = jnp.ones((x.shape[0]), dtype=jnp.int32) if mode=='scan' else None
      for l in range(self.n_layers):
        x, state = self.get(f'layer_{l}', GeneralSequenceLayer,
          ssm=self.init_fn, dropout=self.dropout, d_model=self.d_model, activation=self.act,
          prenorm=self.prenorm, batchnorm=self.batchnorm, bn_momentum=self.bn_momentum
        )(x, state, mode)
      return x, state



#################################################
# General-RSSM Model using Deep State-Space Layers
# Implemented with ninjax
#################################################
class General_RSSM(nj.Module):
  def __init__(
    self, ssm_size, blocks, d_model, n_layers, ssm_post_act, discretization,
    dt_min, dt_max, conj_sym, clip_eigs, bidirectional, C_init,
    dropout, prenorm, batchnorm, bn_momentum, 
    deter=1024, stoch=32, classes=32, unroll=False,
    initial='learned', unimix=0.01, action_clip=1.0, **kw
  ):
    self._deter = deter
    self._stoch = stoch
    self._classes = classes
    self._unroll = unroll
    self._initial = initial
    self._unimix = unimix
    self._action_clip = action_clip
    self._kw = kw

    # HiPPO Matrix set-up args
    self.ssm = 's5'
    self.blocks = blocks
    self.ssm_size = ssm_size
    self.init_block_size = int(self.ssm_size/self.blocks)
    self.conj_sym = conj_sym
    self.ssm_size = self.ssm_size//2 if self.conj_sym else self.ssm_size

    # Discretization and initialization args
    self.C_init = C_init
    self.discretization = discretization
    self.dt_min = dt_min
    self.dt_max = dt_max
    self.clip_eigs = clip_eigs
    self.bidirectional = bidirectional

    # General Sequence Layer args
    self.n_layers = n_layers
    self.dropout = dropout
    self.d_model = d_model
    self.act = ssm_post_act
    self.prenorm = prenorm
    self.batchnorm = batchnorm
    self.bn_momentum = bn_momentum

    # Calling this function instantiates a single layer
    init_fn = SSM_MODELS[self.ssm](
      blocks=self.blocks, init_block_size=self.init_block_size, H=self.d_model, P=self.ssm_size, C_init=self.C_init, discretization=self.discretization,
      dt_min=self.dt_min, dt_max=self.dt_max, conj_sym=self.conj_sym, clip_eigs=self.clip_eigs, bidirectional=self.bidirectional, 
    )
    self.ssm_args = {'init_fn': init_fn, 'n_layers': self.n_layers, 'dropout': self.dropout, 'd_model': self.d_model,
      'act': self.act, 'prenorm': self.prenorm, 'batchnorm': self.batchnorm, 'bn_momentum': self.bn_momentum}



  def initial(self, bs):

    # Define the initial state
    if self._classes:
      state = dict(
          deter=jnp.zeros([bs, self._deter], f32),  # deterministic output
          # hidden=jnp.zeros([bs, self.ssm_size], jnp.complex64), # hidden state
          hidden_re=jnp.zeros([bs, self.ssm_size], f32), # hidden state
          hidden_im=jnp.zeros([bs, self.ssm_size], f32), # hidden state
          logit=jnp.zeros([bs, self._stoch, self._classes], f32), # predictions
          stoch=jnp.zeros([bs, self._stoch, self._classes], f32)) # stochastic output
    else:
      state = dict(
          deter=jnp.zeros([bs, self._deter], f32),
          # hidden=jnp.zeros([bs, self.ssm_size], jnp.complex64),
          hidden_re=jnp.zeros([bs, self.ssm_size], f32),
          hidden_im=jnp.zeros([bs, self.ssm_size], f32),
          mean=jnp.zeros([bs, self._stoch], f32),
          std=jnp.ones([bs, self._stoch], f32),
          stoch=jnp.zeros([bs, self._stoch], f32))
    if self._initial == 'zeros':
      return cast(state)
    elif self._initial == 'learned':
      # deter = self.get('initial', jnp.zeros, state['deter'][0].shape, f32)
      deter = self.get('initial', jnp.zeros, state['deter'][0].shape, jnp.complex64)
      state['deter'] = jnp.repeat(jnp.tanh(deter)[None], bs, 0)
      if self.ssm=='':
        # the stochastic component comes from the hidden state
        state['stoch'] = self.get_stoch(cast(state['deter']))
      else:
        # the stochatic component comes from our output
        state['stoch'] = self.get_stoch(cast(jnp.zeros([bs, self.ssm_size], f32)))
      return cast(state)
    else:
      raise NotImplementedError(self._initial)


  def observe(self, embed, action, is_first, state=None):
    seq_len = action.shape[1]
    expand_to_seq = lambda x: jnp.repeat(x[:, None], seq_len, 1)
    if state is None:
      state = self.initial(action.shape[0])

    # check if deterministic state is initialized with zeros
    # assert jnp.all(state['deter']==0), state['deter']

    # Make sure that everything is right format and within bounds
    prev_state = state
    is_first = cast(is_first)
    prev_action = cast(action)
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))
    print(f'embed shape and action shape {embed.shape} {action.shape}')
    print(f'is first shape {is_first.shape}')
    print(f'prev_state shape {prev_state["deter"].shape}')
    # raise Exception
    # is first : {16,64}
    # embed and action : {16,64,dim}


    # Expand our init variables to match sequence length
    # prev_state = jax.tree_util.tree_map(expand_to_seq, (prev_state))
    # init_seq = jax.tree_util.tree_map(expand_to_seq, (self.initial(len(is_first))))
    # prev_action, prev_state = jax.tree_util.tree_map(   # batched sequence mask
    #   lambda x: self._mask_sequence(x, 1.0 - is_first), (prev_action, prev_state))
    # # match type of prev state components
    # prev_state = jax.tree_util.tree_map(
    #     lambda x, y: x + self._mask_sequence(y, is_first), prev_state, init_seq) 


    # 1. Calculate the prior
    # prior = self.imagine(prev_action, prev_state)
    prior = self.action_imagine(prev_action, prev_state)
    # 2. Calculate the posterior
    x = jnp.concatenate([prior['deter'], embed], -1)  # deter component is out of ssm
    x = self.get('obs_out', Linear, **self._kw)(x)
    stats = self._stats('obs_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    post = cast({'stoch': stoch, 'deter': prior['deter'], 'hidden_re': prior['hidden_re'], 'hidden_im': prior['hidden_im'], **stats})
    raise Exception('obs ckpt')
    return post, prior
  
  def action_imagine(self, action, prev_state):
    prev_action = action
    prev_hidden = prev_state['hidden_re']+1j*prev_state['hidden_im']
    x = jnp.concatenate([prev_action], -1)
    x = self.get('img_in', Linear, **self._kw)(x)
    print(f'shapes prev action prev hidden {prev_action.shape} {prev_hidden.shape}')
    out, deter = self.get('ssm', StackedSSM, **self.ssm_args)(x, prev_hidden)
    x = self.get('img_out', Linear, **self._kw)(out)
    stats = self._stats('img_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    prior = cast({'stoch': stoch, 'deter': out, 'hidden_re': deter.real, 'hidden_im': deter.imag, **stats})
    return cast(prior)


  def imagine(self, action, state=None):
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    state = self.initial(action.shape[0]) if state is None else state
    assert isinstance(state, dict), state

    prev_stoch = state['stoch']  # img step checkpoint
    prev_action = action
    if self._classes:
      shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
      prev_stoch = prev_stoch.reshape(shape)
    if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
      shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
      prev_action = prev_action.reshape(shape)

    # Prior takes in previous output and input action
    x = jnp.concatenate([prev_stoch, prev_action], -1)
    # Model dim encoder
    x = self.get('img_in', Linear, **self._kw)(x)
    # Sequence Model batched scan
    out, deter = self.get('ssm', StackedSSM, **self.ssm_args)(x, state=None)
    # Model decoder
    x = self.get('img_out', Linear, **self._kw)(out)
    # Compute scans through ssm and prior 
    stats = self._stats('img_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    prior = cast({'stoch': stoch, 'deter': out, 'hidden_re': deter.real, 'hidden_im': deter.imag, **stats})
    return cast(prior)
  

  def obs_step(self, prev_state, prev_action, embed, is_first):
    is_first = cast(is_first)
    prev_action = cast(prev_action)
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))
    prev_state, prev_action = jax.tree_util.tree_map(lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
    prev_state = jax.tree_util.tree_map(lambda x, y: x + self._mask(y, is_first), prev_state, self.initial(len(is_first)))
    prior = self.img_step(prev_state, prev_action)
    x = jnp.concatenate([prior['deter'], embed], -1)
    x = self.get('obs_out', Linear, **self._kw)(x)
    stats = self._stats('obs_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    post = cast({'stoch': stoch, 'deter': prior['deter'], **stats})
    return post, prior
  

  def img_step(self, prev_state, prev_action):
    prev_stoch = prev_state['stoch']
    prev_action = cast(prev_action)
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))
    if self._classes:
      shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
      prev_stoch = prev_stoch.reshape(shape)
    if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
      shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
      prev_action = prev_action.reshape(shape)
    # Prior takes in previous output and input action
    x = jnp.concatenate([prev_stoch, prev_action], -1)

    # Model dim encoder
    x = self.get('img_in', Linear, **self._kw)(x)
    deter = prev_state['hidden_re'] + 1j*prev_state['hidden_im']
    # Inference step with SSM
    out, deter = self.get('ssm', StackedSSM, **self.ssm_args)(x=x, state=deter, mode='step')
    # 1. Calculate the prior
    x = self.get('img_out', Linear, **self._kw)(out)
    stats = self._stats('img_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    prior = {'stoch': stoch, 'deter': out, 'hidden_re': deter.real, 'hidden_im': deter.imag, **stats}
    return cast(prior)



  def get_dist(self, state, argmax=False):
    if self._classes:
      logit = state['logit'].astype(f32)
      return tfd.Independent(jaxutils.OneHotDist(logit), 1)
    else:
      mean = state['mean'].astype(f32)
      std = state['std'].astype(f32)
      return tfd.MultivariateNormalDiag(mean, std)


  def get_stoch(self, deter):
    x = self.get('img_out', Linear, **self._kw)(deter)
    stats = self._stats('img_stats', x)
    dist = self.get_dist(stats)
    return cast(dist.mode())
  

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

  def _mask(self, value, mask):
    return jnp.einsum('b...,b->b...', value, mask.astype(value.dtype))

  def _mask_sequence(self, value, mask):
    return jnp.einsum('b l ..., b l -> b l ...', value, mask.astype(value.dtype))

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
  
