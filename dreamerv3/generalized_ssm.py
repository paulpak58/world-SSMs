import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from functools import partial
from tensorflow_probability.substrates import jax as tfp

from . import jaxutils
from . import ninjax as nj
from . import nets

from initializers import make_DPLR_HiPPO
from s5 import S5LayerInit
from s4 import S4LayerInit
from dss import DSSLayerInit

f32 = jnp.float32
tfd = tfp.distributions
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute
Linear = nets.Linear

SSM_MODELS = {
  's4': S4LayerInit,
  'dss': DSSLayerInit,
  's5': S5LayerInit
}


def build_ssm(config):

  if config.ssm =='s5':

    ssm_size = config.s5.ssm_size
    block_size = int(ssm_size/config.s5.blocks)  # size of initial blocks
    # padded = False
    # in_dim = config.embed_dim
    # train_size = config.train_size

    # Initialize DPLR HiPPO matrix
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)
    if config.s5.conj_sym: # conj. pairs halve the state space
      block_size = block_size//2
      ssm_size = ssm_size//2
    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    # Put each HiPPO on each block diagonal
    Lambda = (Lambda * jnp.ones((config.s5.blocks, block_size))).ravel()
    V = jax.scipy.linalg.block_diag(*[V]*config.s5.blocks)
    Vinv = jax.scipy.linalg.block_diag(*[Vc]*config.s5.blocks)
    # Initializes the SSM layer
    init_fn = SSM_MODELS[config.ssm](
      H=config.s5.d_model, P=config.s5.ssm_size, Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
      V=V, Vinv=Vinv, C_init=config.s5.C_init, discretization=config.s5.discretization, dt_min=config.s5.dt_min,
      dt_max=config.s5.dt_max, conj_sym=config.s5.conj_sym, clip_eigs=config.s5.clip_eigs, bidirectional=config.s5.bidirectional,
    )
    # Creates full batched model based on the SSM layer
    model = partial(
      BatchedSequenceModel,
      ssm=init_fn,
      d_output=config.s5.seq_len, #TODO
      d_model=config.s5.d_model,
      n_layers=config.s5.n_layers,
      activation=config.s5.act,
      dropout=config.s5.dropout,
      prenorm=config.s5.prenorm,
      batchnorm=config.s5.batchnorm,
      bn_momentum=config.s5.bn_momentum
    )
    ssm = General_RSSM(model, name='s5')

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
# Dropout, and Nonlinearity
##############################################
class GeneralSequenceLayer(nn.Module):
  ssm: nn.Module
  dropout: float
  d_model: int
  activation: str='gelu'
  training: bool=True
  prenorm: bool=False
  batchnorm: bool=False
  bn_momentum: float=0.9
  step_rescale: float=1.0

  def setup(self):
    self.seq = self.ssm(step_rescale=self.step_rescale)
    if self.activation in ['full_glu']:
      self.out1 = nn.Dense(self.d_model)
      self.out2 = nn.Dense(self.d_model)
    elif self.activation in ['half_glu1', 'half_glu2']:
      self.out2 = nn.Dense(self.d_model)
    if self.batchnorm:
      self.norm = nn.BatchNorm(use_running_average=not self.training, momentum=self.bn_momentum, axis_name='batch')
    else:
      self.norm = nn.LayerNorm()
    self.drop = nn.Dropout(
      self.dropout,
      broadcast_dims=[0],
      deterministic=not self.training
    )

  def __call__(self, x):
    # Takes in (L, d_model) and outputs (L, d_model)
    skip = x
    if self.prenorm:
      x = self.norm(x)
    x, state = self.seq(x)
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
      x = self.drop(nn.gelu(x))
    else:
      raise NotImplementedError(f'Activation {self.activation} not implemented')
    x = skip + x
    if not self.prenorm:
      x = self.norm(x)
    # return x, state
    return x


##############################
# Stacked Deep SSM Layers
# Implemented in ninjax
##############################
class StackedSSM(nj.Module):
    
    def __init__(
      self, init_fn, n_layers, dropout, d_model, act, prenorm, batchnorm, bn_momentum
    ):
      seq_layer = GeneralSequenceLayer(
        ssm=init_fn, dropout=dropout, d_model=d_model, activation=act,
        prenorm=prenorm, batchnorm=batchnorm, bn_momentum=bn_momentum,
      )
      self.n_layers = n_layers
      # ssm layer attributes
      self.init_fn = init_fn
      self.dropout = dropout
      self.d_model = d_model
      self.act = act
      self.prenorm = prenorm
      self.batchnorm = batchnorm
      self.bn_momentum = bn_momentum


    def __call__(self, x, deter=None):
      # x = jnp.concatenate([deter, x], -1)
      for l in range(self.n_layers):
        # x = nj.FlaxModule(
        #   GeneralSequenceLayer, ssm=self.init_fn, dropout=self.dropout, d_model=self.d_model, activation=self.act,
        #   prenorm=self.prenorm, batchnorm=self.batchnorm, bn_momentum=self.bn_momentum,
        #   name=f'layer_{l}'
        # )(x)
        x = nj.FlaxModule(
          BatchedSequenceLayer, ssm=self.init_fn, dropout=self.dropout, d_model=self.d_model, activation=self.act,
          prenorm=self.prenorm, batchnorm=self.batchnorm, bn_momentum=self.bn_momentum,
          name=f'layer_{l}'
        )(x)
      # for l in range(self.n_layers):
      #   x = self.get(f'layer_{l}', nj.FlaxModule(self.ssm))(x)
      #   # x = nj.FlaxModule(self.ssm, x, name=f'layer_{l}')
      #   # x = self.ssm(x)
      #   x = 
      return x



#############################################
# Stacked Encoder Model as implemented by
#############################################
class GeneralSequenceModel(nn.Module):
  ssm: nn.Module
  d_output: int
  d_model: int
  n_layers: int
  activation: str='gelu'
  dropout: float=0.0
  training: bool=True
  mode: str=''
  prenorm: bool=False
  batchnorm: bool=False
  bn_momentum: float=0.9
  step_rescale: float=1.0

  def setup(self):
    # Initializes linear encoder and stack of S5 layers
    self.encoder = nn.Dense(self.d_model)
    self.layers = [
      GeneralSequenceLayer(
        ssm=self.ssm,
        dropout=self.dropout,
        d_model=self.d_model,
        activation=self.activation,
        training=self.training,
        prenorm=self.prenorm,
        batchnorm=self.batchnorm,
        bn_momentum=self.bn_momentum,
        step_rescale=self.step_rescale
      ) for _ in range(self.n_layers)
    ]
    self.decoder = nn.Dense(self.d_output)

  def __call__(self, x):
    # In: (L, d_input), Out: (L, d_model)
    if self.padded:
      x, length = x
    x = self.encoder(x)
    for l in self.layers:
      x = l(x)
    if self.mode in ['pool']:
      if self.padded:
        x = masked_meanpool(x, length)
      else:
        x = jnp.mean(x, axis=0)
    elif self.mode in ['last']:
      if self.padded:
        raise NotImplementedError(f'Mode must be in pool for self.padded=True')
      else:
        x = x[-1]
    else:
      x = x 
    x = self.decoder(x)
    x = nn.log_softmax(x, axis=-1)
    return x


############################################
# Final batched version to use in the pipeline
############################################
BatchedSequenceModel= nn.vmap(
  GeneralSequenceModel,
  in_axes=(0, 0),
  out_axes=0,
  variable_axes={'params':None, 'dropout':None, 'batch_stats':None, 'cache':0, 'prime':None},
  split_rngs={'params': False, 'dropout': True},
  axis_name='batch'
)

# BatchedSequenceLayer = nn.vmap(
#   GeneralSequenceLayer,
#   in_axes=0,
#   out_axes=0,
#   variable_axes={'params':None, 'dropout':None, 'batch_stats':None, 'cache':0, 'prime':None},
#   split_rngs={'params': False, 'dropout': True},
#   axis_name='batch'
# )
BatchedSequenceLayer = nn.vmap(
  GeneralSequenceLayer,
  in_axes=0,
  out_axes=0,
  variable_axes={'params':None, 'dropout':None, 'cache':0, 'prime':None},
  split_rngs={'params': False, 'dropout': True},
  axis_name='batch'
)



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

    # self.ssm = nj.FlaxModule(ssm, name='ssm')
    self.ssm = 's5'
    self.n_layers = n_layers
    self.ssm_size = ssm_size
    self.blocks = blocks
    self.block_size = int(self.ssm_size/self.blocks)
    self.d_model = d_model

    self.dt_min = dt_min
    self.dt_max = dt_max
    self.conj_sym = conj_sym
    self.clip_eigs = clip_eigs
    self.bidirectional = bidirectional
    self.C_init = C_init
    self.discretization = discretization

    self.act = ssm_post_act
    self.dropout = dropout
    self.prenorm = prenorm
    self.batchnorm = batchnorm
    self.bn_momentum = bn_momentum

  def init_ssm(self, bs):

    # Initialize DPLR HiPPO matrix
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(self.block_size)
    if self.conj_sym: # conj. pairs halve the state space
      self.block_size = self.block_size//2
      self.ssm_size = self.ssm_size//2
    Lambda = Lambda[:self.block_size]
    V = V[:, :self.block_size]
    Vc = V.conj().T

    # Put each HiPPO on each block diagonal
    Lambda = (Lambda * jnp.ones((self.blocks, self.block_size))).ravel()
    V = jax.scipy.linalg.block_diag(*[V]*self.blocks)
    Vinv = jax.scipy.linalg.block_diag(*[Vc]*self.blocks)
    # Initializes the SSM layer
    init_fn = SSM_MODELS[self.ssm](
      H=self.d_model, P=self.ssm_size, Lambda_re_init=Lambda.real, Lambda_im_init=Lambda.imag,
      V=V, Vinv=Vinv, C_init=self.C_init, discretization=self.discretization, dt_min=self.dt_min,
      dt_max=self.dt_max, conj_sym=self.conj_sym, clip_eigs=self.clip_eigs, bidirectional=self.bidirectional,
    )
    return init_fn
 

  def initial(self, bs):
    # Define the initial state                
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


  def observe(self, embed, action, is_first, state=None):
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    if state is None:
      state = self.initial(action.shape[0])
    step = lambda prev, inputs: self.obs_step(prev[0], *inputs)
    inputs = swap(action), swap(embed), swap(is_first)
    start = state, state

    #### part of scan
    length = len(jax.tree_util.tree_leaves(inputs)[0])
    expand_to_seq = lambda x: jnp.repeat(x[:, None], length, 1)
    carrydef = jax.tree_util.tree_structure(start)
    carry = start
    outs = []
    print(f'length {length}')
    print(carrydef)

    # action, embed, is_first = swap(action), swap(embed), swap(is_first)
    prev_action = action
    prev_state = start[0]
    is_first = cast(is_first)
    prev_action = cast(prev_action)
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))

    # print(f'shapes {prev_action.shape} {is_first.shape}')
    print(f'prev state shape {prev_state["deter"].shape}')
    print(f'prev action shape {prev_action.shape}')
    print(f'is first shape {is_first.shape}')
    # expand state to length so we have (B,L,D)
    # prev_state = jax.tree_util.tree_map(
    #   lambda x: jnp.repeat(x[:, None], length, 1), (prev_state))
    prev_state = jax.tree_util.tree_map(expand_to_seq, (prev_state))
    init_seq = jax.tree_util.tree_map(expand_to_seq, (self.initial(len(is_first))))
    prev_action, prev_state = jax.tree_util.tree_map(
      lambda x: self._mask_sequence(x, 1.0 - is_first), (prev_action, prev_state))
    prev_state = jax.tree_util.tree_map(
        lambda x, y: x + self._mask_sequence(y, is_first), prev_state, init_seq) 
    prev_stoch = prev_state['stoch']  # img step checkpoint
    if self._classes:
      shape = prev_stoch.shape[:-2] + (self._stoch * self._classes,)
      prev_stoch = prev_stoch.reshape(shape)
    if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
      shape = prev_action.shape[:-2] + (np.prod(prev_action.shape[-2:]),)
      prev_action = prev_action.reshape(shape)
    x = jnp.concatenate([prev_stoch, prev_action], -1)
    print(f'x shape: {x.shape}') 

    # Full forward pass into S5
    x = self.get('img_in', Linear, **self._kw)(x)
    init_fn = self.init_ssm(x.shape[0])
    ssm_args = {
      'init_fn': init_fn, 'n_layers': self.n_layers, 'dropout': self.dropout, 'd_model': self.d_model,
      'act': self.act, 'prenorm': self.prenorm, 'batchnorm': self.batchnorm, 'bn_momentum': self.bn_momentum
    }
    x = self.get('ssm', StackedSSM, **ssm_args)(x)
    # x, deter = self.get('ssm', StackedSSM, **ssm_args)(x)
    print(f'x shape: {x.shape}')
    # print(f'deter shape: {deter.shape}')
    raise Exception('c')
    post, prior = jaxutils.scan(step, inputs, start, self._unroll)

    post = {k: swap(v) for k, v in post.items()}
    prior = {k: swap(v) for k, v in prior.items()}
    # print(f'[*] OBSERVE TRAJECTORY Post keys {post.keys()}')
    return post, prior

  

  def imagine(self, action, state=None):
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    state = self.initial(action.shape[0]) if state is None else state
    assert isinstance(state, dict), state
    action = swap(action)
    prior = jaxutils.scan(self.img_step, action, state, self._unroll)
    prior = {k: swap(v) for k, v in prior.items()}
    # print(f'[*] IMAGINE TRAJECTORY Prior keys {prior.keys()}')
    return prior
  

  def obs_step(self, prev_state, prev_action, embed, is_first):
    is_first = cast(is_first)
    prev_action = cast(prev_action)
    if self._action_clip > 0.0:
      prev_action *= sg(self._action_clip / jnp.maximum(
          self._action_clip, jnp.abs(prev_action)))
    prev_state, prev_action = jax.tree_util.tree_map(
        lambda x: self._mask(x, 1.0 - is_first), (prev_state, prev_action))
    prev_state = jax.tree_util.tree_map(
        lambda x, y: x + self._mask(y, is_first),
        prev_state, self.initial(len(is_first)))
    prior = self.img_step(prev_state, prev_action)


    x = jnp.concatenate([prior['deter'], embed], -1)
    x = self.get('obs_out', Linear, **self._kw)(x)
    stats = self._stats('obs_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    post = {'stoch': stoch, 'deter': prior['deter'], **stats}
    # print(f'[*] OBSERVE STEP posterior stochastic shape {post["stoch"].shape}')
    # print(f'[*] OBSERVE STEP posterior deterministic shape {post["deter"].shape}')
    return cast(post), cast(prior)
  

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
    x = jnp.concatenate([prev_stoch, prev_action], -1)
    x = self.get('img_in', Linear, **self._kw)(x)

    ##########################
    # S5 Single Step
    ##########################
    init_fn = self.init_ssm(x.shape[0])
    ssm_args = {
      'init_fn': init_fn, 'n_layers': self.n_layers, 'dropout': self.dropout, 'd_model': self.d_model,
      'act': self.act, 'prenorm': self.prenorm, 'batchnorm': self.batchnorm, 'bn_momentum': self.bn_momentum
    }
    x, deter = self.get('ssm', StackedSSM, **ssm_args)(x)


    # x, deter = self._gru(x, prev_state['deter'])
    print(f'x shape {x.shape}')
    x = self.get('img_out', Linear, **self._kw)(x)
    raise Exception('ckpt')
  

    stats = self._stats('img_stats', x)
    dist = self.get_dist(stats)
    stoch = dist.sample(seed=nj.rng())
    prior = {'stoch': stoch, 'deter': deter, **stats}
    # print(f'[*] IMAGINE STEP prior stochastic shape {prior["stoch"].shape}')
    # print(f'[*] IMAGINE STEP prior deterministic shape {prior["deter"].shape}')
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
  