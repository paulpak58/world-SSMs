import jax
import jax.numpy as jnp
import ninjax as nj
from flax import linen as nn
from functools import partial

from initializers import make_DPLR_HiPPO
from s5 import S5LayerInit
from s4 import S4LayerInit
from dss import DSSLayerInit

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
    nj_model = FlaxNinjaxWrapper(model, name='s5')

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
  print('Done')
  raise Exception('ckpt')


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
      broadcast_dim=[0],
      deterministic=not self.training
    )

  def __call__(self, x):
    # Takes in (L, d_model) and outputs (L, d_model)
    skip = x
    if self.prenorm:
      x = self.norm(x)
    x = self.seq(x)
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


############################################
# Ninjax wrapper allows us to pass state handling
# directly into our ninjax optimizer
############################################
class FlaxNinjaxWrapper(nj.Module):
  def __init__(self, flax_module: nn.Module):
    self.flax_module = flax_module
    self.ninjax_module = nj.FlaxModule(flax_module, name='flax_s5')

  def __call__(self, x):
    x = self.ninjax_module(x)


if __name__=='__main__':
  raise NotImplementedError