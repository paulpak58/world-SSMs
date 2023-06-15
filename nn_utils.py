import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange




class SquaredReLU(nn.Module):
    def forward(self, x):
        return torch.square(F.relu(x))
    

class Laplace(nn.Module):
    def __init__(self, mu=0.707107, sigma=0.282095):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        x = (x-self.mu).div(self.sigma*math.sqrt(2.0))
        return 0.5*(1.0+torch.erf(x))
    

class TransposedLN(nn.Module):
    ''' Assumes (B,D,L); LayerNorm over 2nd dim '''
    def __init__(self, dim, scalar=True):
        super().__init__()
        self.scalar = scalar
        if self.scalar:
            self.m = nn.Parameter(torch.zeros(1))
            self.s = nn.Parameter(torch.ones(1))
            setattr(self.m, "_optim", {"weight_decay":0.0})
            setattr(self.s, "_optim", {"weight_decay":0.0})
        else:
            self.ln = nn.LayerNorm(dim)
        
    def forward(self, x):
        if self.scalar:
            # calculate statistics over D dim/channels
            s, m = torch.std_mean(x, dim=1, unbiased=False, keepdim=True)
            y = (self.s/s)*(x-m+self.m)
        else:
            # move channel to last axis, apply LN, move channel back
            x = self.ln(rearrange(x, 'b d ... -> b ... d'))
            y = rearrange(x, 'b ... d -> b d ...')
        return y



def Activation(activation=None, size=None, dim=-1):
    if activation in [None,'id','identity','linear']:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation=='gelu':
        return nn.GELU()
    elif activation in ['swish','silu']:
        return nn.SiLU()
    elif activation=='glu':
        return nn.GLU(dim=dim)
    elif activation=='sigmoid':
        return nn.Sigmoid()
    elif activation=='softplus':
        return nn.Softplus()
    elif activation=='modrelu':
        # TODO: https://github.com/Lezcano/expRNN
        return Modrelu(size)
    elif activation in ['sqrelu','relu2']:
        return SquaredReLU()
    elif activation=='laplace':
        return Laplace()
    elif activation=='ln':
        return TransposedLN(dim)
    else:
        raise NotImplementedError(f'Actiation not implemented {activation}')