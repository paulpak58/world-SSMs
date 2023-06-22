# Adapted from https://github.com/facebookresearch/mega/blob/ea355255149d38ffe16bf2c176d47c3864e8b05a/fairseq/modules/moving_average_gated_attention.py
# And https://github.com/HazyResearch/state-spaces/blob/e9ce652126cc773dcb6bb7d6f7270c425d4a36a2/src/models/sequence/mega.py

import torch
import torch.nn as nn

from nn_utils import Activation

############################################
# Exponential Moving Average Gated Attention
#
############################################
class MegaBlock(nn.Module):
    ''' MovingAverageGatedAttention and MegaEncoderLayer as written by Albert G.'''
    def __init__(
        self,
        d_model,            # [mega] embed_dim
        d_attin,            # [mega] zdim
        d_attout,           # [mega] hdim
        d_state,            # [mega] ndim
        dropout=0.0,
        drop_attin=None,    # [mega] attention_dropout
        drop_attout=None,   # [mega] hidden_dropout
        activation='silu',
        attention_activation='softmax',
        bidirectional=False,
        chunk=-1,           # [mega] chunk_size
        l_max=None,         # [mega] truncation 
        norm='layer',       # [mega] norm_type
        prenorm=True,
        tie_dropout=False,  # [mega] feature_dropout
        rel_pos_bias='simple',
        max_positions=1024,
        ff_expand=2,        # [mega] encoder_ffn_embed_dim
        drop_ffn=True,      # [mega] activation_dropout
        transposed=False,   # inputs (B,L,D)
        mode='mega',
        **ssm_args
    ):
        # mode='mega': official Mega MultiHeadEMA class verbatim
        # Otherwise, construct a convolution kernel and use general SSM wrapper
        # mode='ema': use same core kernel code from MultiHeadEMA
        # mode=='nplr': use S4 kernel
        # mode=='diag': use S4D kernel
        # mode=='s5': use S5 kernel
        # mode=='liquid-S5': use liquid-S5 kernel
        super().__init__()
        self.transposed = transposed
        self.d_model = d_model
        self.d_output = d_model
        self.d_attin = d_attin
        self.d_attout = d_attout
        self.d_state = d_state
        self.activation = Activation(activation)
        self.attention_activation_fn = None if attention_activation=='softmax' else Activation(attention_activation)
        self.scaling = self.d_attin ** -0.5 if attention_activation=='softmax' else None

        # Dropout
        if drop_attin is None: drop_attin = dropout
        if drop_attout is None: drop_attout = dropout
        if drop_ffn is None: drop_ffn = dropout
        dropout_fn = partial(DropoutNd, transposed=False) if tie_dropout else nn.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_attout = dropout_fn(drop_attout) if drop_attout > 0.0 else nn.Identity()
        self.drop_attin = nn.Dropout(drop_attin)

        self.l_chunk = chunk
        self.prenorm = prenorm
        self.norm = Normalization(d_model, _name_=norm, transposed=False)


        # Linear SSM
        if mode=='mega':
            self.ssm = MultiHeadEMA(d_model, d_state=d_state, bidirectional=bidirectional, l_max=l_max)
        else:
            self.ssm = S4(d_model, d_state=d_state, bidirectional=bidirectional, l_max=l_max, activation=None, postact=None, mode=mode, transposed=False, **ssm_args)

        self.v_proj = nn.Linear(d_model, d_attout)  # U_v (eq. 10)
        self.mx_proj = nn.Linear(d_model, d_attin + d_attout + 2*d_model)
        self.h_proj = nn.Linear(d_attout, d_model)  # U_h (eq. 14)
        self.gamma = nn.Parameter(torch.Tensor(2, d_attin))
        self.beta = nn.Parameter(torch.Tensor(2, d_attin))

        self.max_positions = max_positions
        max_positions = max_positions if self.l_chunk<0 else self.l_chunk
        if rel_pos_bias=='simple':
            self.rel_pos_bias = SimpleRelativePositionalBias(max_positions)
        elif rel_pos_bias=='rotary':
            self.rel_pos_bias = RotaryRelativePositionalBias(d_attin, max_positions)
        else:
            raise ValueError(f'Unknown relative position bias {rel_pos_bias}')
        
        # Normalized Feed-Forward Network
        if ff_expand is not None and ff_expand>0:
            ffn_cfg = {
                '__name__': 'ff',
                'expand': ff_expand,
                'activation': activation,
                'dropout': drop_ffn,
                'tie_dropout': tie_dropout,
                'transposed': transposed
            }
            self.nffn = SequenceResidualBlock(
                d_model,
                prenorm=prenorm,
                dropout=dropout,
                tie_dropout=tie_dropout,
                residual='R',
                norm=norm,
                layer=ffn_cfg,
                transposed=transposed
            )
        else:
            self.nffn = None

        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.v_proj.bias, 0.0)
        nn.init.normal_(self.mx_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.mx_proj.bias, 0.0)
        nn.init.normal_(self.h_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.h_proj.bias, 0.0)
        nn.init.normal_(self.gamma, mean=0.0, std=std)
        nn.init.constant_(self.beta, 0.0)


    def initial(self, bs):
        pass

    
    def observe(self, embed, action, is_first, state=None):
        pass

    def imagine(self, action, state=None):
        pass

    def get_dist(self, state, argmax=False):
        pass


    def obs_step(self, prev_state, prev_action, embed, is_first):
        pass

    def img_step(self, prev_state, prev_action):
        pass

    def get_stoch(self, deter):
        pass


    def dyn_loss(self, post, prior, impl='kl', free=1.0):
        pass

    def rep_loss(self, post, prior, impl='kl', free=1.0):
        pass 


    ############ INNER TBD ############
    def _gru(self, x, deter):
        pass

    def _stats(self, name, x):
        pass

    def _mask(self, value, mask):
        pass




    def forward(
        self,
        x,
        state=None,
        padding_mask: Optional[torch.Tensor]=None,
        need_weights: bool=False,
        attn_mask: Optional[torch.Tensor]=None,
        before_attn_fn: bool=False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        '''
        Input shape: (B,L,D)
        B: batch size
        L: sequence length
        C: chunk size
        K: number of chunks (L/C)
        D: model dimension (d_model)
        V: dim attention output
        Z: dim attention input 
        '''
        if self.transposed: x = x.transpose(-1,-2)
        B, L, D = x.size()
        assert D==self.d_model

        residual = x
        if self.prenorm:
            x = self.norm(x)
        
        v = self.activation(self.v_proj(x))  # (B,L,V)
        mx, _ = self.ssm(x, state=state, padding_mask=padding_mask) # (B,L,D)

        mx = self.activation(mx)
        mx = self.dropout(mx)
        base = self.mx_proj(mx)  # (B,L,Z+V+2D)
        u, zr, hx = torch.split(base, [D, self.d_attin+self.d_attout, D], dim=1)
        u = torch.sigmoid(u)     # (B,L,D) (eq. 13)

        z, r = torch.split(self.activation(zr), [
            self.d_attin,           # z = (B,L,Z) (eq. 7)
            self.d_attout           # r = (B,L,V) (eq.12)
        ], dim=-1)
        z = z.unsqueeze(2) * self.gamma + self.beta
        q, k = torch.ubind(z, dim=2) # (B,L,Z) Q and K (eq.8 and 9)

        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        if self.l_chunk<0:
            if padding_mask is not None:
                padding_mask = padding_mask.unsqueeze(1)
        else:
            if L>=self.l_chunk:
                q = rearrange(q, 'b 1 (k c) z -> b k c z', c = self.l_chunk)
            
            l_ctx = k.size(2)
            if l_ctx<self.l_chunk:
                if padding_mask is not None:
                    padding_mask = padding_mask.unsqueeze(1)
            else:
                k = rearrange(k, 'b 1 (k c) z -> b k c z', c=self.l_chunk)
                v = rearrange(v, 'b 1 (k c) z -> b k c z', c=self.l_chunk)
                if padding_mask is not None:
                    padding_mask = rearrange(padding_mask, 'b (k c) -> b k c', c=self.l_chunk)

        # workaround to get around fork/join parallelism not supporting Optional
        if padding_mask is not None and padding_mask.dim()==0:
            padding_mask = None
        if self.attention_activation_fn is None:
            attn_weights = self.softmax_attention(q, k, padding_mask, attn_mask, before_attn_fn)
        else:
            attn_weights = self.element_attention(q, k, padding_mask, attn_mask, before_attn_fn)

        if before_attn_fn:
            if self.transposed: v = v.transpose(1,2)
            return v, attn_weights
        
        v = self.drop_attout(v) # (B,K,C,V)
        kernel = self.drop_attin(attn_weights) # (B,K,C,C)
        h = rearrange(torch.matmul(kernel,v), 'b k c v -> b (k c) v') # (B,L,V)
        h = self.activation(hx + self.h_proj(h*r)) # (B,L,D)
        h = self.dropout(h)
        out = torch.addcmul(residual, u, h-residual) # (B,L,D)

        if not self.prenorm:
            out = self.norm(out)

        if self.transposed:
            out = self.transpose(-1,-2)

        # FFN
        out, _ = self.nffn(out, state=None)

        if not need_weights: attn_weights = None
        return out, _   # , attn_weights
    
            




        
