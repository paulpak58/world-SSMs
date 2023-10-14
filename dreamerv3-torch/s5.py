import torch
import torch.nn as nn


def _make_HiPPO(N):
    # P = torch.sqrt(1 + 2 * torch.arange(N, dtype=torch.float32))  # N = state size
    P = torch.sqrt(1 + 2 * torch.arange(N))
    A = P[:, None] * P[None, :]
    A = torch.tril(A) - torch.diag(torch.arange(N))
    return -A  # (N, N) HiPPO LegS matrix


def _make_NPLR_HiPPO(N):
    hippo = _make_HiPPO(N)
    P = torch.sqrt(torch.arange(N) + 0.5)
    B = torch.sqrt(2 * torch.arange(N) + 1.0)
    # (N, N) HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B
    return hippo, P, B  


def make_DPLR_HiPPO(N):
    A, P, B = _make_NPLR_HiPPO(N)
    S = A + P[:, None] * P[None, :]
    S_diag = torch.diagonal(S)
    Lambda_real = torch.mean(S_diag) + torch.ones_like(S_diag)
    Lambda_imag, V = torch.linalg.eigh(S*-1j)  # Diagonalize S to V Lambda V^*
    P = V.conj().T @ P.to(torch.complex64)
    B = V.conj().T @ B.to(torch.complex64)
    # Eigenvalues Lambda, low-rank factor P, conjugated HiPPO input matrix B, eigenvectors V
    return Lambda_real + 1j * Lambda_imag, V


def discretize_zoh(Lambda, B, step):
    Lambda_bar = torch.exp(Lambda*step)
    B_bar = (1/Lambda)*(Lambda_bar-torch.ones_like(Lambda_bar))[..., None]*B
    return Lambda_bar, B_bar


def parallel_scan(Ls, us):
    return Ls, us 



class S5(nn.Module):
    def __init__(self, blocks, ssm_size, d_model, conj_sym, clip_eigs, dt_max, dt_min, step_rescale):
        super().__init__()
        self.ssm_size = ssm_size
        self.conj_sym = conj_sym
        self.clip_eigs = clip_eigs
        
        self.dt_max = dt_max
        self.dt_min = dt_min
        self.step_rescale = step_rescale

        block_size = ssm_size//blocks

        Lambda, V = make_DPLR_HiPPO(N=block_size)
        self.block_size = block_size // 2 if self.conj_sym else block_size

        Lambda = Lambda[: self.block_size]
        V = V[:, : self.block_size]

        Lambda = (Lambda * torch.ones((blocks, self.block_size))).flatten()
        self.Lambda_re = nn.Parameter(Lambda.real)
        self.Lambda_imag = nn.Parameter(Lambda.imag)
        self.V = nn.Parameter(torch.block_diag(*[V] * blocks))
        self.Vinv = nn.Parameter(torch.block_diag(*[V.conj().T] * blocks))

        local_ssm_size = ssm_size * 2 if conj_sym else ssm_size
        input_to_state = (local_ssm_size, d_model)
        state_to_output = (d_model, local_ssm_size, 2)
        feedthrough = (d_model,)

        self.B = nn.Parameter(torch.empty(input_to_state))
        torch.nn.init.xavier_normal_(self.B)

        C = torch.normal(mean=0.0, std=1.0, size=state_to_output)
        CV = (C[..., 0] + 1j*C[..., 1]) @ self.V
        self.C = nn.Parameter(torch.concatenate((CV.real[..., None], CV.imag[..., None]), axis=-1))
        self.D = nn.Parameter(torch.randn(feedthrough))
        self.step = nn.Parameter(torch.log(torch.rand((d_model, 1)) * (dt_max-dt_min) + dt_min))

    def forward(self, u):
        if self.clip_eigs:
            Lambda = torch.clamp(self.Lambda_re, max=-1e-4) + 1j*self.Lambda_imag
        else:
            Lambda = self.Lambda_re + 1j*self.Lambda_imag

        B_tilde = self.B[..., 0] + 1j*self.B[..., 1]

        Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, self.step)

        Lambda = Lambda[None, None, ...].repeat((u.shape[0], u.shape[1], 1))

        # Lambda_bar = Lambda_bar[None, ...] * torch.ones((u.shape[0], 1, self.block_size))
        print(f'u and b bar shape {u.shape} {B_bar.shape}')
        Bu = torch.matmul(u, B_bar.T)    # (B,L,d_model)*(d_model,ssm_size)

        _, state = parallel_scan(Lambda_bar, Bu)

        return state
        


"""
class StackedSSM(nn.Module): 
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

      self.layers = []
      for i in range(n_layers):
          #TODO

    def __call__(self, x, state=None, mode='train'):
      for l in range(self.n_layers):
        x, state = self.get(f'layer_{l}', GeneralSequenceLayer,
          ssm=self.init_fn, dropout=self.dropout, d_model=self.d_model, activation=self.act,
          prenorm=self.prenorm, batchnorm=self.batchnorm, bn_momentum=self.bn_momentum
        )(x, state, mode)
      return x, state
"""

def main():
    blocks = 8
    ssm_size = 256
    d_model = 512
    conj_sym = False
    clip_eigs = True
    dt_max = 0.1
    dt_min = 0.001
    step_rescale = 1.0

    L = 64
    bsz = 4

    us = torch.randn((L, d_model))
    model = S5(blocks, ssm_size, d_model, conj_sym, clip_eigs, dt_max, dt_min, step_rescale)
    print(model(us))
    for name,param in model.named_parameters():
        print(f'Name {name}')

if __name__ == "__main__":
    main()
