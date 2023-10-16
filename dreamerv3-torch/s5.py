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
    B_bar = ((1/Lambda)*(Lambda_bar-1))[..., None]*B
    return Lambda_bar, B_bar

def parallel_scan(Ls, us):
    return Ls, us 


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "softplus":
        return nn.Softplus()
    elif activation == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation function {activation}")





class S5(nn.Module):
    def __init__(self, blocks, ssm_size, d_model, conj_sym, clip_eigs, dt_max, dt_min, step_rescale, activation="relu"):
        super().__init__()
        self.ssm_size = ssm_size
        self.conj_sym = conj_sym
        self.clip_eigs = clip_eigs
        
        self.dt_max = dt_max
        self.dt_min = dt_min
        self.step_rescale = step_rescale
        self.activation = activation

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

        # self.B = nn.Parameter(torch.empty(input_to_state))
        # torch.nn.init.xavier_normal_(self.B)
        B = torch.randn(input_to_state)
        VinvB = self.Vinv @ B.to(torch.complex64)
        self.B = nn.Parameter(torch.concatenate([VinvB.real[..., None], VinvB.imag[..., None]], axis=-1))

        C = torch.normal(mean=0.0, std=1.0, size=state_to_output)
        CV = (C[..., 0] + 1j*C[..., 1]) @ self.V
        self.C = nn.Parameter(torch.concatenate((CV.real[..., None], CV.imag[..., None]), axis=-1))
        self.D = nn.Parameter(torch.randn(feedthrough))
        self.step = nn.Parameter(torch.log(torch.rand((local_ssm_size, 1)) * (dt_max-dt_min) + dt_min))

    def forward(self, u):
        if self.clip_eigs:
            Lambda = torch.clamp(self.Lambda_re, max=-1e-4) + 1j*self.Lambda_imag
        else:
            Lambda = self.Lambda_re + 1j*self.Lambda_imag

        B_tilde = self.B[..., 0] + 1j*self.B[..., 1]
        C_tilde = self.C[..., 0] + 1j*self.C[..., 1]

        step = self.step_rescale * torch.exp(self.step.squeeze())

        Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)

        Lambda = Lambda[None, None, ...].repeat((u.shape[0], u.shape[1], 1))

        Bu = u.to(torch.complex64) @ B_bar.T    # (L,bsz,d)x(d,ssm_size)

        _, state = parallel_scan(Lambda_bar, Bu)    # (L,bsz,ssm_size)x(L,bsz,ssm_size)

        y = torch.matmul(state, C_tilde.T).real + self.D*u

        y = self.act(y)

        return y, state
    


class StackedSSM(nn.Module): 
    def __init__(
        self, ssm_layer, n_layers, **ssm_args
    #   self, ssm_layer, n_layers, dropout, d_model, act, prenorm, batchnorm, bn_momentum,
    ):
      self.n_layers = n_layers
      # ssm layer attributes
      self.ssm_layer = ssm_layer

      self.layers = []
      for i in range(n_layers):
        self.layers.append(ssm_layer(**ssm_args))

    def forward(self, x, state=None, mode='train'):
      for i in range(self.n_layers):
        x, state = self.layers[i](x, state, mode)
      return x, state


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

    us = torch.randn((L, bsz, d_model))
    model = S5(blocks, ssm_size, d_model, conj_sym, clip_eigs, dt_max, dt_min, step_rescale)
    print(model(us))
    for name,param in model.named_parameters():
        print(f'Name {name}')

if __name__ == "__main__":
    main()
