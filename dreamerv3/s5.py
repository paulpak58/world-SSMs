# Revised and inspired from https://github.com/lindermanlab/S5/blob/liquid/s5/ssm.py
# Revision author: Paul Pak

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from functools import partial
from jax.nn.initializers import lecun_normal, normal


############################
# Creates HiPPO-LegS matrix
############################
def make_HiPPO(N):
    # N is the state size
    P = jnp.sqrt(1 + 2*jnp.arange(N))
    A = P[:, jnp.newaxis]*P[jnp.newaxis, :]
    A = jnp.tril(A) - jnp.diag(jnp.arange(N))
    # (N, N) HiPPO LegS matrix
    return -A


############################################
# Creates NPLR Representation of HiPPO-LegS matrix
############################################
def make_NPLR_HiPPO(N):
    # N is the state size
    hippo = make_HiPPO(N)
    P = jnp.sqrt(jnp.arange(N) + 0.5)
    B = jnp.sqrt(2*jnp.arange(N) + 1.0)
    # (N, N) HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B
    return hippo, P, B


############################################
# Creates DPLR Representation of HiPPO-LegS matrix
############################################
def make_DPLR_HiPPO(N):
    # N is the state size
    A, P, B = make_NPLR_HiPPO(N)
    S = A + P[:, jnp.newaxis]*P[jnp.newaxis, :]
    S_diag = jnp.diagonal(S)
    Lambda_real = jnp.mean(S_diag) + jnp.ones_like(S_diag)
    # Diagonalize S to V Lambda V^*
    Lambda_imag, V = jnp.linalg.eigh(S*-1j)
    P = V.conj().T@P
    B_orig = B
    B = V.conj().T@B
    # Eigenvalues Lambda, low-rank factor P, conjugated HiPPO input matrix B, eigenvectors V, original HiPPO input matrix B
    return Lambda_real + 1j*Lambda_imag, P, B, V, B_orig


############################
# SSM Initilization helper
############################
def trunc_standard_normal(key, shape):
    H, P, _ = shape
    Cs = []
    for i in range(H):
        key, skey = random.split(key)
        # sample C matrix
        C = lecun_normal()(skey, (1, P, 2))                                         
        Cs.append(C)
    # (H, P, 2)
    return jnp.array(Cs)[:, 0]


####################################
# Discretization Initilization helper
###################################
def init_log_steps(key, input):
    H, dt_min, dt_max = input
    dt_min = 0.001 if dt_min is None else dt_min
    dt_max = 0.1 if dt_max is None else dt_max
    log_steps = []
    for i in range(H):
        key, skey = random.split(key)
        log_step = random.uniform(skey, (1,))**(jnp.log(dt_max) - jnp.log(dt_min)) + jnp.log(dt_min)
        log_steps.append(log_step)
    # (H,)
    return jnp.array(log_steps)


########################
# Initialize C_tilde=CV
########################
def init_CV(init_fn, rng, shape, V):
    # Desired shape of matrix (H, P)
    C_ = init_fn(rng, shape)
    C = C_[..., 0] + 1j*C_[..., 1]
    # V is the eigenvectors of the diagonalized state matrix
    CV = C@V
    # (H, P, 2)
    return jnp.concatenate([CV.real[..., None], CV.imag[..., None]], axis=-1)


########################
# Initialize VinvB=VinvB
########################
def init_VinvB(init_fn, rng, shape, Vinv):
    # Desired shape of matrix (P, H)
    B = init_fn(rng, shape)
    Vinv = Vinv@B
    # (P, H, 2)
    return jnp.concatenate([Vinv.real[..., None], Vinv.imag[..., None]], axis=-1)


#############################################
# Discretize diagonalized, continuous-time SSM
# using bilinear transform
#############################################
def discretize_bilinear(Lambda, B_tilde, Delta):
    # Lambda: Diagonal state matrix (P,)
    Identity = jnp.ones(Lambda.shape[0])
    # B_tilde: Input matrix (P, H), Delta: Time step (P, )
    BL = 1/(Identity-(Delta/2.0)*Lambda)
    # Discretized diagonal state matrix (P, )
    Lambda_bar = (Identity+(Delta/2.0)*Lambda)*BL
    # Discretized input matrix (P, H)
    B_bar = (BL*Delta)[..., None]*B_tilde
    return Lambda_bar, B_bar


#############################################
# Discretize diagonalized, continuous-time SSM
# using zero-order hold
#############################################
def discretize_zoh(Lambda, B_tilde, Delta):
    # Lambda: Diagonal state matrix (P,)
    Identity = jnp.ones(Lambda.shape[0])
    # Lambda_bar: Discretized diagonal state matrix (P, )
    Lambda_bar = jnp.exp(Delta*Lambda)
    # B_bar: Discretized input matrix (P, H)
    B_bar = (1/Lambda*(Lambda_bar-Identity))[..., None]*B_tilde
    return Lambda_bar, B_bar


#############################################
# Binary operator for parallel scan of linear recurrence
###############################################
@jax.vmap
def binary_operator(element_i, element_j):
    # [(P,),(P,)], [(P,),(P,)]
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    # (A_out, Bu_out)
    return a_i*a_j, a_j*bu_i + bu_j


###################################################
# Method to compute the LxH output of a discretized 
# SSM given an LxH input sequence
####################################################
def apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym, bidirectional):
    # Discretized diagonal state matrix (P,P), input_sequence: (L, H)
    Lambda_elements = Lambda_bar * jnp.ones((input_sequence.shape[0], Lambda_bar.shape[0]))
    # B_bar: Discretized input matrix (P, H), C_tilde: Output matrix (H, P) or (2H, P)
    Bu_elements = jax.vmap(lambda u: B_bar@u)(input_sequence)
    _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))
    if bidirectional:
        _, xs2 = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements), reverse=True)
        xs = jnp.concatenate((xs, xs2), axis=1)
    if conj_sym:
        return jax.vmap(lambda x: 2*(C_tilde@x).real)(xs)
    else:
        # SSM output y: (L, H)
        return jax.vmap(lambda x: (C_tilde@x).real)(xs)


#######################
# S5 State Space Model
#######################
class S5(nn.Module):

    def __init__(
        self,
        Lambda_re_init: jnp.DeviceArray,
        Lambda_im_init: jnp.DeviceArray,
        V: jnp.DeviceArray,
        Vinv: jnp.DeviceArray,
        H: int,
        P: int,
        C_init: str,
        discretization: str,
        dt_min: float,
        dt_max: float,
        conj_sym: bool=True,
        clip_eigs: bool=False,
        bidirectional: bool=False,
        step_rescale: float=1.0
    ):
        '''
            [Args]
                Lambda_re_init (complex64): Real part of the initial diagonal state matrix (P,)
                Lambda_im_init (complex64): Imaginary part of the initial diagonal state matrix (P,)
                V (complex64): Eigenvectors used for initialization (P, P)
                Vinv (complex64): Inverse of eigenvectors used for initialization (P, P)
                H (int32): Number of input sequence features
                P (int32): State size
                C_init (string): specifies how C is initialized
                    1. trunc_standard_normal: sample from truncated standard normal and then multiply by V
                    2. lecun_normal: sample from lecun normal and then multiply by V
                    3. complex_normal: sample from complex valued output from standard normal, and don't multiply by V
                conj_sym (bool): whether to have conjugate symmetry
                clip_egs (bool): whether to clip eigenvalues (enforce left-half plane condition s.t. real part of eigenvalues are negative)
                bidirectional (bool): whether to use bidirectional; if True, use two C-matrices
                discretization (string): specifies the discretization method
                    1. zoh: zero-order hold method
                    2. bilinear: bilinear transformation
                dt_min (float32): minimum time step when initializing log_step
                dt_max (float32): maximum time step when initializing log_step
                step_rescale (float32): can uniformly rescale timescale parameter if desired
        '''
        super().__init__()
        self.Lambda_re_init = Lambda_re_init
        self.Lambda_im_init = Lambda_im_init
        self.V = V
        self.Vinv = V
        self.H = H
        self.P = P
        self.C_init = C_init
        self.discretization = discretization
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.conj_sym = conj_sym
        self.clip_eigs = clip_eigs
        self.bidirectional = bidirectional
        self.step_rescale = step_rescale

    def setup(self):
        ''' Initializes parameters once and performs discretization '''
        if self.conj_sym:
            # sample real B, C and multiply by half of V
            local_P = 2*self.P     
        else:
            local_P = self.P
        
        # A -- Initialize diagonal state to state matrix Lambda (eigenvalues)
        self.Lambda_re = self.param('Lambda_re', lambda rng, shape: self.Lambda_re_init, (None,))
        self.Lambda_im = self.param('Lambda_im', lambda rng, shape: self.Lambda_im_init, (None,))
        if self.clip_eigs:
            self.Lambda = jnp.clip(self.Lambda_re, None, -1e-4) + 1j*self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j*self.Lambda_im

        # B -- Initialize input to state matrix
        B_init = lecun_normal()
        B_shape = (local_P, self.H)
        self.B = self.param('B', lambda rng, shape: init_VinvB(B_init, rng, shape, self.Vinv), B_shape)
        B_tilde = self.B[..., 0] + 1j*self.B[..., 1]

        # C -- Initialize state to output matrix
        if self.C_init in ['trunc_standard_normal']:
            C_init = trunc_standard_normal
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ['lecun_normal']:
            C_init = lecun_normal()
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ['complex_normal']:
            C_init = normal(stddev=0.5**0.5)
        else:
            raise NotImplementedError(f'C_init method {self.C_init} not implemented')

        if self.C_init in ['complex_normal']:
            C = self.param('C', C_init, (self.H, self.P*2, 2)) if self.bidirectional else self.param('C', C_init, (self.H, self.P, 2))
            self.C_tilde = C[..., 0] + 1j*C[..., 1]
        else:
            if self.bidirectional:
                self.C1 = self.param('C1', lambda rng, shape: init_CV(C_init, rng, shape, self.V), C_shape)
                self.C2 = self.param('C2', lambda rng, shape: init_CV(C_init, rng, shape, self.V), C_shape)
                C1 = self.C1[..., 0] + 1j*self.C1[..., 1]
                C2 = self.C2[..., 0] + 1j*self.C2[..., 1]
                self.C_tilde = jnp.concatenate([C1, C2], axis=-1)
            else:
                self.C = self.param('C', lambda rng, shape: init_CV(C_init, rng, shape, self.V), C_shape)
                self.C_tilde = self.C[..., 0] + 1j*self.C[..., 1]


        # D -- Initialize feedthrough matrix
        self.D = self.param('D', normal(stddev=1.0), (self.H,))

        # Initialize learnable discretization timescale value
        self.log_step = self.param('log_step', init_log_steps, (self.P, self.dt_min, self.dt_max))
        step = self.step_rescale * jnp.exp(self.log_step[:, 0])

        # Discretization
        if self.discretization in ['zoh']:
            self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)
        elif self.discretization in ['bilinear']:
            self.Lambda_bar, self.B_bar = discretize_bilinear(self.Lambda, B_tilde, step)
        else:
            raise NotImplementedError(f'Discretization method {self.discretization} not implemented')
        
    def __call__(self, input_sequence):
        # Input sequence (L, H) -> Output sequence (L, H)
        ys = apply_ssm(self.Lambda_bar, self.B_bar, self.C_tilde, input_sequence, self.conj_sym, self.bidirectional)
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du
