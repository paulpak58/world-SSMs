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
def _make_HiPPO(N):
    # N is the state size
    P = jnp.sqrt(1 + 2*jnp.arange(N))
    A = P[:, jnp.newaxis]*P[jnp.newaxis, :]
    A = jnp.tril(A) - jnp.diag(jnp.arange(N))
    # (N, N) HiPPO LegS matrix
    return -A


############################################
# Creates NPLR Representation of HiPPO-LegS matrix
############################################
def _make_NPLR_HiPPO(N):
    # N is the state size
    hippo = _make_HiPPO(N)
    P = jnp.sqrt(jnp.arange(N) + 0.5)
    B = jnp.sqrt(2*jnp.arange(N) + 1.0)
    # (N, N) HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B
    return hippo, P, B


############################################
# Creates DPLR Representation of HiPPO-LegS matrix
############################################
def make_DPLR_HiPPO(N):
    # N is the state size
    A, P, B = _make_NPLR_HiPPO(N)
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