import jax
import jax.numpy as jnp



################################
# S4 SSM scan operation:
# <This is what's happening under the hood>
# 
# result = x[0]
# x_k, y_k = 0, 0
# for i in range(u):
#   x_k = A @ x[i] + B @ u[i]
#   y_k = C @ x[i]
#   result.append(x_k)
#   final = x_k
################################
def scan_SSM(Ab, Bb, Cb, u, x0):
  def step(x_k_1, u_k):
    x_k = Ab @ x_k_1 + Bb @ u_k
    y_k = Cb @ x_k
    return x_k, y_k
  return jax.lax.scan(step, x0, u)


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
# [S5] Method to compute the LxH output of a discretized 
# SSM given an LxH input sequence
####################################################
def apply_ssm(Lambda_bar, B_bar, C_tilde, input_sequence, conj_sym, bidirectional):
  # Discretized diagonal state matrix (P,), input_sequence: (L, H)
  Lambda_elements = Lambda_bar * jnp.ones((input_sequence.shape[0], Lambda_bar.shape[0]))
  # B_bar: Discretized input matrix (P, H), C_tilde: Output matrix (H, P) or (2H, P)
  Bu_elements = jax.vmap(lambda u: B_bar@u)(input_sequence)
  _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))
  if bidirectional:
    _, xs2 = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements), reverse=True)
    xs = jnp.concatenate((xs, xs2), axis=1)
  state = xs
  if conj_sym:
    y = jax.vmap(lambda x: 2*(C_tilde@x).real)(xs)
  else:
    # SSM output y: (L, H)
    y = jax.vmap(lambda x: (C_tilde@x).real)(xs)
  return y, state


def apply_ssm_state(state, input_sequence, Lambda_bar, B_bar, C_tilde, conj_sym, bidirectional):
  if state is None:
    Lambda_elements = Lambda_bar * jnp.ones((input_sequence.shape[0], Lambda_bar.shape[0]))
  else:
    ones_mask = jnp.ones((input_sequence.shape[0], Lambda_bar.shape[0]))
    # set first element to state
    initial = jnp.concatenate((state[None, :], ones_mask[:-1, :]), axis=0)
    Lambda_elements = Lambda_bar * initial
    # Lambda_elements = Lambda_bar * jnp.ones((input_sequence.shape[0], Lambda_bar.shape[0]))
  Bu_elements = jax.vmap(lambda u: B_bar@u)(input_sequence)
  _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))
  if bidirectional:
    _, xs2 = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements), reverse=True)
    xs = jnp.concatenate((xs, xs2), axis=1)
  state = xs
  if conj_sym:
    y = jax.vmap(lambda x: 2*(C_tilde@x).real)(xs)
  else:
    # SSM output y: (L, H)
    y = jax.vmap(lambda x: (C_tilde@x).real)(xs)
  return y, state
