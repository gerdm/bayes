# Based on
# https://github.com/gerdm/misc/blob/master/2022-03/bocpd.ipynb

import jax
import chex
import jax.numpy as jnp
from functools import partial
from jaxtyping import Float, Array

@chex.dataclass
class BOCPDState:
    time: int
    param_eta: float # Count hyperparameter
    param_chi: float # Sufficient statistic hyperparameter
    log_evidence: Float[Array, "T"]
    log_joint: Float[Array, "T"]
    suff_stats: Float[Array, "T M"]
    total_time: int # Total number of time steps (For jax scan)


@partial(jax.jit, static_argnames=("T",))
def slice_tril_array(t, tril_array, T):
    """
    tril_array: represent a lower-triangular matrix in the form of an aray
    """
    # num_elements = T * (T + 1) // 2
    mask = jnp.arange(T)
    mask = mask <= t
    
    t = t.astype(int)
    s_init = t * (t + 1) // 2
    
    vslice = jnp.roll(tril_array, -s_init)
    vslice = jax.lax.slice_in_dim(vslice, 0, T)
    
    return vslice * mask


@jax.jit
def insert_in_tril_array(t, x, tril_array):
    start_from = t * (t + 1) // 2
    return jax.lax.dynamic_update_slice(tril_array, x, (start_from,))


@partial(jax.jit, static_argnames=("T",))
def reconstruct_tril_array(x, T):
    ixs = jnp.tril_indices(T)
    M = jnp.zeros((T, T))
    M = M.at[ixs].set(x)
    return M


def log_prob_transition(run_length, run_length_prev):
    ...


def _step(bocs_state, x):
    ...