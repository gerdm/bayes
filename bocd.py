# Based on
# https://github.com/gerdm/misc/blob/master/2022-03/bocpd.ipynb

import jax
import chex
import jax.numpy as jnp
from functools import partial
from jaxtyping import Float, Array, PyTree

@chex.dataclass
class BOCPDState:
    T: int # total number of time steps
    params_expfam: Float[PyTree, "T M"] # Params of the predictive distribution
    log_evidence: Float[Array, "T"]
    log_joint: Float[Array, "T"]
    time: int = 0


@chex.dataclass
class GaussParams:
    chi: float
    nu: float

    def __len__(self):
        return 2

    @property
    def mu(self):
        return self.chi / self.nu
    
    @property
    def sigma2(self):
        return 1 / self.nu

    @classmethod
    def from_org(cls, mean, variance):
        chi = mean / variance
        nu = 1 / variance
        return cls(chi=chi, nu=nu)

    @classmethod
    def from_array(cls, params):
        return cls(chi=params[0], nu=params[1])
    
    @property
    def array(self):
        return jnp.array([self.chi, self.nu])


class ExpfamGaussConj:
    """
    Exponential family Gaussian conjugate prior
    for the mean of a Gaussian likelihood
    with known variance.
    """
    def __init__(self, variance):
        self.variance = variance

    def suff_stat(self, x):
        Tx = jnp.array([x / self.variance])
        return Tx
    
    def natural_params(self, params):
        mu = params.mu
        return jnp.array([mu])
    
    def log_partition(self, params):
        A_eta = params.mu / (2 * self.variance)
        
        return A_eta
    
    def base_measure(self, x):
        hx = jnp.exp(-x**2 / (2 * self.variance))
        hx = hx / jnp.sqrt(2 * jnp.pi * self.variance)
        return hx
    
    def log_pdf(self, params, x):
        hx = self.base_measure(x)
        Tx = self.suff_stat(x)
        A_eta = self.log_partition(params)
        
        lpdf = jnp.log(hx) + Tx * params.chi + params.nu * A_eta
        return lpdf
    
    def pdf(self, params, x):
        return jnp.exp(self.log_pdf(params, x))

    def update_params(self, params, x):
        Tx = self.suff_stat(x)
        chi_new = params.chi + Tx
        nu_new = params.nu + 1

        params_new = GaussParams(chi=chi_new, nu=nu_new)
        return params_new


def init_bocd_state(num_timesteps, params_init):
    log_evidence = jnp.zeros(num_timesteps)
    log_joint = jnp.zeros(num_timesteps)
    M = len(params_init)
    
    params_init = params_init.array
    params = jnp.zeros((num_timesteps, M))
    params = params.at[0].set(params_init)

    return BOCPDState(
        T=num_timesteps,
        log_evidence=log_evidence,
        log_joint=log_joint,
        params_expfam=params,
    )


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
    cond1 = run_length == run_length_prev + 1
    cond2 = run_length == 0

    return cond1 * jnp.log(0.7) + cond2 * jnp.log(0.3)


def _step(bocs_state, x):
    ...