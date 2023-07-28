"""
Viking: Variational Bayesian Variance Tracking

See: https://arxiv.org/abs/2104.10777


Bel parameters:
* mean_latent, cov_latent
* mean_a, cov_a -- For observed-state covariance matrix
* mean_b, cov_b -- For latent-state covariance matrix


Cfg parameters:
* Transition (K)
* Projection
* num_samples
* rho_a
* rho_b
"""

import pandas as pd
import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def phi(b):
    z = np.maximum(0, b)
    return np.log(z + 1) * (b > 0)


@jit(nopython=True, cache=True)
def dphi(x):
    z = np.maximum(0, x)
    return 1 / (z + 1) * (x > 0)


@jit(nopython=True, cache=True)
def ddphi(x):
    z = np.maximum(0, x)
    return - 1 / (z + 1) ** 2 * (x > 0)


@jit(nopython=True, cache=True)
def f(b):
    return np.diag(phi(b))


@jit(nopython=True, cache=True)
def _compute_dPsi_H_factor(bel_latent, bel_b_prev, bel_latent_prev, transition, f):
    """
    Proposition 5
    """
    pmean_latent, pcov_latent = bel_latent
    pmean_latent_prev, _ = bel_latent_prev
    pmean_b_prev, _ = bel_b_prev
    M, _ = transition.shape
    I = np.eye(M)

    pmean_diff = (pmean_latent - transition @ pmean_latent_prev)
    B = pcov_latent + pmean_diff @ pmean_diff.T
    C = transition @ pcov_latent @ transition.T + f(pmean_b_prev)

    D_ddphi = np.diag(ddphi(pmean_b_prev).ravel())

    C_inv = np.linalg.inv(C)
    dphi_b = dphi(pmean_b_prev)
    
    dPsi = np.diag(C_inv @ (I - B @ C_inv)).reshape(-1, 1) * dphi_b
    H = - (C_inv @ B @ C_inv @ D_ddphi) * I + 2 * (C_inv @ B @ C_inv) * C_inv * (dphi_b @ dphi_b.T)

    return dPsi, H
    

@jit(nopython=True, cache=True)
def sample_multivariate_normal(n_samples, mean, cov, seed):
    np.random.seed(seed)
    dim = len(mean)
    L = np.linalg.cholesky(cov)
    eps = np.random.randn(n_samples, dim)

    samps = eps @ L.T + mean.T
    return samps


@jit(nopython=True, cache=True)
def _estimate_inv_A_matrix(b_samples, bel_latent, transition, f):
    """
    Compute Eq (3) for the posterior mean and covariance of the
    latent state
    """
    _, pcov_latent = bel_latent
    M, _ = transition.shape

    A_est = np.zeros((M, M))
    n_samples = len(b_samples)
    for n in range(n_samples):
        b = b_samples[n]
        A = np.linalg.inv(transition @ pcov_latent @ transition.T + f(b))
        A_est = A_est + A

    A_est = A_est / len(b_samples)
    return np.linalg.inv(A_est)


@jit(nopython=True, cache=True)
def _latent_update_posterior_covariance(x, A_inv, bel_a):
    """
    Equation (4)
    """
    pmean_a, pcov_a = bel_a # Covariance system state
    numerator = A_inv @ x @ x.T @ A_inv
    denominator = x.T @ A_inv @ x + np.exp(pmean_a - pcov_a / 2)
    posterior_covariance = A_inv - numerator / denominator
    return posterior_covariance


@jit(nopython=True, cache=True)
def _latent_update_posterior_mean(y, x, cov_next, mean_prev, bel_a, transition):
    """
    Equation (5)

    Parameters
    ----------
    x: covariate at time t
    cov_next: np.array
        Posterior covariance obtained at time t
    mean_prev: np.array
        Posterior mean obtained at time t-1
    bel_a: tuple
        Posterior belief of a obtained at time t
    """
    pmean_a, pcov_a = bel_a
    innovation = y - x.T @ transition @ mean_prev

    mean_next = transition @ mean_prev + cov_next @ x @ innovation / np.exp(pmean_a - pcov_a / 2)
    return mean_next


@jit(nopython=True, cache=True)
def _cov_obs_update_bel(y, x, bel_latent, bel_a, bel_a_prev, rho_a):
    """
    Update bel_a
    Propositions 3 and 4
    --------------------
    Estimate the posterior mean and covariance of the observation covariance
    """
    pmean_a, _ = bel_a
    pmean_a_prev, pcov_a_prev = bel_a_prev
    pmean_latent, pcov_latent = bel_latent
    M_a = 3 * pcov_a_prev

    innovation = y - pmean_latent.T @ x
    inn_plus_mahal = innovation ** 2 + x.T @ pcov_latent @ x


    # Posterior covariance for a
    term1 = 1 / (pcov_a_prev + rho_a)
    pcov_a_next = 1 / (term1 + inn_plus_mahal * np.exp(-pmean_a) / 2)

    # Posterior mean for a
    term2 = inn_plus_mahal * np.exp(-pmean_a_prev + pcov_a_next / 2 + M_a) / 2
    term3 = inn_plus_mahal * np.exp(-pmean_a_prev + pcov_a_next / 2) - 1

    denominator = term1 + term2 / 2
    hat_a = pmean_a_prev + term3 / (2 * denominator)
    pmean_a_next = np.maximum(np.minimum(hat_a, pmean_a_prev + M_a), pmean_a_prev - M_a)

    bel_a_new = (pmean_a_next, pcov_a_next)
    return bel_a_new
    

@jit(nopython=True, cache=True)
def _cov_dyn_update_bel(bel_latent, bel_latent_prev, bel_b_prev, rho_b, transition):
    """
    Update bel_b
    Proposition 6
    -------------
    Estimate the posterior mean and covariance of the dynamics covariance
    """
    I = np.eye(transition.shape[0])
    dPsi, H = _compute_dPsi_H_factor(bel_latent, bel_b_prev, bel_latent_prev, transition, f)

    pmean_b_prev, pcov_b_prev = bel_b_prev

    pcov_b_next = np.linalg.inv(np.linalg.inv(pcov_b_prev + rho_b * I) + H / 2)
    pmean_b_next = pmean_b_prev - pcov_b_next @ dPsi / 2

    pmean_b_next = np.maximum(pmean_b_next, 0.0)
    bel_b_next = (pmean_b_next, pcov_b_next)
    return bel_b_next


@jit(nopython=True, cache=True)
def _latent_update_bel(x, y, bel, cfg, seed):
    """
    Theorem 2
    --------
    Estimate the posterior mean and covariance of the latent bel
    """
    bel_latent, bel_b, bel_a = bel
    transition, n_samples, _, _ = cfg

    pmean_b, pcov_b = bel_b # Covariance observations

    b_samples = sample_multivariate_normal(n_samples, pmean_b, pcov_b, seed)
    A_inv = _estimate_inv_A_matrix(b_samples, bel_latent, transition, f)

    bel_latent_mean, _ = bel_latent
    bel_latent_cov_new = _latent_update_posterior_covariance(x, A_inv, bel_a)
    bel_latent_mean_new = _latent_update_posterior_mean(y, x, bel_latent_cov_new, bel_latent_mean, bel_a, transition)

    bel_latent_new = (bel_latent_mean_new, bel_latent_cov_new)
    return bel_latent_new


@jit(nopython=True, cache=True)
def initialise_bel(bel_prev, rho_a, rho_b):
    bel_latent, bel_b, bel_a = bel_prev
    pmean_b, pcov_b = bel_b
    pmean_a, pcov_a = bel_a
    bel_b = pmean_b, pcov_b + rho_b
    bel_a = pmean_a, pcov_a + rho_a

    bel = bel_latent, bel_b, bel_a
    return bel


@jit(nopython=True, cache=True)
def _viking_step(x, y, bel_prev, cfg, seed, n_inner):
    """
    """
    x = np.atleast_2d(x).T
    bel_latent, bel_b, bel_a = bel_prev
    transition, _, rho_a, rho_b = cfg
    bel_latent_prev, bel_b_prev, bel_a_prev = bel_prev

    bel = initialise_bel(bel_prev, rho_a, rho_b)
    for n in range(n_inner):
        seed = seed + 1
        bel_latent = _latent_update_bel(x, y, bel, cfg, seed)
        # Update terms for sigma
        bel_a = _cov_obs_update_bel(y, x, bel_latent, bel_a, bel_a_prev, rho_a)
        # Update terms for Q
        bel_b = _cov_dyn_update_bel(bel_latent, bel_latent_prev, bel_b_prev, rho_b, transition)
        bel = bel_latent, bel_b, bel_a
    
    return bel

@jit(nopython=True, cache=True)
def run(y, x, bel, cfg, n_inner=2, seed=314):
    """
    bel_latent, bel_b, bel_a = bel
    transition, _, rho_a, rho_b = cfg
    """
    T = len(y)
    bel_latent_hist = []
    bel_b_hist = []
    bel_a_hist = []
    for t in range(T):
        seed = seed + 1
        bel = _viking_step(x[t], y[t], bel, cfg, seed, n_inner)
        bel_latent, bel_b, bel_a = bel
        bel_latent_hist.append(bel_latent)
        bel_b_hist.append(bel_b)
        bel_a_hist.append(bel_a)
    
    bel_hist = bel_latent_hist, bel_b_hist, bel_a_hist
    return bel_hist


@jit(nopython=True, cache=True)
def posterior_predictive_samples(x, bel_hist, n_samples=100, seed=314):
    T = len(x)
    bel_latent_hist, _, _ = bel_hist
    yhat_hist = []
    for t in range(T):
        np.random.seed(seed)
        pmean, pcov = bel_latent_hist[t]
        params_sample = sample_multivariate_normal(n_samples, pmean, pcov, seed)
        yhat = params_sample @ x[t]

        yhat_hist.append(yhat)
        seed = seed + 1
    return yhat_hist


@jit(nopython=True, cache=True)
def run_regression(
    targets: pd.DataFrame,
    covariates: pd.DataFrame,
    rho_a: float = np.exp(-9),
    rho_b: float = np.exp(-6),
    weights_mean_init: float = 1.0,
    weights_cov_init: float = 1.0,
    a_mean_init: float = 1.0,
    a_cov_init: float = 1.0,
    b_mean_init: float = 1.0,
    b_cov_init: float = 1.0,
    n_inner: int = 2,
    n_samples: int = 50,
    seed: int = 314
):
    M = covariates.shape[-1]
    transition = np.eye(M)
    cfg = transition, n_samples, rho_a, rho_b

    bel_latent = np.ones((M, 1)) * weights_mean_init, np.eye(M) * weights_cov_init
    bel_b = np.ones((M, 1)) * b_mean_init, np.eye(M) * b_cov_init
    bel_a = np.ones((1,1)) * a_mean_init, np.eye(1) * a_cov_init
    bel  = bel_latent, bel_b, bel_a
    
    bel_hist = run(targets, covariates, bel, cfg, n_inner=n_inner, seed=seed)
    return bel_hist
