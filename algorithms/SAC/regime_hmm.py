##############################################
### Gaussian HMM with diagonal covariance; ###
### EM fit; forward-filter probabilities   ###
##############################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans

@dataclass
class StandardScaler:
    """Per-feature standard scaler. Fit on train only."""
    mean_: np.ndarray
    std_: np.ndarray

    @staticmethod
    def fit(x: np.ndarray, eps: float = 1e-8) -> "StandardScaler":
        x = np.asarray(x, dtype=np.float64)
        mean = x.mean(axis=0)
        std = x.std(axis=0, ddof=0)
        std = np.where(std < eps, 1.0, std)
        return StandardScaler(mean_=mean, std_=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return (x - self.mean_) / self.std_


@dataclass
class GaussianHMMParams:
    """Diagonal Gaussian HMM params in *scaled* feature space."""
    pi: np.ndarray        # (K,)
    A: np.ndarray         # (K,K)
    means: np.ndarray     # (K,D)
    vars: np.ndarray      # (K,D)

    @property
    def n_states(self) -> int:
        return int(self.pi.shape[0])


def _log_gaussian_diag(x: np.ndarray, means: np.ndarray, vars_: np.ndarray) -> np.ndarray:
    """
    x: (T,D)
    means: (K,D)
    vars_: (K,D) diagonal variances
    returns logB: (T,K)
    """
    x = np.asarray(x, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    vars_ = np.asarray(vars_, dtype=np.float64)

    T, D = x.shape
    K = means.shape[0]

    log_det = np.sum(np.log(vars_), axis=1)          # (K,)
    inv_vars = 1.0 / vars_                           # (K,D)
    diff = x[:, None, :] - means[None, :, :]         # (T,K,D)
    quad = np.sum(diff * diff * inv_vars[None, :, :], axis=2)  # (T,K)

    return -0.5 * (D * np.log(2.0 * np.pi) + log_det[None, :] + quad)


def forward_filter(
    params: GaussianHMMParams,
    x: np.ndarray,
    init_pi: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Filtered probs p(z_t | x_1:t) computed in log-space."""
    x = np.asarray(x, dtype=np.float64)
    K = params.n_states

    pi = params.pi if init_pi is None else np.asarray(init_pi, dtype=np.float64)
    pi = np.clip(pi, 1e-12, 1.0)
    pi = pi / pi.sum()

    logA = np.log(np.clip(params.A, 1e-12, 1.0))
    logpi = np.log(pi)

    logB = _log_gaussian_diag(x, params.means, params.vars)  # (T,K)
    T = x.shape[0]

    log_alpha = np.full((T, K), -np.inf, dtype=np.float64)
    log_alpha[0] = logpi + logB[0]

    for t in range(1, T):
        prev = log_alpha[t - 1][:, None] + logA    # (K,K)
        log_alpha[t] = logB[t] + logsumexp(prev, axis=0)

    loglik = float(logsumexp(log_alpha[-1]))
    probs = np.exp(log_alpha - logsumexp(log_alpha, axis=1, keepdims=True))
    return probs, loglik


def _forward_backward(params: GaussianHMMParams, x: np.ndarray):
    """Forward-backward for EM (gamma, xi, loglik)."""
    x = np.asarray(x, dtype=np.float64)
    K = params.n_states
    T = x.shape[0]

    logA = np.log(np.clip(params.A, 1e-12, 1.0))
    logpi = np.log(np.clip(params.pi, 1e-12, 1.0))
    logB = _log_gaussian_diag(x, params.means, params.vars)

    log_alpha = np.full((T, K), -np.inf, dtype=np.float64)
    log_alpha[0] = logpi + logB[0]
    for t in range(1, T):
        prev = log_alpha[t - 1][:, None] + logA
        log_alpha[t] = logB[t] + logsumexp(prev, axis=0)

    log_beta = np.full((T, K), -np.inf, dtype=np.float64)
    log_beta[-1] = 0.0
    for t in range(T - 2, -1, -1):
        nxt = logA + (logB[t + 1] + log_beta[t + 1])[None, :]
        log_beta[t] = logsumexp(nxt, axis=1)

    loglik = float(logsumexp(log_alpha[-1]))

    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)

    xi = np.zeros((T - 1, K, K), dtype=np.float64)
    for t in range(T - 1):
        log_xi_t = (
            log_alpha[t][:, None] + logA + logB[t + 1][None, :] + log_beta[t + 1][None, :]
        )
        log_xi_t -= logsumexp(log_xi_t)
        xi[t] = np.exp(log_xi_t)

    return gamma, xi, loglik


def fit_gaussian_hmm_em(
    x_train: np.ndarray,
    n_states: int = 3,
    n_iter: int = 50,
    tol: float = 1e-4,
    min_var: float = 1e-4,
    seed: int = 42,
) -> Tuple[GaussianHMMParams, StandardScaler]:
    """
    Fit diagonal Gaussian HMM via EM.
    Returns params in *scaled* space + the scaler.
    """
    x_train = np.asarray(x_train, dtype=np.float64)
    if x_train.ndim != 2:
        raise ValueError(f"x_train must be (T,D), got {x_train.shape}")
    T, D = x_train.shape
    if T < 50:
        raise ValueError(f"Need >= ~50 points for stable HMM fit. got T={T}")

    scaler = StandardScaler.fit(x_train)
    x = scaler.transform(x_train)

    # KMeans init (deterministic via random_state)
    km = KMeans(n_clusters=n_states, n_init=10, random_state=seed)
    z0 = km.fit_predict(x)
    means = km.cluster_centers_.astype(np.float64)

    vars_ = np.zeros((n_states, D), dtype=np.float64)
    for k in range(n_states):
        xk = x[z0 == k]
        vars_[k] = (xk.var(axis=0) if xk.shape[0] >= 2 else x.var(axis=0)) + min_var

    # sticky transitions
    A = np.full((n_states, n_states), 1.0 / n_states, dtype=np.float64)
    np.fill_diagonal(A, 0.97)
    off = (1.0 - 0.97) / (n_states - 1)
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                A[i, j] = off

    pi = np.bincount(z0, minlength=n_states).astype(np.float64)
    pi = np.clip(pi, 1e-12, None)
    pi /= pi.sum()

    params = GaussianHMMParams(pi=pi, A=A, means=means, vars=vars_)

    prev_ll = -np.inf
    for _ in range(int(n_iter)):
        gamma, xi, ll = _forward_backward(params, x)

        pi_new = gamma[0].copy()

        A_new = xi.sum(axis=0)
        A_new = np.clip(A_new, 1e-12, None)
        A_new /= A_new.sum(axis=1, keepdims=True)

        means_new = (gamma.T @ x) / np.clip(gamma.sum(axis=0)[:, None], 1e-12, None)

        vars_new = np.zeros_like(vars_)
        for k in range(n_states):
            diff = x - means_new[k]
            w = gamma[:, k][:, None]
            vars_new[k] = (w * diff * diff).sum(axis=0) / np.clip(gamma[:, k].sum(), 1e-12, None)
        vars_new = np.clip(vars_new, min_var, None)

        params = GaussianHMMParams(pi=pi_new, A=A_new, means=means_new, vars=vars_new)

        if np.isfinite(prev_ll) and abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    return params, scaler


def reorder_states_by_feature(params: GaussianHMMParams, feature_index: int, ascending: bool = True):
    """Reorder states by means[:, feature_index]. Useful to map low-vol->stable."""
    m = params.means[:, int(feature_index)]
    perm = np.argsort(m) if ascending else np.argsort(-m)

    pi = params.pi[perm]
    A = params.A[perm][:, perm]
    means = params.means[perm]
    vars_ = params.vars[perm]
    return GaussianHMMParams(pi=pi, A=A, means=means, vars=vars_), perm
