"""
black_litterman.py - Black-Litterman posterior expected returns
"""

import numpy as np


def black_litterman(prior_returns, cov_matrix, views, view_confidences, tau=0.05):
    """
    Compute Black-Litterman posterior expected returns.

    Parameters:
    - prior_returns: np.ndarray of shape (n_assets,), prior expected returns (e.g., from factor model)
    - cov_matrix: np.ndarray (n_assets, n_assets), prior covariance matrix
    - views: np.ndarray (n_assets,), view for each asset (the composite signal score, standardised)
    - view_confidences: float or np.ndarray, diagonal of Omega (uncertainty). If scalar, same for all assets.
    - tau: scalar, scaling factor for covariance of prior (typically small, 0.025-0.05)

    Returns:
    - posterior_returns: np.ndarray (n_assets,)
    """
    n = len(prior_returns)
    # P = identity (views on each asset)
    P = np.eye(n)
    # Q = views (column vector)
    Q = views.reshape(-1, 1)
    # Omega diagonal matrix
    if np.isscalar(view_confidences):
        Omega = np.eye(n) * view_confidences
    else:
        Omega = np.diag(view_confidences)

    # Compute intermediate matrices
    tau_cov = tau * cov_matrix
    # Posterior covariance of expected returns (not needed for mean, but used in formula)
    # Posterior mean:
    # E[R] = Π + τ Σ P' (P τ Σ P' + Ω)^{-1} (Q - P Π)
    Π = prior_returns.reshape(-1, 1)
    # Compute P τ Σ P' = τ Σ (since P=I)
    A = tau_cov + Omega  # (n, n)
    # Solve A * X = (Q - Π) for X (more stable than inverting)
    try:
        X = np.linalg.solve(A, Q - Π)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        X = np.linalg.pinv(A) @ (Q - Π)
    posterior = Π + tau_cov @ X
    return posterior.flatten()


def standardise_views(views, method='zero_mean'):
    """
    Standardise the composite signal scores to have zero mean and unit variance (across assets).
    This makes the views comparable across different regimes.
    """
    if method == 'zero_mean':
        return views - np.mean(views)
    elif method == 'zscore':
        std = np.std(views)
        if std < 1e-8:
            return views - np.mean(views)
        return (views - np.mean(views)) / std
    else:
        return views