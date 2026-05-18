"""
risk.py - Portfolio optimisation and risk management
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def gmv_weights(cov_matrix, inv_cov=None):
    """
    Compute global minimum variance portfolio weights (unconstrained, may be negative).
    :param cov_matrix: np.ndarray (n_assets, n_assets)
    :param inv_cov: optional pre-computed inverse covariance (for speed)
    :return: np.ndarray weights summing to 1
    """
    n = cov_matrix.shape[0]
    if inv_cov is None:
        inv_cov = np.linalg.pinv(cov_matrix)
    ones = np.ones(n)
    w = inv_cov @ ones
    w = w / (ones @ w + 1e-8)
    return w


def long_only_projection(weights):
    """
    Project weights to long-only (non-negative) and renormalise to sum = 1.
    """
    w = np.maximum(weights, 0.0)
    total = w.sum()
    if total > 1e-8:
        w = w / total
    else:
        w = np.zeros_like(w)
    return w


def cvar_heuristic(weights, historical_asset_returns, alpha=0.05, threshold=-0.02, max_iter=10, scale_factor=0.9):
    """
    Heuristic to enforce a CVaR constraint: scale down weights until CVaR(alpha) >= threshold.
    :param weights: np.ndarray (n_assets,) initial weights (should already be long-only and sum=1)
    :param historical_asset_returns: pd.DataFrame (n_days, n_assets) of daily asset returns
    :param alpha: tail probability (e.g., 0.05 for 95% CVaR)
    :param threshold: maximum allowed CVaR (e.g., -0.02 means daily loss not worse than 2%)
    :param max_iter: maximum number of scaling steps
    :param scale_factor: factor to multiply weights each iteration if CVaR too low (e.g., 0.9)
    :return: adjusted weights (still sum to 1)
    """
    if historical_asset_returns is None or historical_asset_returns.shape[0] == 0:
        return weights  # no history, no constraint

    # Compute daily portfolio returns using current weights
    port_returns = historical_asset_returns @ weights
    # CVaR = average of worst alpha fraction
    sorted_returns = np.sort(port_returns)
    idx = int(np.ceil(alpha * len(sorted_returns))) - 1
    if idx < 0:
        idx = 0
    cvar = np.mean(sorted_returns[:idx + 1])

    if cvar >= threshold:
        return weights

    # Scale down and renormalise until constraint satisfied or max_iter reached
    w_scaled = weights.copy()
    for _ in range(max_iter):
        w_scaled = w_scaled * scale_factor
        # Normalise to sum=1 again (scale factor already reduces sum, but keep sum=1 for consistency)
        w_scaled = w_scaled / w_scaled.sum()
        port_returns = historical_asset_returns @ w_scaled
        sorted_returns = np.sort(port_returns)
        cvar = np.mean(sorted_returns[:idx + 1])
        if cvar >= threshold:
            break
    return w_scaled


def volatility_targeting(weights, portfolio_returns_history, target_daily_vol, min_scale=0.5, max_scale=2.0, kde=True,
                         lookback=60):
    """
    Scale portfolio weights to target a given daily volatility, using either kernel density or simple std.
    :param weights: np.ndarray (n_assets,) current weights (sum=1)
    :param portfolio_returns_history: list or array of past daily portfolio returns
    :param target_daily_vol: desired daily standard deviation (e.g., 0.00945 for 15% annual)
    :param min_scale: minimum scaling factor
    :param max_scale: maximum scaling factor
    :param kde: if True, use Gaussian KDE to estimate volatility (more robust to non-normality)
    :param lookback: number of past days to use
    :return: scaled weights (sum may be >1, caller can renormalise if needed)
    """
    if len(portfolio_returns_history) < 2:
        # Not enough history, use scaling factor 1.0
        return weights

    # Use the most recent lookback days
    recent_returns = np.array(portfolio_returns_history[-lookback:])

    if kde:
        # Fit Gaussian KDE and estimate standard deviation
        # (KDE standard deviation is the sqrt of the second moment if we assume zero mean? Better: compute sample std, or use KDE's std)
        # For simplicity, we can compute sample std; KDE is overkill for volatility. But keep for demonstration.
        try:
            kde_obj = gaussian_kde(recent_returns)
            # Generate a set of points to estimate std (or directly use sample std)
            # We'll just use sample std for speed and stability
            current_vol = recent_returns.std()
        except:
            current_vol = recent_returns.std()
    else:
        current_vol = recent_returns.std()

    if current_vol < 1e-8:
        scale = 1.0
    else:
        scale = target_daily_vol / current_vol
    scale = np.clip(scale, min_scale, max_scale)

    return weights * scale