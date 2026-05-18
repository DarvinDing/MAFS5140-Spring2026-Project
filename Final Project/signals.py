"""
signals.py - Compute momentum, mean-reversion, and volume-trend signals
"""

import pandas as pd
import numpy as np


def momentum_score(returns_history, window=78):
    """
    Momentum = total return over the last 'window' periods.
    :param returns_history: list of Series (each Series = 5-min returns for all assets)
    :param window: number of periods to sum (default 78 = 1 trading day)
    :return: Series of momentum scores (one per asset)
    """
    if len(returns_history) < window:
        return None
    # Take the last 'window' returns and sum
    recent_returns = pd.DataFrame(returns_history[-window:])
    momentum = recent_returns.sum(axis=0)
    return momentum


def mean_reversion_score(close_history, window=78):
    """
    Mean-reversion = -z-score of the latest close relative to the last 'window' closes.
    :param close_history: list of Series (each Series = close prices for all assets)
    :param window: number of periods to look back (default 78 = 1 day)
    :return: Series of mean-reversion scores (higher = more oversold)
    """
    if len(close_history) < window + 1:
        return None
    # Last window+1 closes (including current)
    closes = pd.DataFrame(close_history[-(window + 1):])
    latest = closes.iloc[-1]
    mean = closes.iloc[:-1].mean(axis=0)
    std = closes.iloc[:-1].std(axis=0) + 1e-8
    zscore = (latest - mean) / std
    # Negative zscore -> price below mean -> buy signal -> positive score
    return -zscore


def volume_trend_score(returns_history, volume_history, window=78, long_window=390, volume_profile=None):
    """
    Volume-confirmed trend = return over last 'window' periods multiplied by
    (average volume over window) / (average volume over long_window).
    If volume_profile is provided, normalise volume by the intraday profile.
    :param returns_history: list of Series (returns)
    :param volume_history: list of Series (volumes)
    :param window: short window (default 78 = 1 day)
    :param long_window: long window for baseline volume (default 390 = 5 days)
    :param volume_profile: Series of average volume per intraday slot (optional)
    :return: Series of volume trend scores
    """
    if len(returns_history) < window or len(volume_history) < long_window:
        return None

    # Latest window returns
    returns_window = pd.DataFrame(returns_history[-window:])
    total_return = returns_window.sum(axis=0)

    # Volume averages
    vol_window = pd.DataFrame(volume_history[-window:])
    vol_long = pd.DataFrame(volume_history[-long_window:])
    avg_vol_short = vol_window.mean(axis=0)
    avg_vol_long = vol_long.mean(axis=0) + 1e-8

    # If volume profile exists, normalise each bar's volume before averaging?
    # For simplicity, we skip profile here (can be added later).
    vol_ratio = avg_vol_short / avg_vol_long
    # Score = return * volume ratio (capped at 3 to avoid extreme)
    vol_ratio = vol_ratio.clip(upper=3.0)
    return total_return * vol_ratio


def normalise_signals(scores_dict, method='zscore'):
    """
    Normalise each signal's scores across assets to have mean 0 and std 1.
    :param scores_dict: dict with keys 'momentum', 'mean_rev', 'volume' -> Series
    :param method: 'zscore' only for now
    :return: dict with normalised Series
    """
    normed = {}
    for name, scores in scores_dict.items():
        if scores is None:
            normed[name] = None
        else:
            std = scores.std()
            if std < 1e-8:
                normed[name] = scores - scores.mean()
            else:
                normed[name] = (scores - scores.mean()) / std
    return normed