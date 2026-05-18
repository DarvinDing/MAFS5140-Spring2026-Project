"""
train_final.py - Enhanced training for final project
Loads large 5-min parquet file, extracts features, trains regime-specific factor models,
signal thresholds, Bayesian priors, and saves everything to final_model.pkl
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

# -------------------------------
# 1. Load training data (5-min close + volume)
# -------------------------------
print("Loading training data (train_3.parquet)...")
try:
    df = pd.read_parquet("train_f.parquet", engine='pyarrow')
except:
    df = pd.read_parquet("train_f.parquet", engine='fastparquet')

# Remove timezone from index
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

# Extract close and volume
if isinstance(df.columns, pd.MultiIndex):
    close = df.xs('close', axis=1, level=1).astype(np.float32)
    volume = df.xs('volume', axis=1, level=1).astype(np.float32)
else:
    # Single level: assume close only, volume not available
    close = df.astype(np.float32)
    volume = None
    print("Warning: No volume data found. Volume signals will be disabled.")

n_assets = close.shape[1]
tickers = close.columns.tolist()
print(f"Data shape: {close.shape}, Assets: {n_assets}")

# -------------------------------
# 2. Compute intraday features (summary statistics only)
# -------------------------------
print("Computing intraday features...")
# 2.1 Volume profile (if volume exists)
intraday_volume_profile = None
if volume is not None:
    # Average volume per 5-min slot (e.g., 09:30, 09:35, ...)
    time_of_day = close.index.time
    volume_profile = volume.groupby(time_of_day).mean()
    intraday_volume_profile = volume_profile.mean(axis=1)  # Series over time slots
    print(f"  Volume profile computed for {len(intraday_volume_profile)} intraday slots.")

# 2.2 Summary stats for momentum and volatility thresholds (from intraday returns)
# We'll compute rolling stats on a sample to get distributions (not storing full)
returns_5min = close.pct_change().iloc[1:].fillna(0)
# Use a random sample of 10,000 rows to get percentiles (fast)
sample_returns = returns_5min.sample(n=min(10000, len(returns_5min)), random_state=42)
# Intraday momentum (5-bar return)
momentum_5 = sample_returns.rolling(5).sum().stack().dropna()
momentum_threshold = momentum_5.quantile(0.8)  # top 20% is "strong"
# Intraday volatility (20-bar std)
vol_20 = sample_returns.rolling(20).std().stack().dropna()
vol_threshold_high = vol_20.quantile(0.8)
vol_threshold_low = vol_20.quantile(0.2)
print(f"  Momentum threshold (80th percentile): {momentum_threshold:.6f}")
print(f"  Volatility thresholds: low={vol_threshold_low:.6f}, high={vol_threshold_high:.6f}")

# -------------------------------
# 3. Resample to daily data
# -------------------------------
print("Resampling to daily data...")
daily_close = close.resample('D').last().dropna()
daily_returns = daily_close.pct_change().dropna().astype(np.float32)
print(f"Daily returns shape: {daily_returns.shape}")

if volume is not None:
    daily_volume = volume.resample('D').sum().dropna()
    # Align dates with daily_returns
    common_dates = daily_returns.index.intersection(daily_volume.index)
    daily_returns = daily_returns.loc[common_dates]
    daily_volume = daily_volume.loc[common_dates]
    # Daily relative volume (volume / average volume for that asset)
    avg_vol = daily_volume.mean()
    daily_rel_volume = daily_volume / avg_vol
else:
    daily_rel_volume = None

# Also compute daily realised volatility (from intraday returns)
# For each day, aggregate squared 5-min returns
# daily realised volatility (from intraday returns)
daily_vol = returns_5min.groupby(returns_5min.index.date).apply(lambda x: np.sqrt((x ** 2).sum()))
daily_vol = daily_vol.astype(np.float32)
daily_vol.index = pd.to_datetime(daily_vol.index)
# Align with daily_returns
daily_vol = daily_vol.reindex(daily_returns.index).fillna(0)

# -------------------------------
# 4. Rolling factor models and regime clustering
# -------------------------------
print("Building rolling factor models...")
window_days = 120  # 6 months
step_days = 60
n_factors = 15  # fixed, can be tuned later

rolling_models = []  # store model parameters for each window
dates = daily_returns.index
n_days = len(dates)

for start_idx in range(0, n_days - window_days + 1, step_days):
    end_idx = start_idx + window_days
    window_returns = daily_returns.iloc[start_idx:end_idx]
    print(
        f"  Processing window {start_idx + 1}-{end_idx} ({window_returns.index[0].date()} to {window_returns.index[-1].date()})")

    # Standardise
    mean_win = window_returns.mean().values.copy()
    std_win = window_returns.std().values.copy()
    std_win[std_win < 1e-8] = 1.0
    returns_scaled = (window_returns.values - mean_win) / std_win

    # PCA
    pca = PCA(n_components=n_factors)
    factor_returns = pca.fit_transform(returns_scaled)  # (window_days, n_factors)
    loadings = pca.components_.T  # (n_assets, n_factors)

    # OLS per asset to get betas and alphas
    betas = np.zeros((n_assets, n_factors))
    alphas = np.zeros(n_assets)
    for i in range(n_assets):
        y = window_returns.values[:, i]
        X = factor_returns
        X_with_const = np.column_stack([np.ones(window_days), X])
        coeffs, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
        alphas[i] = coeffs[0]
        betas[i, :] = coeffs[1:]

    # Shrunk covariance
    lw = LedoitWolf()
    lw.fit(window_returns.values)
    cov_matrix = lw.covariance_

    # Mean factor returns over window
    mean_factor_ret = factor_returns.mean(axis=0)

    rolling_models.append({
        'start_date': window_returns.index[0],
        'end_date': window_returns.index[-1],
        'loadings': loadings,
        'mean': mean_win,
        'std': std_win,
        'betas': betas,
        'alphas': alphas,
        'cov_matrix': cov_matrix,
        'mean_factor_ret': mean_factor_ret,
        'explained_variance_ratio': pca.explained_variance_ratio_
    })

print(f"Created {len(rolling_models)} rolling windows.")

# 4.2 Cluster the windows into regimes
# Use the concatenated loadings (flattened) as features for clustering
print("Clustering windows into regimes...")
features = []
for m in rolling_models:
    # Flatten loadings (n_assets * n_factors) - too large. Instead use mean factor returns + explained variance?
    # Better: use the first 5 principal components of the loadings? Or just use the explained variance ratio.
    # Simpler: use the mean factor returns (n_factors) and the average volatility of the window.
    vol_avg = np.mean(np.diag(m['cov_matrix']))
    feat = np.concatenate([m['mean_factor_ret'], [vol_avg]])
    features.append(feat)
features = np.array(features)
# Standardise features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
# K-means with 2 clusters (high/low volatility regimes)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(features_scaled)
print(f"  Cluster sizes: {np.bincount(labels)}")

# For each cluster, compute the average model (or pick the model closest to centroid)
regime_models = []
for cluster_id in range(2):
    idxs = [i for i, lab in enumerate(labels) if lab == cluster_id]
    # Average all models in the cluster
    avg_loadings = np.mean([rolling_models[i]['loadings'] for i in idxs], axis=0)
    avg_mean = np.mean([rolling_models[i]['mean'] for i in idxs], axis=0)
    avg_std = np.mean([rolling_models[i]['std'] for i in idxs], axis=0)
    avg_betas = np.mean([rolling_models[i]['betas'] for i in idxs], axis=0)
    avg_alphas = np.mean([rolling_models[i]['alphas'] for i in idxs], axis=0)
    avg_cov = np.mean([rolling_models[i]['cov_matrix'] for i in idxs], axis=0)
    avg_factor_ret = np.mean([rolling_models[i]['mean_factor_ret'] for i in idxs], axis=0)
    regime_models.append({
        'loadings': avg_loadings,
        'train_mean': avg_mean,
        'train_std': avg_std,
        'betas': avg_betas,
        'alphas': avg_alphas,
        'cov_matrix': avg_cov,
        'mean_factor_ret': avg_factor_ret,
    })

# Classifier for online regime detection: use recent volatility
# Compute percentiles of daily volatility from training
vol_series = daily_vol.values.flatten()
vol_low_thresh = np.percentile(vol_series, 40)
vol_high_thresh = np.percentile(vol_series, 60)
regime_classifier = {
    'vol_low_thresh': vol_low_thresh,
    'vol_high_thresh': vol_high_thresh,
    'method': 'volatility_percentile'
}
print(f"  Volatility thresholds: low={vol_low_thresh:.6f}, high={vol_high_thresh:.6f}")

# -------------------------------
# 5. Train signal thresholds and Bayesian priors
# -------------------------------
print("Computing signal thresholds and priors...")
# We need to compute daily signal scores on the training set
# We'll do this on daily aggregates (fast) using 5-min data aggregated to daily.
# For each day, we need the last day's 5-min returns and volumes.
# Instead of re-iterating over all 5-min bars, we can compute directly from daily_returns and daily_rel_volume.

# Simulate daily signals (simplified but consistent with online computation)
# Momentum: daily return (already have)
momentum_scores = daily_returns  # DataFrame (days, assets)
# Mean-reversion: -z-score of daily close relative to last 5 days? Use daily returns as proxy?
# Better: use daily close vs 5-day moving average
ma5 = daily_close.rolling(5).mean()
mean_rev_scores = -(daily_close - ma5) / (daily_close.rolling(5).std() + 1e-8)
mean_rev_scores = mean_rev_scores.fillna(0)

# Volume trend: daily return * (daily_rel_volume)
if daily_rel_volume is not None:
    volume_scores = daily_returns * daily_rel_volume
else:
    volume_scores = pd.DataFrame(0.0, index=daily_returns.index, columns=daily_returns.columns)

# For each signal, compute correlation with next day's return
next_returns = daily_returns.shift(-1).dropna()
aligned_dates = daily_returns.index[:-1]
corr_momentum = []
corr_meanrev = []
corr_volume = []
for date in aligned_dates:
    # For each day, compute correlation across assets between signal and next day's return
    sig_m = momentum_scores.loc[date]
    sig_r = mean_rev_scores.loc[date]
    sig_v = volume_scores.loc[date]
    ret_next = next_returns.loc[date]
    # Only use finite values
    mask = sig_m.notna() & ret_next.notna()
    if mask.sum() > 10:
        corr_momentum.append(sig_m[mask].corr(ret_next[mask]))
        corr_meanrev.append(sig_r[mask].corr(ret_next[mask]))
        corr_volume.append(sig_v[mask].corr(ret_next[mask]))
    else:
        corr_momentum.append(0)
        corr_meanrev.append(0)
        corr_volume.append(0)
# Average correlations
avg_corr_momentum = np.nanmean(corr_momentum)
avg_corr_meanrev = np.nanmean(corr_meanrev)
avg_corr_volume = np.nanmean(corr_volume)
# Priors for Beta-Binomial: we assume a prior that the probability of correct prediction is around (corr+1)/2
# But we'll store correlations directly as prior weights.
signal_params = {
    'momentum': {'prior_corr': avg_corr_momentum, 'threshold': momentum_threshold},
    'mean_rev': {'prior_corr': avg_corr_meanrev, 'threshold': 0.0},  # threshold 0 means positive score -> buy
    'volume': {'prior_corr': avg_corr_volume, 'threshold': 0.0}
}
print(
    f"  Prior correlations: momentum={avg_corr_momentum:.4f}, mean-rev={avg_corr_meanrev:.4f}, volume={avg_corr_volume:.4f}")

# -------------------------------
# 6. Volatility targeting parameters
# -------------------------------
# Target annual volatility (typical for equity strategies)
target_vol_annual = 0.15
daily_target_vol = target_vol_annual / np.sqrt(252)
# Long-term average daily volatility from training
long_term_daily_vol = daily_returns.std().mean()
print(f"  Long-term daily volatility: {long_term_daily_vol:.6f}, target daily: {daily_target_vol:.6f}")

vol_target_params = {
    'target_vol_annual': target_vol_annual,
    'daily_target': daily_target_vol,
    'long_term_vol': long_term_daily_vol
}

# -------------------------------
# 7. Assemble final model dictionary
# -------------------------------
final_model = {
    'regime_models': regime_models,
    'regime_classifier': regime_classifier,
    'signal_params': signal_params,
    'vol_target': vol_target_params,
    'intraday_volume_profile': intraday_volume_profile,
    'tickers': tickers,
    'n_factors': n_factors,
    'rebalance_freq': 78,
    'return_window': 390,
    'cvar_alpha': 0.05,
    'cvar_threshold': -0.02,
    'volatility_window': 60,
    'signal_corr_window': 20,
}

# Save to pickle
with open('final_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

print("\nTraining completed successfully.")
print(f"Model saved to final_model.pkl (size: {len(pickle.dumps(final_model)) / 1024 / 1024:.2f} MB)")
print(f"  Number of assets: {n_assets}")
print(f"  Number of factors: {n_factors}")
print(f"  Number of regimes: {len(regime_models)}")