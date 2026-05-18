import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# 1. Load training data
# -------------------------------
print("Loading training data (train_2.parquet)...")
try:
    df = pd.read_parquet("train_2.parquet", engine='pyarrow')
except:
    df = pd.read_parquet("train_2.parquet", engine='fastparquet')

# Remove timezone from index if present
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

# Extract close prices (handle both MultiIndex and single-level)
if isinstance(df.columns, pd.MultiIndex):
    close_prices = df.xs('close', axis=1, level=1)
else:
    close_prices = df  # assume single level = close

print(f"Original shape: {close_prices.shape}")

# -------------------------------
# 2. Resample to daily closes
# -------------------------------
daily_close = close_prices.resample('D').last()
daily_close = daily_close.dropna(axis=0, how='any')
print(f"Daily closes shape: {daily_close.shape}")

# Compute daily returns
daily_returns = daily_close.pct_change().dropna()
print(f"Daily returns shape: {daily_returns.shape}")

# -------------------------------
# 3. Standardise daily returns for PCA
# -------------------------------
# Store mean and std for later use in inference
train_mean = daily_returns.mean(axis=0).values
train_std = daily_returns.std(axis=0).values
# Avoid division by zero
train_std = np.where(train_std < 1e-8, 1.0, train_std)

returns_scaled = (daily_returns.values - train_mean) / train_std

# -------------------------------
# 4. PCA - determine number of factors
# -------------------------------
# Cap at 20 factors to avoid overfitting (training set has ~2310 days)
max_factors = min(20, daily_returns.shape[1])
pca_full = PCA().fit(returns_scaled)
cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_factors = np.argmax(cumsum_var >= 0.80) + 1
n_factors = min(n_factors, max_factors)
print(f"Number of factors (80% variance, capped at {max_factors}): {n_factors}")

# Refit PCA with chosen n_factors
pca = PCA(n_components=n_factors)
returns_scaled_pca = pca.fit_transform(returns_scaled)  # shape (n_days, n_factors)
loadings = pca.components_.T  # shape (n_assets, n_factors)

# Save PCA parameters for later standardisation (using our own mean/std)
pca_components = pca.components_  # (n_factors, n_assets)

# Factor returns (already have returns_scaled_pca)
factor_returns = returns_scaled_pca  # shape (n_days, n_factors)

# -------------------------------
# 5. OLS per asset to estimate betas and alphas
# -------------------------------
n_assets = daily_returns.shape[1]
n_days = daily_returns.shape[0]
betas = np.zeros((n_assets, n_factors))
alphas = np.zeros(n_assets)

print("Running OLS for each asset...")
for i in range(n_assets):
    y = daily_returns.values[:, i]
    X = factor_returns
    # Add intercept column for OLS
    X_with_intercept = np.column_stack([np.ones(n_days), X])
    # Use lstsq for speed
    coeffs, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
    alphas[i] = coeffs[0]
    betas[i, :] = coeffs[1:]

# -------------------------------
# 6. Shrinkage covariance matrix of asset daily returns
# -------------------------------
print("Computing Ledoit-Wolf shrunk covariance matrix...")
lw = LedoitWolf()
lw.fit(daily_returns.values)
cov_matrix = lw.covariance_  # shape (n_assets, n_assets)

# -------------------------------
# 7. Mean factor returns over training period
# -------------------------------
mean_factor_ret = factor_returns.mean(axis=0)  # shape (n_factors,)

# -------------------------------
# 8. Get asset names
# -------------------------------
asset_names = daily_returns.columns.tolist()

# -------------------------------
# 9. Save all model components
# -------------------------------
model = {
    'loadings': loadings,                    # (n_assets, n_factors)
    'pca_components': pca_components,        # (n_factors, n_assets) - alternative
    'train_mean': train_mean,                # (n_assets,) for standardisation
    'train_std': train_std,                  # (n_assets,) for standardisation
    'betas': betas,                          # (n_assets, n_factors)
    'alphas': alphas,                        # (n_assets,)
    'cov_matrix': cov_matrix,                # (n_assets, n_assets)
    'mean_factor_ret': mean_factor_ret,      # (n_factors,)
    'tickers': asset_names,
    'n_assets': n_assets,
    'n_factors': n_factors
}

with open('mini2_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to mini2_model.pkl")
print(f"  Assets: {n_assets}, Factors: {n_factors}")
print("Training complete.")