import pandas as pd
import numpy as np
import json
from itertools import product

# -------------------------------
# 1. Load training data (large parquet)
# -------------------------------
print("Loading training data (this may take a moment)...")
# Use engine='pyarrow' with timezone fix (same as in data_feed.py)
try:
    df_train = pd.read_parquet("train_1.parquet", engine='pyarrow')
except:
    df_train = pd.read_parquet("train_1.parquet", engine='fastparquet')

# Remove timezone from index if present
if df_train.index.tz is not None:
    df_train.index = df_train.index.tz_localize(None)

# Ensure we have MultiIndex columns with 'close'
if isinstance(df_train.columns, pd.MultiIndex):
    close_data = df_train.xs('close', axis=1, level=1)
else:
    close_data = df_train  # assume single level = close prices

print(f"Training data shape: {close_data.shape}")  # (n_steps, n_assets)

# -------------------------------
# 2. Use a subset for faster tuning (first 20,000 rows ~ 25 trading days)
# -------------------------------
subset = close_data.iloc[:20000]  # adjust if you want more/less
print(f"Using subset of {len(subset)} rows for tuning.")

# -------------------------------
# 3. Define hyperparameter grid
# -------------------------------
lookbacks = [10, 20, 40]
z_thresholds = [0.5, 1.0, 1.5]
top_ks = [20, 30, 50]
cash_fractions = [0.0, 0.1, 0.2]

param_grid = list(product(lookbacks, z_thresholds, top_ks, cash_fractions))


# -------------------------------
# 4. Function to evaluate a single parameter set on the subset
# -------------------------------
def evaluate_params(lookback, z_thresh, top_k, cash_frac, data):
    """
    Simulates the strategy on the given data (prices) and returns the Sharpe ratio.
    Simplified version without full BacktestEngine for speed.
    """
    n_steps, n_assets = data.shape
    weights_history = []
    price_history = []

    for i in range(n_steps):
        current_prices = data.iloc[i]

        # Update price history
        price_history.append(current_prices.values)
        if len(price_history) > lookback + 1:
            price_history.pop(0)

        # Not enough data -> cash
        if len(price_history) < lookback + 1:
            weights = np.zeros(n_assets)
            weights_history.append(weights)
            continue

        # Build returns matrix (lookback periods)
        close_array = np.array(price_history)  # (T, n_assets)
        returns = np.diff(close_array, axis=0) / close_array[:-1]  # (T-1, n_assets)

        # Compute t-statistic for each asset
        mean_ret = returns.mean(axis=0)
        std_ret = returns.std(axis=0, ddof=1)
        n = returns.shape[0]
        with np.errstate(divide='ignore', invalid='ignore'):
            t_stat = mean_ret / (std_ret / np.sqrt(n))
            t_stat = np.nan_to_num(t_stat)  # replace NaN with 0

        # Filter assets
        mask = t_stat > z_thresh
        selected_idx = np.where(mask)[0]

        if len(selected_idx) == 0:
            weights = np.zeros(n_assets)
            weights_history.append(weights)
            continue

        # Rank by t-stat, take top_k
        sorted_idx = selected_idx[np.argsort(t_stat[selected_idx])[::-1]]
        top_idx = sorted_idx[:top_k]

        # Inverse volatility weighting
        vol = std_ret[top_idx]
        inv_vol = 1.0 / (vol + 1e-8)
        w_selected = inv_vol / inv_vol.sum()
        w_selected = w_selected * (1.0 - cash_frac)

        weights = np.zeros(n_assets)
        weights[top_idx] = w_selected
        weights_history.append(weights)

    # Compute portfolio returns
    portfolio_returns = []
    for t in range(1, len(weights_history)):
        # Asset returns from t-1 to t
        ret_t = (data.iloc[t].values - data.iloc[t - 1].values) / data.iloc[t - 1].values
        port_ret = np.dot(weights_history[t - 1], ret_t)
        portfolio_returns.append(port_ret)

    if len(portfolio_returns) == 0:
        return -np.inf

    # Sharpe ratio (annualized, but for ranking we can use raw Sharpe)
    rets = np.array(portfolio_returns)
    sharpe = rets.mean() / (rets.std() + 1e-8)
    return sharpe


# -------------------------------
# 5. Grid search
# -------------------------------
best_sharpe = -np.inf
best_params = None

print(f"Testing {len(param_grid)} combinations...")
for idx, (lb, zt, tk, cf) in enumerate(param_grid):
    sharpe = evaluate_params(lb, zt, tk, cf, subset)
    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_params = (lb, zt, tk, cf)
    if (idx + 1) % 20 == 0:
        print(f"  Progress: {idx + 1}/{len(param_grid)}")

print("\nBest parameters found:")
print(f"  lookback = {best_params[0]}")
print(f"  z_threshold = {best_params[1]}")
print(f"  top_k = {best_params[2]}")
print(f"  cash_fraction = {best_params[3]}")
print(f"  Sharpe (on subset) = {best_sharpe:.4f}")

# -------------------------------
# 6. Save parameters to JSON
# -------------------------------
params_dict = {
    "lookback": best_params[0],
    "z_threshold": best_params[1],
    "top_k": best_params[2],
    "cash_fraction": best_params[3]
}
with open("mini1_params.json", "w") as f:
    json.dump(params_dict, f, indent=4)

print("Parameters saved to mini1_params.json")