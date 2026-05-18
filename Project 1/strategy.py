import pandas as pd
import numpy as np
import json


class Strategy:
    def __init__(self, params_path="mini1_params.json"):
        """
        Loads hyperparameters from the training script.
        If the JSON file is not found, uses default fallback values.
        """
        try:
            with open(params_path, "r") as f:
                params = json.load(f)
            self.lookback = params["lookback"]
            self.z_threshold = params["z_threshold"]
            self.top_k = params["top_k"]
            self.cash_fraction = params["cash_fraction"]
            print(f"Loaded Mini1 params: lookback={self.lookback}, z_threshold={self.z_threshold}, "
                  f"top_k={self.top_k}, cash_fraction={self.cash_fraction}")
        except FileNotFoundError:
            # Fallback defaults (reasonable values)
            self.lookback = 20
            self.z_threshold = 1.0
            self.top_k = 30
            self.cash_fraction = 0.1
            print("mini1_params.json not found. Using default parameters.")

        # State variables
        self.price_history = []

    def step(self, current_market_data: pd.DataFrame) -> pd.Series:
        """
        Called at every 5-min timestamp.
        Returns target portfolio weights.
        """
        if "close" not in current_market_data.columns:
            raise ValueError("Market data must contain a 'close' column.")

        current_close = current_market_data["close"]
        tickers = current_close.index

        # Update price history
        self.price_history.append(current_close)
        if len(self.price_history) > self.lookback + 1:
            self.price_history.pop(0)

        # Not enough data -> cash
        if len(self.price_history) < self.lookback + 1:
            return pd.Series(0.0, index=tickers)

        # Build returns DataFrame
        close_df = pd.DataFrame(self.price_history)
        returns_df = close_df.pct_change().dropna()  # shape (lookback, n_assets)

        # Compute t-statistic for each asset
        mean_ret = returns_df.mean(axis=0)
        std_ret = returns_df.std(axis=0, ddof=1)
        n = len(returns_df)
        # Avoid division by zero
        std_ret = std_ret.replace(0, np.nan)
        t_stat = mean_ret / (std_ret / np.sqrt(n))
        t_stat = t_stat.fillna(0.0)

        # Filter by threshold
        mask = t_stat > self.z_threshold
        selected_tickers = t_stat[mask].index

        if len(selected_tickers) == 0:
            return pd.Series(0.0, index=tickers)

        # Rank and select top K
        sorted_tickers = t_stat[selected_tickers].sort_values(ascending=False).index
        top_tickers = sorted_tickers[:self.top_k]

        # Inverse volatility weighting
        vol_selected = std_ret[top_tickers]
        inv_vol = 1.0 / (vol_selected + 1e-8)
        weights_selected = inv_vol / inv_vol.sum()
        weights_selected = weights_selected * (1.0 - self.cash_fraction)

        # Build full weight Series
        weights = pd.Series(0.0, index=tickers)
        weights[top_tickers] = weights_selected

        # Safety: ensure sum <= 1.0
        total = weights.sum()
        if total > 1.0:
            weights = weights / total

        return weights