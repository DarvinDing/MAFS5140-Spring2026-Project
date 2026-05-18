import pandas as pd
import numpy as np
import pickle


class Strategy:
    def __init__(self, model_path="mini2_model.pkl", cash_fraction=0.1, return_window=390):
        """
        Load pre‑trained factor model and set strategy parameters.

        Parameters:
        - model_path: path to the pickle file from train_mini2.py
        - cash_fraction: fraction of capital held in cash (0.1 = 10% cash)
        - return_window: number of 5‑min periods to use for current factor estimation (default 390 = 5 days)
        """
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        self.loadings = model['loadings']  # (n_assets, n_factors)
        self.train_mean = model['train_mean']  # (n_assets,)
        self.train_std = model['train_std']  # (n_assets,)
        self.betas = model['betas']  # (n_assets, n_factors)
        self.alphas = model['alphas']  # (n_assets,)
        self.cov_matrix = model['cov_matrix']  # (n_assets, n_assets)
        self.tickers = model['tickers']  # list of asset names
        self.n_assets = len(self.tickers)

        self.cash_fraction = cash_fraction
        self.return_window = return_window
        self.rebalance_freq = 78  # rebalance daily (78 five‑min periods)

        # State variables
        self.return_history = []  # list of return Series (5‑min returns)
        self.prev_close = None  # previous close to compute returns
        self.step_counter = 0
        self.current_weights = None

        # Pre‑compute inverse covariance matrix (for speed)
        # Add small regularization to avoid singular matrix
        reg_cov = self.cov_matrix + np.eye(self.n_assets) * 1e-6
        self.inv_cov = np.linalg.pinv(reg_cov)

        print(f"Mini Project 2 strategy initialised.")
        print(f"  Assets: {self.n_assets}, Factors: {self.loadings.shape[1]}")
        print(f"  Cash fraction: {self.cash_fraction}, Rebalance every {self.rebalance_freq} steps")

    def step(self, current_market_data: pd.DataFrame) -> pd.Series:
        """
        Called at each 5‑min timestamp.
        Returns target portfolio weights.
        """
        if "close" not in current_market_data.columns:
            raise ValueError("Market data must contain a 'close' column.")

        current_close = current_market_data["close"]
        tickers = current_close.index

        # 1. Compute 5‑min return since last step
        if self.prev_close is not None:
            # Ensure same order of tickers (use .loc to align)
            prev_aligned = self.prev_close.reindex(tickers)
            period_return = (current_close - prev_aligned) / prev_aligned
            period_return = period_return.fillna(0.0)
            self.return_history.append(period_return)
            # Trim history to return_window + 1
            if len(self.return_history) > self.return_window + 1:
                self.return_history.pop(0)
        self.prev_close = current_close

        # 2. Check if it's time to rebalance
        self.step_counter += 1
        if self.step_counter % self.rebalance_freq != 0:
            if self.current_weights is None:
                return pd.Series(0.0, index=tickers)
            else:
                return self.current_weights.reindex(tickers, fill_value=0.0)

        # 3. Rebalancing: need at least return_window returns
        if len(self.return_history) < self.return_window:
            self.current_weights = pd.Series(0.0, index=tickers)
            return self.current_weights

        # Build returns matrix from the last `return_window` returns
        # Shape: (return_window, n_assets)
        returns_list = []
        for ret_series in self.return_history[-self.return_window:]:
            # Align to the training tickers order
            ret_aligned = ret_series.reindex(self.tickers, fill_value=0.0)
            returns_list.append(ret_aligned.values)
        returns_array = np.array(returns_list)

        # Standardise using training set's mean and std
        returns_std = (returns_array - self.train_mean) / self.train_std
        # Replace any inf/nan
        returns_std = np.nan_to_num(returns_std, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute current factor returns for each period
        # factor_returns_period = returns_std @ loadings (shape: window × n_factors)
        factor_returns_period = returns_std @ self.loadings

        # Use the most recent factor realisation (last row)
        current_factors = factor_returns_period[-1, :]  # shape (n_factors,)

        # Compute expected returns: alpha + beta @ current_factors
        expected_ret = self.alphas + self.betas @ current_factors  # shape (n_assets,)

        # 4. Global minimum variance portfolio
        ones = np.ones(self.n_assets)
        w_gmv = self.inv_cov @ ones
        w_gmv = w_gmv / (ones @ w_gmv + 1e-8)  # normalise, avoid division by zero

        # 5. Project to long‑only
        w_long = np.maximum(w_gmv, 0.0)
        w_sum = w_long.sum()
        if w_sum > 1e-8:
            w_long = w_long / w_sum
        else:
            w_long = np.zeros(self.n_assets)

        # 6. Apply cash buffer
        w_long = w_long * (1.0 - self.cash_fraction)

        # 7. Convert to pandas Series
        weights = pd.Series(w_long, index=self.tickers)
        weights = weights.reindex(tickers, fill_value=0.0)

        # 8. Final safety
        total = weights.sum()
        if total > 1.0:
            weights = weights / total

        self.current_weights = weights
        return weights