"""
strategy.py - Fully implemented final project strategy
Includes day-boundary detection, Bayesian signal weighting, regime switching,
Black-Litterman with posterior returns, CVaR, and volatility targeting.
"""

import pandas as pd
import numpy as np
import pickle
from signals import momentum_score, mean_reversion_score, volume_trend_score, normalise_signals
from bayesian import BayesianSignalWeights
from black_litterman import black_litterman, standardise_views
from risk import gmv_weights, long_only_projection, cvar_heuristic, volatility_targeting


class Strategy:
    def __init__(self, model_path="final_model.pkl", cash_fraction=0.0):
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Parameters
        self.rebalance_freq = self.model['rebalance_freq']  # 78
        self.return_window = self.model['return_window']  # 390
        self.cash_fraction = cash_fraction
        self.target_daily_vol = self.model['vol_target']['daily_target']
        self.signal_corr_window = self.model['signal_corr_window']  # 20

        # Regime thresholds
        self.vol_low_thresh = self.model['regime_classifier']['vol_low_thresh']
        self.vol_high_thresh = self.model['regime_classifier']['vol_high_thresh']

        # Signal priors
        signal_priors = {
            name: self.model['signal_params'][name]['prior_corr']
            for name in ['momentum', 'mean_rev', 'volume']
        }
        self.signal_weighter = BayesianSignalWeights(
            signal_names=['momentum', 'mean_rev', 'volume'],
            window=self.signal_corr_window,
            prior_correlations=signal_priors
        )

        # Pre-compute inverses for each regime
        self.inv_covs = []
        for regime in self.model['regime_models']:
            cov = regime['cov_matrix']
            reg_cov = cov + np.eye(cov.shape[0]) * 1e-6
            self.inv_covs.append(np.linalg.pinv(reg_cov))

        # Online buffers (5‑min granularity)
        self.return_history = []  # list of Series (5‑min returns)
        self.close_history = []  # list of Series (closes)
        self.volume_history = []  # list of Series (volumes)
        self.prev_close = None
        self.step_counter = 0
        self.current_weights = None
        self.current_regime = 0

        # Daily aggregated buffers (for risk and signal adaptation)
        self.daily_asset_returns = []  # list of arrays (daily returns per asset)
        self.daily_portfolio_returns = []  # list of floats
        self.daily_dates = []  # list of dates for debugging
        self.pending_signals = None  # store today's signals to correlate with tomorrow's return

        # For day-boundary detection
        self.last_date = None

        print("Fully integrated final project strategy initialised.")
        print(f"  Assets: {len(self.model['tickers'])}, Factors: {self.model['n_factors']}")
        print(f"  Regimes: {len(self.model['regime_models'])}, Cash fraction: {self.cash_fraction}")

    def _aggregate_daily_returns(self, returns_last_day):
        """
        Aggregate a list of 5‑min return Series (length 78) into a single daily return Series.
        Daily return = (1 + r1)*(1 + r2)*... - 1 (approximated by sum for small returns)
        """
        daily_ret = returns_last_day[0].copy() * 0.0
        for ret_series in returns_last_day:
            daily_ret += ret_series  # sum of logs approximation; fine for 5‑min returns
        return daily_ret

    def _update_daily_buffers(self, date, daily_asset_ret, portfolio_ret):
        """Append daily data and trim to reasonable length."""
        self.daily_asset_returns.append(daily_asset_ret.values)
        self.daily_portfolio_returns.append(portfolio_ret)
        self.daily_dates.append(date)
        # Keep last 200 days (more than enough for vol targeting and CVaR)
        max_len = 200
        if len(self.daily_asset_returns) > max_len:
            self.daily_asset_returns.pop(0)
            self.daily_portfolio_returns.pop(0)
            self.daily_dates.pop(0)

    def step(self, current_market_data: pd.DataFrame) -> pd.Series:
        """Called at each 5‑min bar. Returns target weights."""
        # --- 1. Extract data ---
        if "close" not in current_market_data.columns:
            raise ValueError("Market data must contain 'close' column.")
        current_close = current_market_data["close"]
        current_volume = current_market_data.get("volume", None)
        tickers = current_close.index

        # --- 2. Update rolling 5‑min histories ---
        self.close_history.append(current_close)
        if len(self.close_history) > 2 * self.return_window:
            self.close_history.pop(0)

        if self.prev_close is not None:
            ret = (current_close - self.prev_close) / self.prev_close
            ret = ret.fillna(0.0)
            self.return_history.append(ret)
            if len(self.return_history) > self.return_window + 1:
                self.return_history.pop(0)

        if current_volume is not None:
            self.volume_history.append(current_volume)
            if len(self.volume_history) > self.return_window + 1:
                self.volume_history.pop(0)

        self.prev_close = current_close
        self.step_counter += 1

        # --- 3. Daily aggregation using step counter (every 78 steps = 1 day) ---
        # At the first step of a new day (step_counter % 78 == 1), we finalise previous day's data
        if self.step_counter % self.rebalance_freq == 1 and len(self.return_history) >= 78:
            # Aggregate the last 78 returns into a daily return
            prev_day_returns = self.return_history[-78:]
            daily_ret_series = self._aggregate_daily_returns(prev_day_returns)
            daily_ret_aligned = daily_ret_series.reindex(self.model['tickers']).fillna(0.0).values
            self.daily_asset_returns.append(daily_ret_aligned)

            # Compute portfolio's daily return for the previous day
            if self.current_weights is not None:
                w_aligned = self.current_weights.reindex(self.model['tickers']).fillna(0.0).values
                daily_port_ret = np.dot(w_aligned, daily_ret_aligned)
                self.daily_portfolio_returns.append(daily_port_ret)

            # Update Bayesian signal weights using correlation with next day's return
            if hasattr(self, 'pending_signals') and self.pending_signals is not None:
                daily_corr = {}
                for name, scores in self.pending_signals.items():
                    if scores is not None:
                        scores_aligned = scores.reindex(self.model['tickers']).fillna(0.0).values
                        if np.std(scores_aligned) > 1e-6 and np.std(daily_ret_aligned) > 1e-6:
                            corr = np.corrcoef(scores_aligned, daily_ret_aligned)[0, 1]
                            if not np.isnan(corr):
                                daily_corr[name] = corr
                        else:
                            daily_corr[name] = 0.0
                if daily_corr:
                    self.signal_weighter.update(daily_corr)
                self.pending_signals = None

        # --- 4. Rebalance decision (first step of each day) ---
        if self.step_counter % self.rebalance_freq == 1 and len(self.return_history) >= self.return_window:
            self._rebalance(tickers)
            # Store today's signals for next day's correlation update
            if hasattr(self, 'today_signals'):
                self.pending_signals = self.today_signals

        # Return cached weights or zeros
        if self.current_weights is None:
            return pd.Series(0.0, index=tickers)
        else:
            return self.current_weights.reindex(tickers, fill_value=0.0)

    def _rebalance(self, tickers):
        """Rebalance portfolio using all statistical components."""
        # --- 1. Regime detection (using daily asset returns buffer) ---
        if len(self.daily_asset_returns) >= 20:
            # Use last 20 days' average volatility across assets
            recent_vol = np.mean(np.std(self.daily_asset_returns[-20:], axis=0))
        else:
            recent_vol = self.model['vol_target']['long_term_vol']

        if recent_vol < self.vol_low_thresh:
            regime_idx = 0
        elif recent_vol > self.vol_high_thresh:
            regime_idx = 1
        else:
            regime_idx = 0  # default low‑vol regime
        self.current_regime = regime_idx
        regime = self.model['regime_models'][regime_idx]
        inv_cov = self.inv_covs[regime_idx]

        # --- 2. Compute signals for the previous complete day (last 78 bars) ---
        # Use the most recent 78 returns, closes, and volumes (if available)
        returns_last_day = self.return_history[-78:]
        closes_last_day = self.close_history[-79:]  # need 79 for mean‑reversion (window=78)
        volumes_last_day = self.volume_history[-78:] if self.volume_history else None

        # Momentum
        mom = momentum_score(returns_last_day, window=78)
        # Mean‑reversion (using closes)
        mean_rev = mean_reversion_score(closes_last_day, window=78)
        # Volume trend (if volume exists)
        vol_signal = None
        if volumes_last_day is not None:
            vol_signal = volume_trend_score(
                returns_last_day, volumes_last_day, window=78,
                long_window=min(390, len(self.volume_history)),
                volume_profile=self.model.get('intraday_volume_profile')
            )

        signals = {'momentum': mom, 'mean_rev': mean_rev, 'volume': vol_signal}
        # Remove None signals
        signals = {k: v for k, v in signals.items() if v is not None}
        # Normalise across assets (z‑score)
        signals_norm = normalise_signals(signals, method='zscore')
        # Store today's signals for next day's Bayesian update
        self.today_signals = signals_norm

        # --- 3. Get Bayesian signal weights (adaptive) ---
        signal_weights = self.signal_weighter.get_weights()
        active_signals = [s for s in signal_weights if s in signals_norm]
        if not active_signals:
            composite_score = pd.Series(0.0, index=closes_last_day[-1].index)
        else:
            composite_score = sum(signal_weights[s] * signals_norm[s] for s in active_signals)

        # --- 4. Factor model prior expected returns ---
        # Build returns matrix from last return_window 5‑min returns
        ret_matrix = np.array([
            ret.reindex(self.model['tickers']).values
            for ret in self.return_history[-self.return_window:]
        ])
        # Standardise using regime's mean and std (from training)
        std_scaled = (ret_matrix - regime['train_mean']) / (regime['train_std'] + 1e-8)
        std_scaled = np.nan_to_num(std_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        # Factor returns for each period
        factor_returns_period = std_scaled @ regime['loadings']
        # Most recent factor returns (last row)
        current_factors = factor_returns_period[-1, :]
        # Prior = alpha + beta * current_factors
        prior_returns = regime['alphas'] + regime['betas'] @ current_factors

        # --- 5. Black‑Litterman blending ---
        # Views: composite signal scores (standardised)
        views = standardise_views(composite_score.reindex(self.model['tickers']).values, method='zscore')
        # View confidence (fixed; could be made adaptive)
        view_conf = 0.1
        posterior_returns = black_litterman(prior_returns, regime['cov_matrix'], views, view_conf, tau=0.05)

        # --- 6. Mean‑variance optimal portfolio (tangency) ---
        # w = inv(cov) * posterior_returns   (unconstrained)
        w_mv = inv_cov @ posterior_returns
        # Long‑only projection
        w_long = long_only_projection(w_mv)

        # --- 7. CVaR constraint (if enough daily history) ---
        if len(self.daily_asset_returns) >= 60:
            hist_asset_returns = np.array(self.daily_asset_returns[-60:])
            w_cvar = cvar_heuristic(w_long, hist_asset_returns, alpha=0.05, threshold=-0.02, max_iter=10)
        else:
            w_cvar = w_long

        # --- 8. Volatility targeting (if enough portfolio history) ---
        if len(self.daily_portfolio_returns) >= 20:
            w_scaled = volatility_targeting(
                w_cvar, self.daily_portfolio_returns,
                self.target_daily_vol, min_scale=0.5, max_scale=2.0, kde=True
            )
        else:
            w_scaled = w_cvar

        # --- 9. Apply cash fraction and ensure sum ≤ 1 ---
        w_final = w_scaled * (1.0 - self.cash_fraction)
        total = w_final.sum()
        if total > 1.0:
            w_final = w_final / total

        # Convert to pandas Series with original ticker order
        weights_series = pd.Series(w_final, index=self.model['tickers'])
        weights_series = weights_series.reindex(tickers, fill_value=0.0)
        self.current_weights = weights_series