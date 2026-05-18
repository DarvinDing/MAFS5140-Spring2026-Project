"""
bayesian.py - Dynamic signal weighting using rolling correlations
"""

import numpy as np
import pandas as pd


class BayesianSignalWeights:
    """
    Maintains a rolling buffer of signal-return correlations for each signal,
    and computes adaptive weights via softmax of the recent average correlation.
    """

    def __init__(self, signal_names, window=20, prior_correlations=None):
        """
        :param signal_names: list of strings, e.g. ['momentum', 'mean_rev', 'volume']
        :param window: number of past days to use for rolling average correlation
        :param prior_correlations: dict with initial correlation values (from training)
        """
        self.signal_names = signal_names
        self.window = window
        # Buffer for daily correlations (each element: dict of signal->correlation for that day)
        self.corr_buffer = []
        # Prior correlations (used before buffer is full)
        if prior_correlations is None:
            self.prior_corr = {name: 0.0 for name in signal_names}
        else:
            self.prior_corr = prior_correlations

    def update(self, daily_correlations):
        """
        Add a new day's correlations (between each signal's score and the next day's return).
        :param daily_correlations: dict {signal_name: correlation (float)}
        """
        self.corr_buffer.append(daily_correlations)
        if len(self.corr_buffer) > self.window:
            self.corr_buffer.pop(0)

    def get_weights(self):
        """
        Compute current signal weights based on recent average correlations.
        Returns dict {signal_name: weight} summing to 1.
        """
        if len(self.corr_buffer) == 0:
            # Use prior correlations, but ensure non-negative
            raw = {name: max(0.0, self.prior_corr.get(name, 0.0)) for name in self.signal_names}
        else:
            # Average correlations over buffer
            avg_corr = {name: 0.0 for name in self.signal_names}
            for day_corr in self.corr_buffer:
                for name in self.signal_names:
                    avg_corr[name] += day_corr.get(name, 0.0)
            for name in self.signal_names:
                avg_corr[name] /= len(self.corr_buffer)
            # Clip negative correlations to zero (don't want to bet against a signal)
            raw = {name: max(0.0, avg_corr[name]) for name in self.signal_names}

        # Softmax (or simple normalisation) to sum to 1
        total = sum(raw.values())
        if total < 1e-8:
            # Equal weights if all zero
            weights = {name: 1.0 / len(self.signal_names) for name in self.signal_names}
        else:
            weights = {name: val / total for name, val in raw.items()}
        return weights