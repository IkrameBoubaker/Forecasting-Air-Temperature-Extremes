"""
Feature engineering shared by the ANN, SVR, LSTM and CNN-LSTM models.

Feature set (7 features per sample):
    - 7 lagged temperature values (t-7 .. t-1)

This is a lag-only, purely autoregressive feature set: no seasonal
(month sin/cos) or volatility (rolling std) context is added. The 7
lags remain a genuine time sequence for LSTM/CNN-LSTM (fed as a
(lag, 1) tensor). For ANN and SVR, which have no temporal structure to
preserve, the same 7 lags are used as a flat feature vector.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Preprocessor:
    """Builds lag sequences and flat feature matrices."""

    def __init__(self):
        self.temp_scaler: MinMaxScaler | None = None

    @staticmethod
    def make_sequences(values: np.ndarray, lag: int):
        X, y = [], []
        for i in range(lag, len(values)):
            X.append(values[i - lag: i])
            y.append(values[i])
        return np.array(X), np.array(y)

    def prepare(self, train_s: pd.Series, test_s: pd.Series, lag: int):
        """
        Scale temperature on train only (no leakage), then build lag
        sequences across the train/test boundary (so the first test
        day already has a valid lag context — see module docstring).

        Parameters
        ----------
        train_s, test_s : pd.Series with a DatetimeIndex (observed
            temperature), as returned by DataLoader.split().
        lag : int, lookback window length.

        Returns
        -------
        scaler   : fitted MinMaxScaler for the temperature series
                   (needed to inverse-transform predictions back to °C)
        X_tr, y_tr : flat lag-only training matrix / target, (n, lag)
        X_te, y_te : flat lag-only test matrix / target,     (n, lag)
        X_tr_dl, X_te_dl : (n, lag, 1) lag-only sequence for LSTM/CNN-LSTM
        """
        self.temp_scaler = MinMaxScaler((0, 1))
        tr_sc = self.temp_scaler.fit_transform(
            train_s.values.reshape(-1, 1)).flatten()
        te_sc = self.temp_scaler.transform(
            test_s.values.reshape(-1, 1)).flatten()

        combined_vals = np.concatenate([tr_sc, te_sc])
        X_all, y_all = self.make_sequences(combined_vals, lag)

        n = len(tr_sc) - lag  # number of training sequences
        X_tr_lag, y_tr = X_all[:n], y_all[:n]
        X_te_lag, y_te = X_all[n:], y_all[n:]

        # ── Flat lag-only matrix for ANN / SVR ───────────────────────────
        X_tr = X_tr_lag
        X_te = X_te_lag

        # ── (lag, 1) sequence branch for LSTM / CNN-LSTM ────────────────
        X_tr_dl = X_tr_lag.reshape(-1, lag, 1)
        X_te_dl = X_te_lag.reshape(-1, lag, 1)

        return (self.temp_scaler, X_tr, y_tr, X_te, y_te,
                X_tr_dl, X_te_dl)
