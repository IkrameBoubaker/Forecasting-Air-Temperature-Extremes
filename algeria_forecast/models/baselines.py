"""Fixed, non-learned baseline forecasters: Persistence, Climatology, GFS passthrough."""

import time

import numpy as np
import pandas as pd

from ..metrics import Metrics


class PersistenceModel:
    """Naive forecast: tomorrow's value = today's observed value."""

    def __init__(self, metrics: Metrics):
        self.metrics = metrics

    def run(self, train_s: pd.Series, test_s: pd.Series):
        t0 = time.perf_counter()
        all_v = pd.concat([train_s, test_s])
        pred = all_v.shift(1).loc[test_s.index].values
        mask = ~np.isnan(pred)
        m = self.metrics.compute(test_s.values[mask], pred[mask])
        return m, pred, 0.0, time.perf_counter() - t0


class ClimatologyModel:
    """Forecast = smoothed day-of-year mean computed from the training period."""

    def __init__(self, metrics: Metrics, window: int = 7):
        self.metrics = metrics
        self.window = window

    def run(self, train_s: pd.Series, test_s: pd.Series):
        t0 = time.perf_counter()
        doy_mean = {}
        for doy in range(1, 367):
            win = {(doy + d - 1) % 366 + 1
                    for d in range(-self.window, self.window + 1)}
            mask = train_s.index.dayofyear.isin(win)
            doy_mean[doy] = (train_s[mask].mean()
                              if mask.any() else train_s.mean())
        t_train = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = np.array([doy_mean.get(d, train_s.mean())
                          for d in test_s.index.dayofyear])
        m = self.metrics.compute(test_s.values, pred)
        return m, pred, t_train, time.perf_counter() - t0


class GFSModel:
    """Wraps an externally produced NWP (GFS) forecast series for comparison."""

    def __init__(self, metrics: Metrics):
        self.metrics = metrics

    def run(self, test_s: pd.Series, gfs_series):
        if gfs_series is None:
            return (self.metrics.nan_metrics(),
                    np.full(len(test_s), np.nan), 0.0, 0.0)

        t0 = time.perf_counter()
        aligned = gfs_series.reindex(test_s.index)  # no fill — NaNs stay visible
        pred = aligned.values
        mask = ~np.isnan(pred)
        if mask.sum() < len(pred):
            print(f"      GFS: {(~mask).sum()} missing day(s) in test period "
                  f"— excluded from metrics.")
        m = self.metrics.compute(test_s.values[mask], pred[mask])
        return m, pred, 0.0, time.perf_counter() - t0
