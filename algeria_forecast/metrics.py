"""Point-forecast metrics, climatology-based ACC, and RMSE skill score."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

_SEASON = {12: "Winter", 1: "Winter",  2: "Winter",
           3:  "Spring", 4: "Spring",  5: "Spring",
           6:  "Summer", 7: "Summer",  8: "Summer",
           9:  "Autumn", 10: "Autumn", 11: "Autumn"}


class Metrics:
    """RMSE / MAE / R2 plus ACC, RMSE skill score, and multi-run aggregation."""

    def __init__(self, acc_window: int = 15):
        self.acc_window = acc_window

    @staticmethod
    def compute(true: np.ndarray, pred: np.ndarray) -> dict:
        """
        RMSE, MAE, R2.

        MAPE is intentionally excluded: Algeria winter Tmin can be near
        0 °C, making percentage errors arbitrarily large and misleading.
        """
        return {
            "RMSE": round(float(np.sqrt(mean_squared_error(true, pred))), 4),
            "MAE":  round(float(mean_absolute_error(true, pred)), 4),
            "R2":   round(float(r2_score(true, pred)), 4),
        }

    @staticmethod
    def nan_metrics() -> dict:
        return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan}

    def build_doy_climatology(self, train_s: pd.Series) -> dict:
        """Smoothed day-of-year mean, used as the reference for ACC."""
        clim = {}
        window = self.acc_window
        for doy in range(1, 367):
            win = {(doy + d - 1) % 366 + 1 for d in range(-window, window + 1)}
            mask = train_s.index.dayofyear.isin(win)
            clim[doy] = float(train_s[mask].mean()
                               if mask.any() else train_s.mean())
        return clim

    @staticmethod
    def compute_acc(true: np.ndarray, pred: np.ndarray,
                     dates: pd.DatetimeIndex, clim: dict) -> float:
        """Anomaly Correlation Coefficient vs. climatology."""
        clim_vals = np.array([clim.get(int(d), np.nan) for d in dates.dayofyear])
        obs_anom = true - clim_vals
        fcst_anom = pred - clim_vals
        mask = ~(np.isnan(obs_anom) | np.isnan(fcst_anom))
        if mask.sum() < 3:
            return np.nan
        oa, fa = obs_anom[mask], fcst_anom[mask]
        if np.std(oa) < 1e-9 or np.std(fa) < 1e-9:
            return np.nan
        return round(float(np.corrcoef(fa, oa)[0, 1]), 4)

    @staticmethod
    def rmse_skill_score(rmse_model: float, rmse_ref: float) -> float:
        if np.isnan(rmse_ref) or rmse_ref < 1e-12:
            return np.nan
        return round(float((rmse_ref - rmse_model) / rmse_ref), 4)

    @staticmethod
    def print_metrics(label: str, m: dict, acc: float = None,
                       std_rmse: float = None) -> None:
        std_str = (f" ±{std_rmse:.4f}"
                   if (std_rmse is not None and not np.isnan(std_rmse)) else "")
        acc_str = (f"  ACC={acc:6.4f}"
                   if (acc is not None and not np.isnan(acc)) else "")
        print(f"    {label:<22s}  RMSE={m['RMSE']:7.4f}{std_str}  "
              f"MAE={m['MAE']:7.4f}  R2={m['R2']:7.4f}{acc_str}")

    @staticmethod
    def aggregate_runs(run_metrics: list) -> dict:
        """Mean ± std for RMSE, MAE, R2 across multiple neural-net runs."""
        keys = ["RMSE", "MAE", "R2"]
        agg = {}
        for k in keys:
            vals = [r[k] for r in run_metrics if not np.isnan(r[k])]
            agg[k] = round(float(np.mean(vals)), 4)
            agg[f"{k}_std"] = round(float(np.std(vals, ddof=1)), 6)
        return agg

    @staticmethod
    def seasonal_metrics(true, pred, dates) -> dict:
        """Per-season RMSE (Winter/Spring/Summer/Autumn)."""
        df = pd.DataFrame({"t": true, "p": pred}, index=dates)
        df["s"] = df.index.month.map(_SEASON)
        out = {}
        for s in ["Winter", "Spring", "Summer", "Autumn"]:
            sub = df[df["s"] == s]
            out[f"{s}_RMSE"] = (
                round(float(np.sqrt(mean_squared_error(sub["t"], sub["p"]))), 4)
                if len(sub) > 0 else np.nan)
        return out
