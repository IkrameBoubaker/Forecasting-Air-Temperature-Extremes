"""Diebold-Mariano forecast comparison test with Benjamini-Hochberg FDR correction."""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


class StatisticalTests:
    """Diebold-Mariano test plus per-station and global BH-FDR correction."""

    @staticmethod
    def diebold_mariano(true, pred_a, pred_b, h: int = 1) -> dict:
        e_a = true - pred_a
        e_b = true - pred_b
        d = e_a ** 2 - e_b ** 2
        d_bar = np.mean(d)
        nw_var = np.var(d, ddof=1)
        for lag_k in range(1, h):
            gamma_k = np.mean((d[lag_k:] - d_bar) * (d[:-lag_k] - d_bar))
            nw_var += 2 * (1 - lag_k / h) * gamma_k
        nw_var = max(nw_var, 1e-12)
        dm_stat = d_bar / np.sqrt(nw_var / len(true))
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
        better = ("A" if dm_stat < 0 else "B") if p_value < 0.05 else "equal"
        return {"DM_stat": round(float(dm_stat), 4),
                "p_value": round(float(p_value), 4),
                "better": better}

    def run_dm_tests(self, true, predictions: dict,
                      cnn_key: str = "CNN-LSTM") -> pd.DataFrame:
        """DM test of CNN-LSTM vs every other model, with per-station BH-FDR."""
        if cnn_key not in predictions:
            return pd.DataFrame()
        pred_cnn = predictions[cnn_key]
        rows = []
        for name, pred in predictions.items():
            if name == cnn_key:
                continue
            if pred is None or np.all(np.isnan(pred)):
                continue
            n = min(len(true), len(pred_cnn), len(pred))
            dm = self.diebold_mariano(true[:n], pred_cnn[:n], pred[:n])
            rows.append({
                "comparison": f"{cnn_key} vs {name}",
                "DM_stat": dm["DM_stat"],
                "p_value": dm["p_value"],
                "better_raw": cnn_key if dm["better"] == "A"
                               else (name if dm["better"] == "B" else "equal"),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return df

        reject, p_adj, _, _ = multipletests(
            df["p_value"].values, alpha=0.05, method="fdr_bh")
        df["p_value_adj"] = np.round(p_adj, 6)
        df["significant_raw"] = df["p_value"] < 0.05
        df["significant_fdr"] = reject
        df["better"] = df.apply(
            lambda r: r["better_raw"] if r["significant_fdr"] else "equal", axis=1)
        return df

    @staticmethod
    def apply_global_fdr(df_dm_all: pd.DataFrame) -> pd.DataFrame:
        """Global BH-FDR correction across ALL station-variable DM tests."""
        if df_dm_all.empty:
            return df_dm_all
        reject, p_adj, _, _ = multipletests(
            df_dm_all["p_value"].values, alpha=0.05, method="fdr_bh")
        df_dm_all["p_value_global_adj"] = np.round(p_adj, 6)
        df_dm_all["significant_global_fdr"] = reject
        df_dm_all["better_global"] = df_dm_all.apply(
            lambda r: r["better_raw"] if r["significant_global_fdr"] else "equal",
            axis=1)
        return df_dm_all
