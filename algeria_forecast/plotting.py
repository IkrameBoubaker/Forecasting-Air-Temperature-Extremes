"""All matplotlib plotting for the forecasting study."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config
from .utils import safe_name


class Plotter:
    """Generates and saves every diagnostic plot used by the pipeline."""

    def __init__(self, config: Config):
        self.config = config

    # ── Per-station forecast plot ───────────────────────────────────
    def forecast_all(self, dates, true, pred_cnnlstm, pred_lstm,
                      pred_ann, pred_svr, pred_gfs, name, zone, target, sid):
        cfg = self.config
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.plot(dates, true, color="black", lw=0.9, alpha=0.9, label="Observed")
        ax.plot(dates, pred_cnnlstm, color=cfg.zone_colors[zone],
                 lw=1.3, ls="--", label="CNN-LSTM")
        ax.plot(dates, pred_lstm, color="orange", lw=1.0, ls=":",
                 alpha=0.80, label="LSTM")
        ax.plot(dates, pred_ann, color="#009688", lw=1.0, ls="--",
                 alpha=0.75, label="ANN")
        ax.plot(dates, pred_svr, color="#795548", lw=1.0, ls=":",
                 alpha=0.75, label="SVR")
        if pred_gfs is not None and not np.all(np.isnan(pred_gfs)):
            ax.plot(dates, pred_gfs[:len(dates)], color="green", lw=1.0,
                     ls=":", alpha=0.70, label="GFS")
        ax.set_title(f"{name} — {target.upper()} — {zone}", fontsize=10)
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{target.upper()} (°C)")
        ax.legend(fontsize=7, ncol=6)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = cfg.out_dir / f"{target}_{sid}_{safe_name(name)}_forecast.png"
        plt.savefig(p, dpi=150)
        plt.close()

    # ── All-model / all-station RMSE heatmap ────────────────────────
    def heatmap_all(self, df: pd.DataFrame, target: str):
        cfg = self.config
        cols = [f"{m}_RMSE" for m in cfg.model_order if f"{m}_RMSE" in df.columns]
        labels = [c.replace("_RMSE", "") for c in cols]
        data = df.set_index("station")[cols].values.astype(float)

        fig, ax = plt.subplots(figsize=(max(12, 1.5 * len(cols)), 6))
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["station"].values, fontsize=9)
        vmax = np.nanmax(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if not np.isnan(v):
                    c = "white" if v > vmax * 0.65 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                             fontsize=7, color=c)
        plt.colorbar(im, ax=ax, label="RMSE (°C)")
        ax.set_title(f"{target.upper()} — RMSE heatmap  "
                      f"(lag={cfg.lag}, neural: mean of {cfg.n_runs} runs)",
                      fontsize=11)
        plt.tight_layout()
        p = cfg.out_dir / f"{target}_heatmap_all_models_lag{cfg.lag}.png"
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"  Plot saved: {p.name}")

    # ── Seasonal RMSE heatmap (CNN-LSTM) ─────────────────────────────
    def seasonal(self, df_sea: pd.DataFrame, target: str):
        cfg = self.config
        rmse_cols = [f"{s}_RMSE" for s in ["Winter", "Spring", "Summer", "Autumn"]
                      if f"{s}_RMSE" in df_sea.columns]
        if not rmse_cols or df_sea.empty:
            return
        sm = df_sea.set_index("station")[rmse_cols]
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(sm.values.astype(float), aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(rmse_cols)))
        ax.set_xticklabels([c.replace("_RMSE", "") for c in rmse_cols], fontsize=10)
        ax.set_yticks(range(len(sm)))
        ax.set_yticklabels(sm.index, fontsize=9)
        for i in range(sm.shape[0]):
            for j in range(sm.shape[1]):
                v = sm.values[i, j]
                if not np.isnan(float(v)):
                    ax.text(j, i, f"{float(v):.2f}", ha="center",
                             va="center", fontsize=8)
        plt.colorbar(im, ax=ax, label="RMSE (°C)")
        ax.set_title(f"{target.upper()} — Seasonal RMSE — CNN-LSTM", fontsize=11)
        plt.tight_layout()
        p = cfg.out_dir / f"{target}_seasonal_heatmap_cnnlstm_lag{cfg.lag}.png"
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"  Plot saved: {p.name}")

    # ── DM significance heatmap (single target) ─────────────────────
    def dm_heatmap(self, df_dm: pd.DataFrame, target: str):
        cfg = self.config
        if df_dm.empty:
            return
        sig_col = ("significant_global_fdr" if "significant_global_fdr" in df_dm.columns
                    else "significant_fdr")
        better_col = "better_global" if "better_global" in df_dm.columns else "better"

        comparisons = df_dm["comparison"].unique().tolist()
        stations = df_dm["station"].unique().tolist()
        mat = np.zeros((len(stations), len(comparisons)))
        for i, st in enumerate(stations):
            for j, comp in enumerate(comparisons):
                sub = df_dm[(df_dm["station"] == st) & (df_dm["comparison"] == comp)]
                if not sub.empty:
                    r = sub.iloc[0]
                    if r[sig_col]:
                        mat[i, j] = 1 if r[better_col] == "CNN-LSTM" else -1

        fig, ax = plt.subplots(figsize=(max(10, 1.8 * len(comparisons)), 5))
        im = ax.imshow(mat, aspect="auto", cmap=plt.cm.RdYlGn, vmin=-1, vmax=1)
        ax.set_xticks(range(len(comparisons)))
        ax.set_xticklabels([c.replace("CNN-LSTM vs ", "") for c in comparisons],
                             fontsize=9, rotation=30, ha="right")
        ax.set_yticks(range(len(stations)))
        ax.set_yticklabels(stations, fontsize=9)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                txt = ("✓ CNN" if v == 1 else ("✗ Base" if v == -1 else "—"))
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, ticks=[-1, 0, 1])
        ax.set_title(f"{target.upper()} — DM test (global BH-FDR corrected, α=0.05)",
                      fontsize=11)
        plt.tight_layout()
        p = cfg.out_dir / f"{target}_dm_heatmap_lag{cfg.lag}.png"
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"  Plot saved: {p.name}")

    # ── Training-loss curves (run 1 of each model) ──────────────────
    def training_histories(self, target: str):
        cfg = self.config
        json_files = sorted(cfg.history_dir.glob(f"{target}_*_run1_history.json"))
        if not json_files:
            return
        for jf in json_files:
            parts = jf.stem.split("_")
            sid = parts[1]
            info = cfg.stations.get(int(sid))
            s_name = info.name if info else sid
            model_tag = parts[-3]
            with open(jf) as f:
                hist = json.load(f)
            fig, ax = plt.subplots(figsize=(7, 4))
            epochs = range(1, len(hist["loss"]) + 1)
            ax.plot(epochs, hist["loss"], lw=1.5, label="Train loss")
            ax.plot(epochs, hist["val_loss"], lw=1.5, ls="--", label="Val loss")
            ax.set_title(f"{s_name} — {model_tag} run-1 "
                          f"({target.upper()}, lag={cfg.lag})", fontsize=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE Loss")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            p = cfg.out_dir / f"{target}_{sid}_{safe_name(s_name)}_{model_tag}_loss.png"
            plt.savefig(p, dpi=150)
            plt.close()

    # ── Tmax vs Tmin RMSE bar comparison ─────────────────────────────
    def compare_targets(self, df_tmax: pd.DataFrame, df_tmin: pd.DataFrame):
        cfg = self.config
        dl_models = ["ANN", "SVR", "LSTM", "CNN-LSTM"]
        colors = ["#009688", "#795548", "#FF9800", "#E57373"]
        light = ["#80CBC4", "#D7CCC8", "#FFCC80", "#EF9A9A"]
        stations = df_tmax["station"].values
        x = np.arange(len(stations))
        n_m = len(dl_models)
        w = 0.8 / (n_m * 2)

        fig, ax = plt.subplots(figsize=(16, 5))
        for i, (mdl, col_d, col_l) in enumerate(zip(dl_models, colors, light)):
            c = f"{mdl}_RMSE"
            if c not in df_tmax.columns:
                continue
            offset = (i * 2 - n_m + 0.5) * w
            ax.bar(x + offset - w / 2,
                    pd.to_numeric(df_tmax[c], errors="coerce").values,
                    w, label=f"{mdl} Tmax", color=col_d, alpha=0.88)
            ax.bar(x + offset + w / 2,
                    pd.to_numeric(df_tmin[c], errors="coerce").values,
                    w, label=f"{mdl} Tmin", color=col_l, alpha=0.88)

        zones = [info.zone for info in cfg.stations.values()]
        for i, zone in enumerate(zones):
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.05,
                        color=cfg.zone_colors.get(zone, "gray"), zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels(stations, rotation=25, fontsize=9)
        ax.set_ylabel("RMSE (°C)")
        ax.set_title(f"ANN / SVR / LSTM / CNN-LSTM — RMSE: Tmax vs Tmin  "
                      f"(mean of {cfg.n_runs} runs for neural models)", fontsize=11)
        ax.legend(ncol=4, fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        p = cfg.out_dir / f"tmax_vs_tmin_comparison_lag{cfg.lag}.png"
        plt.savefig(p, dpi=150)
        plt.close()
        print(f"  Plot saved: {p.name}")

    # ── Side-by-side DM heatmaps for Tmax and Tmin ──────────────────
    def compare_dm(self, df_dm_tmax: pd.DataFrame, df_dm_tmin: pd.DataFrame):
        cfg = self.config
        if df_dm_tmax.empty and df_dm_tmin.empty:
            return

        fig, axes = plt.subplots(1, 2, figsize=(22, 5))
        for ax, df_dm, ttl in [(axes[0], df_dm_tmax, "Tmax"),
                                 (axes[1], df_dm_tmin, "Tmin")]:
            if df_dm.empty:
                ax.axis("off")
                continue

            sig_col = ("significant_global_fdr"
                        if "significant_global_fdr" in df_dm.columns else "significant_fdr")
            better_col = "better_global" if "better_global" in df_dm.columns else "better"

            comparisons = df_dm["comparison"].unique().tolist()
            stations = df_dm["station"].unique().tolist()
            mat = np.zeros((len(stations), len(comparisons)))
            for i, st in enumerate(stations):
                for j, comp in enumerate(comparisons):
                    sub = df_dm[(df_dm["station"] == st) & (df_dm["comparison"] == comp)]
                    if not sub.empty:
                        r = sub.iloc[0]
                        if r[sig_col]:
                            mat[i, j] = 1 if r[better_col] == "CNN-LSTM" else -1

            im = ax.imshow(mat, aspect="auto", cmap=plt.cm.RdYlGn, vmin=-1, vmax=1)
            ax.set_xticks(range(len(comparisons)))
            ax.set_xticklabels([c.replace("CNN-LSTM vs ", "") for c in comparisons],
                                 fontsize=8, rotation=30, ha="right")
            ax.set_yticks(range(len(stations)))
            ax.set_yticklabels(stations, fontsize=8)
            for i2 in range(mat.shape[0]):
                for j2 in range(mat.shape[1]):
                    v = mat[i2, j2]
                    txt = ("✓" if v == 1 else ("✗" if v == -1 else "—"))
                    ax.text(j2, i2, txt, ha="center", va="center", fontsize=9)
            ax.set_title(f"{ttl} — DM test (global BH-FDR, p<0.05)", fontsize=11)
            plt.colorbar(im, ax=ax, ticks=[-1, 0, 1])

        plt.tight_layout()
        p = cfg.out_dir / f"tmax_tmin_dm_comparison_lag{cfg.lag}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved: {p.name}")
