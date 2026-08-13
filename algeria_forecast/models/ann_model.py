"""ANN forecaster: grid search followed by multi-run training with mean±std reporting."""

import itertools
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from ..config import Config
from ..metrics import Metrics
from ..utils import set_global_seeds
from .dl_builders import build_ann, dl_callbacks


class ANNModel:
    """Feed-forward ANN with grid search and N_RUNS independent trainings."""

    def __init__(self, config: Config, metrics: Metrics):
        self.config = config
        self.metrics = metrics

    def _grid_search(self, X_tr, y_tr, sid, target):
        import tensorflow as tf
        cfg = self.config

        seen, pruned = set(), []
        grid_keys = list(cfg.dl_grid.keys())
        for combo in itertools.product(*cfg.dl_grid.values()):
            p = dict(zip(grid_keys, combo))
            key = (p["lstm_units"], p["dropout"], p["lr"])
            if key not in seen:
                seen.add(key)
                pruned.append(p)

        n_val = max(30, int(0.1 * len(X_tr)))
        X_val, y_val = X_tr[-n_val:], y_tr[-n_val:]
        X_trn, y_trn = X_tr[:-n_val], y_tr[:-n_val]

        print(f"      ANN: grid search over {len(pruned)} configs ...")
        gs_log = []
        best_rmse, best_params = np.inf, None

        set_global_seeds(cfg.run_seeds[0])
        for p in pruned:
            tf.keras.backend.clear_session()
            try:
                mdl = build_ann(X_tr.shape[1], p["lstm_units"], p["dropout"], p["lr"])
                mdl.fit(X_trn, y_trn, validation_data=(X_val, y_val),
                         epochs=cfg.epochs, batch_size=cfg.batch_size,
                         callbacks=dl_callbacks(cfg.patience), verbose=0)
                val_pred = mdl.predict(X_val, verbose=0).flatten()
                rmse_val = float(np.sqrt(mean_squared_error(y_val, val_pred)))
            except Exception:
                rmse_val = np.inf
            gs_log.append({"hidden_units": p["lstm_units"],
                            "dropout": p["dropout"], "lr": p["lr"],
                            "val_rmse": round(rmse_val, 6)
                            if np.isfinite(rmse_val) else 9999})
            if rmse_val < best_rmse:
                best_rmse, best_params = rmse_val, p

        pd.DataFrame(gs_log).to_csv(
            cfg.gs_dir / f"{target}_{sid}_ann_gridsearch.csv", index=False)
        print(f"      ANN: best={best_params}  val_RMSE={best_rmse:.4f}")
        return best_params

    def run(self, X_tr, y_tr, X_te, y_te, scaler, sid, target, s_safe):
        import tensorflow as tf
        cfg = self.config

        best_params = self._grid_search(X_tr, y_tr, sid, target)

        with open(cfg.hp_dir / f"{target}_{sid}_ann_best_hyperparams.json", "w") as f:
            json.dump({
                "model": "ANN", "station_id": sid, "target": target,
                "hidden_units": best_params["lstm_units"],
                "dropout": best_params["dropout"],
                "lr": best_params["lr"],
                "global_seed": cfg.global_seed,
                "run_seeds": cfg.run_seeds,
                "n_runs": cfg.n_runs,
            }, f, indent=2)

        print(f"      ANN: {cfg.n_runs} independent runs ...")
        run_preds, run_metrics_list, run_epochs = [], [], []
        run_train_times, run_infer_times = [], []
        true_out = None

        for run_idx, seed in enumerate(cfg.run_seeds):
            set_global_seeds(seed)
            tf.keras.backend.clear_session()
            model = build_ann(X_tr.shape[1], best_params["lstm_units"],
                                best_params["dropout"], best_params["lr"])
            t0 = time.perf_counter()
            hist = model.fit(X_tr, y_tr, validation_split=0.1,
                               epochs=cfg.epochs, batch_size=cfg.batch_size,
                               callbacks=dl_callbacks(cfg.patience), verbose=0)
            run_train_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            pred_sc = model.predict(X_te, verbose=0).flatten()
            run_infer_times.append(time.perf_counter() - t0)

            pred = scaler.inverse_transform(pred_sc.reshape(-1, 1)).flatten()
            true_out = scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()
            m_run = self.metrics.compute(true_out, pred)
            run_preds.append(pred)
            run_metrics_list.append(m_run)
            run_epochs.append(len(hist.history["loss"]))
            print(f"        Run {run_idx + 1}/{cfg.n_runs}  seed={seed}  "
                  f"RMSE={m_run['RMSE']:.4f}  MAE={m_run['MAE']:.4f}  "
                  f"R2={m_run['R2']:.4f}  epochs={run_epochs[-1]}")

            with open(cfg.history_dir /
                       f"{target}_{sid}_{s_safe}_ANN_run{run_idx + 1}_history.json",
                       "w") as f:
                json.dump({k: [float(v) for v in vals]
                            for k, vals in hist.history.items()}, f, indent=2)

        best_run_idx = int(np.argmin([r["RMSE"] for r in run_metrics_list]))
        pred_final = run_preds[best_run_idx]
        m_agg = self.metrics.aggregate_runs(run_metrics_list)
        m_agg["train_time_s"] = round(float(np.mean(run_train_times)), 4)
        m_agg["infer_time_s"] = round(float(np.mean(run_infer_times)), 4)

        run_summary = {
            "model": "ANN", "station_id": sid, "target": target,
            "run_rmses": [r["RMSE"] for r in run_metrics_list],
            "run_maes": [r["MAE"] for r in run_metrics_list],
            "run_r2s": [r["R2"] for r in run_metrics_list],
            "mean_rmse": m_agg["RMSE"], "std_rmse": m_agg["RMSE_std"],
            "mean_mae": m_agg["MAE"], "std_mae": m_agg["MAE_std"],
            "mean_r2": m_agg["R2"], "std_r2": m_agg["R2_std"],
            "best_run": best_run_idx + 1,
            "run_epochs": run_epochs,
            "run_seeds": cfg.run_seeds,
            "run_train_times_s": [round(t, 4) for t in run_train_times],
            "run_infer_times_s": [round(t, 4) for t in run_infer_times],
            "mean_train_time_s": m_agg["train_time_s"],
            "mean_infer_time_s": m_agg["infer_time_s"],
        }
        with open(cfg.hp_dir / f"{target}_{sid}_ann_run_summary.json", "w") as f:
            json.dump(run_summary, f, indent=2)

        print(f"      ANN: RMSE={m_agg['RMSE']:.4f}±{m_agg['RMSE_std']:.4f}  "
              f"MAE={m_agg['MAE']:.4f}±{m_agg['MAE_std']:.4f}  "
              f"R2={m_agg['R2']:.4f}±{m_agg['R2_std']:.4f}  "
              f"train={m_agg['train_time_s']:.3f}s  infer={m_agg['infer_time_s']:.3f}s")

        return m_agg, pred_final, true_out
