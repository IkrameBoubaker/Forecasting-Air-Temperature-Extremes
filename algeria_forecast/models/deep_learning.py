"""LSTM and CNN-LSTM forecasters: shared grid search + multi-run training logic.

Both architectures take lag-only (lag, 1) sequence input, fed through
their recurrent/convolutional layers directly to the output layer
(see dl_builders.build_lstm/build_cnn_lstm).
"""

import itertools
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from ..config import Config
from ..metrics import Metrics
from ..utils import set_global_seeds
from .dl_builders import build_lstm, build_cnn_lstm, dl_callbacks


class DeepLearningModel:
    """
    Shared implementation for LSTM and CNN-LSTM: grid search over
    architecture/hyperparameters, followed by N_RUNS independent
    trainings whose metrics are aggregated as mean ± std.

    Parameters
    ----------
    model_type : "lstm" or "cnn_lstm"
    """

    def __init__(self, model_type: str, config: Config, metrics: Metrics):
        if model_type not in ("lstm", "cnn_lstm"):
            raise ValueError("model_type must be 'lstm' or 'cnn_lstm'")
        self.model_type = model_type
        self.config = config
        self.metrics = metrics

    def _build(self, lag, params):
        if self.model_type == "lstm":
            return build_lstm(lag, params["lstm_units"],
                                params["dropout"], params["lr"])
        return build_cnn_lstm(lag, params["filters"], params["lstm_units"],
                                params["kernel_size"], params["dropout"],
                                params["lr"])

    def _grid_search(self, X_tr_dl, y_tr, sid, target):
        import tensorflow as tf
        cfg = self.config

        n_val = max(30, int(0.1 * len(X_tr_dl)))
        X_val_dl, y_val = X_tr_dl[-n_val:], y_tr[-n_val:]
        X_trn_dl, y_trn = X_tr_dl[:-n_val], y_tr[:-n_val]

        grid_keys = list(cfg.dl_grid.keys())
        all_combos = list(itertools.product(*cfg.dl_grid.values()))

        # LSTM does not use filters / kernel_size → deduplicate
        if self.model_type == "lstm":
            seen, pruned = set(), []
            for c in all_combos:
                p = dict(zip(grid_keys, c))
                key = (p["lstm_units"], p["dropout"], p["lr"])
                if key not in seen:
                    seen.add(key)
                    pruned.append(c)
            all_combos = pruned

        print(f"      {self.model_type.upper()} grid search: "
              f"{len(all_combos)} configs ...")
        gs_log = []
        best_rmse, best_params = np.inf, None

        set_global_seeds(cfg.run_seeds[0])
        for combo in all_combos:
            p = dict(zip(grid_keys, combo))
            tf.keras.backend.clear_session()
            try:
                mdl = self._build(cfg.lag, p)
                mdl.fit(X_trn_dl, y_trn,
                         validation_data=(X_val_dl, y_val),
                         epochs=cfg.epochs, batch_size=cfg.batch_size,
                         callbacks=dl_callbacks(cfg.patience), verbose=0)
                val_pred = mdl.predict(X_val_dl, verbose=0).flatten()
                rmse_val = float(np.sqrt(mean_squared_error(y_val, val_pred)))
            except Exception as e:
                rmse_val = np.inf
                print(f"        skip {p}: {e}")

            gs_log.append({**p, "val_rmse": round(rmse_val, 6)
                            if np.isfinite(rmse_val) else 9999})
            if rmse_val < best_rmse:
                best_rmse, best_params = rmse_val, p

        pd.DataFrame(gs_log).to_csv(
            cfg.gs_dir / f"{target}_{sid}_{self.model_type}_gridsearch.csv",
            index=False)
        print(f"      {self.model_type.upper()} best: {best_params}  "
              f"val_RMSE={best_rmse:.4f}")
        return best_params

    def run(self, X_tr_dl, y_tr, X_te_dl, y_te,
            scaler, sid, target, s_safe):
        import tensorflow as tf
        cfg = self.config

        best_p = self._grid_search(X_tr_dl, y_tr, sid, target)

        with open(cfg.hp_dir /
                   f"{target}_{sid}_{self.model_type}_best_hyperparams.json",
                   "w") as f:
            json.dump({
                "model": self.model_type.upper(), "station_id": sid,
                "target": target, **best_p,
                "global_seed": cfg.global_seed,
                "run_seeds": cfg.run_seeds, "n_runs": cfg.n_runs,
            }, f, indent=2)

        tag = self.model_type.upper()
        print(f"      {tag}: {cfg.n_runs} independent runs ...")
        run_preds, run_metrics_list, run_epochs = [], [], []
        run_train_times, run_infer_times = [], []
        true_out = None

        for run_idx, seed in enumerate(cfg.run_seeds):
            set_global_seeds(seed)
            tf.keras.backend.clear_session()
            model = self._build(cfg.lag, best_p)
            t0 = time.perf_counter()
            hist = model.fit(X_tr_dl, y_tr, validation_split=0.1,
                               epochs=cfg.epochs, batch_size=cfg.batch_size,
                               callbacks=dl_callbacks(cfg.patience), verbose=0)
            run_train_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            pred_sc = model.predict(X_te_dl, verbose=0).flatten()
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
                       f"{target}_{sid}_{s_safe}_{tag}_run{run_idx + 1}_history.json",
                       "w") as f:
                json.dump({k: [float(v) for v in vals]
                            for k, vals in hist.history.items()}, f, indent=2)

        best_run_idx = int(np.argmin([r["RMSE"] for r in run_metrics_list]))
        pred_final = run_preds[best_run_idx]

        # Retrain the best-seed run once more, then persist model weights
        set_global_seeds(cfg.run_seeds[best_run_idx])
        tf.keras.backend.clear_session()
        model = self._build(cfg.lag, best_p)
        model.fit(X_tr_dl, y_tr, validation_split=0.1,
                   epochs=cfg.epochs, batch_size=cfg.batch_size,
                   callbacks=dl_callbacks(cfg.patience), verbose=0)
        model.save(cfg.models_dir / f"{target}_{sid}_{s_safe}_{tag}_best.keras")

        m_agg = self.metrics.aggregate_runs(run_metrics_list)
        m_agg["train_time_s"] = round(float(np.mean(run_train_times)), 4)
        m_agg["infer_time_s"] = round(float(np.mean(run_infer_times)), 4)

        run_summary = {
            "model": tag, "station_id": sid, "target": target,
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
        with open(cfg.hp_dir /
                   f"{target}_{sid}_{self.model_type}_run_summary.json", "w") as f:
            json.dump(run_summary, f, indent=2)

        print(f"      {tag}: RMSE={m_agg['RMSE']:.4f}±{m_agg['RMSE_std']:.4f}  "
              f"MAE={m_agg['MAE']:.4f}±{m_agg['MAE_std']:.4f}  "
              f"R2={m_agg['R2']:.4f}±{m_agg['R2_std']:.4f}  "
              f"train={m_agg['train_time_s']:.3f}s  infer={m_agg['infer_time_s']:.3f}s")

        return m_agg, pred_final, true_out
