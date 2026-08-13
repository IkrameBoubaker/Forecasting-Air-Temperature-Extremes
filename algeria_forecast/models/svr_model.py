"""Support Vector Regression forecaster with grid search."""

import itertools
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

from ..config import Config
from ..metrics import Metrics


class SVRModel:
    """SVR with a grid search over C, epsilon, and kernel."""

    def __init__(self, config: Config, metrics: Metrics):
        self.config = config
        self.metrics = metrics

    def run(self, X_tr, y_tr, X_te, y_te, scaler, sid, target):
        cfg = self.config
        grid = cfg.svr_grid
        n_combos = len(grid["C"]) * len(grid["epsilon"]) * len(grid["kernel"])
        print(f"      SVR: grid search over {n_combos} configs ...")

        n_val = max(30, int(0.1 * len(X_tr)))
        X_val, y_val = X_tr[-n_val:], y_tr[-n_val:]
        X_trn, y_trn = X_tr[:-n_val], y_tr[:-n_val]

        best_rmse, best_params, gs_log = np.inf, None, []
        for C_val, eps, kernel in itertools.product(
                grid["C"], grid["epsilon"], grid["kernel"]):
            svr = SVR(C=C_val, epsilon=eps, kernel=kernel, cache_size=500)
            svr.fit(X_trn, y_trn)
            rmse_val = float(np.sqrt(mean_squared_error(y_val, svr.predict(X_val))))
            gs_log.append({"C": C_val, "epsilon": eps, "kernel": kernel,
                            "val_rmse": round(rmse_val, 6)})
            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_params = {"C": C_val, "epsilon": eps, "kernel": kernel}

        pd.DataFrame(gs_log).to_csv(
            cfg.gs_dir / f"{target}_{sid}_svr_gridsearch.csv", index=False)
        print(f"      SVR: best={best_params}  val_RMSE={best_rmse:.4f}")

        with open(cfg.hp_dir / f"{target}_{sid}_svr_best_hyperparams.json", "w") as f:
            json.dump({"model": "SVR", "station_id": sid,
                       "target": target, **best_params}, f, indent=2)

        final_svr = SVR(**best_params, cache_size=500)
        t0 = time.perf_counter()
        final_svr.fit(X_tr, y_tr)
        t_train = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred_sc = final_svr.predict(X_te)
        t_infer = time.perf_counter() - t0

        pred = scaler.inverse_transform(pred_sc.reshape(-1, 1)).flatten()
        true = scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()
        m = self.metrics.compute(true, pred)
        m["train_time_s"] = round(t_train, 4)
        m["infer_time_s"] = round(t_infer, 4)
        return m, pred, true
