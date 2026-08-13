"""End-to-end orchestration of the forecasting study for one target (tmax/tmin)."""

import json
import platform

import numpy as np
import pandas as pd

from .config import Config
from .data import DataLoader
from .metrics import Metrics
from .models.ann_model import ANNModel
from .models.baselines import ClimatologyModel, GFSModel, PersistenceModel
from .models.classical import ARIMAModel, SARIMAModel
from .models.deep_learning import DeepLearningModel
from .models.svr_model import SVRModel
from .plotting import Plotter
from .preprocessing import Preprocessor
from .stats_tests import StatisticalTests
from .utils import safe_name, set_global_seeds

SEP = "─" * 75
SEP2 = "═" * 75


class ForecastPipeline:
    """
    Orchestrates the full baseline / classical / ML / DL comparison for
    a single forecast target ("tmax" or "tmin") across all configured
    stations, plus the cross-target (Tmax vs Tmin) comparison plots.
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        set_global_seeds(self.config.global_seed)

        self.data_loader = DataLoader(self.config)
        self.preprocessor = Preprocessor()
        self.metrics = Metrics(acc_window=self.config.acc_window)
        self.stats_tests = StatisticalTests()
        self.plotter = Plotter(self.config)

        self.persistence_model = PersistenceModel(self.metrics)
        self.climatology_model = ClimatologyModel(self.metrics, window=7)
        self.gfs_model = GFSModel(self.metrics)
        self.arima_model = ARIMAModel(self.config, self.metrics)
        self.sarima_model = SARIMAModel(self.config, self.metrics)
        self.ann_model = ANNModel(self.config, self.metrics)
        self.svr_model = SVRModel(self.config, self.metrics)
        self.lstm_model = DeepLearningModel("lstm", self.config, self.metrics)
        self.cnn_lstm_model = DeepLearningModel("cnn_lstm", self.config, self.metrics)

    # ── Environment logging ─────────────────────────────────────────
    def log_environment(self) -> dict:
        import sklearn
        import scipy
        import statsmodels.api as sm
        import tensorflow as tf

        cfg = self.config
        env = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "keras": tf.keras.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "statsmodels": sm.__version__,
            "global_seed": cfg.global_seed,
            "n_runs": cfg.n_runs,
            "run_seeds": cfg.run_seeds,
            "lag": cfg.lag,
            "test_start": str(cfg.test_start.date()),
            "test_end": str(cfg.test_end.date()),
        }
        gpus = tf.config.list_physical_devices("GPU")
        env["gpu"] = [g.name for g in gpus] if gpus else "CPU only"

        path = cfg.out_dir / "environment.json"
        with open(path, "w") as f:
            json.dump(env, f, indent=2)

        print(f"\n{'─' * 50}")
        print("  ENVIRONMENT")
        for k, v in env.items():
            print(f"    {k:<20s}: {v}")
        print(f"  Saved → {path.name}")
        print(f"{'─' * 50}")
        return env

    # ── Main pipeline for one target ────────────────────────────────
    def run(self, target: str):
        cfg = self.config
        print(f"\n{SEP2}")
        print(f"  TARGET : {target.upper()}  |  LAG : {cfg.lag}  |  N_RUNS : {cfg.n_runs}")
        print(f"  Train  : up to {(cfg.test_start - pd.Timedelta(days=1)).date()}")
        print(f"  Test   : {cfg.test_start.date()} → {cfg.test_end.date()}")
        print(SEP2)

        all_data = self.data_loader.load_all(target)
        all_results, seasonal_rows, dm_rows = [], [], []

        for sid, series in all_data.items():
            info = cfg.stations[sid]
            s_name, zone = info.name, info.zone
            s_safe = safe_name(s_name)

            print(f"\n{SEP}")
            print(f"  Station : {s_name} ({sid})  |  Zone : {zone}")
            print(SEP)

            row = self._run_station(sid, series, s_name, zone, s_safe, target,
                                      seasonal_rows, dm_rows)
            all_results.append(row)

        df_dm_all = pd.DataFrame()
        if dm_rows:
            df_dm_all = pd.concat(dm_rows, ignore_index=True)
            df_dm_all = self.stats_tests.apply_global_fdr(df_dm_all)
            p_dm = cfg.out_dir / f"{target}_dm_tests_lag{cfg.lag}.csv"
            df_dm_all.to_csv(p_dm, index=False)
            print(f"\n  DM CSV saved: {p_dm.name}")

        df_results = pd.DataFrame(all_results)
        df_sea = pd.DataFrame(seasonal_rows)
        df_results.to_csv(cfg.out_dir / f"{target}_results_lag{cfg.lag}.csv", index=False)
        df_sea.to_csv(cfg.out_dir / f"{target}_seasonal_cnnlstm_lag{cfg.lag}.csv", index=False)
        print("\n  Results CSV saved.")

        self.plotter.heatmap_all(df_results, target)
        self.plotter.seasonal(df_sea, target)
        self.plotter.dm_heatmap(df_dm_all, target)
        self.plotter.training_histories(target)

        return df_results, df_sea, df_dm_all

    # ── Per-station work ─────────────────────────────────────────────
    def _run_station(self, sid, series, s_name, zone, s_safe, target,
                       seasonal_rows, dm_rows):
        cfg = self.config
        train_s, test_s = self.data_loader.split(series)
        gfs_s = self.data_loader.load_gfs(sid, target)
        doy_clim = self.metrics.build_doy_climatology(train_s)

        print(f"  Train : {len(train_s)} days  "
              f"({train_s.index[0].date()} → {train_s.index[-1].date()})")
        print(f"  Test  : {len(test_s)} days  "
              f"({test_s.index[0].date()} → {test_s.index[-1].date()})")

        row = {"station": s_name, "id": sid, "zone": zone}

        # ── Persistence ──────────────────────────────────────────────
        m_pers, pred_pers, t_train, t_infer = self.persistence_model.run(train_s, test_s)
        acc_pers = self.metrics.compute_acc(test_s.values, pred_pers, test_s.index, doy_clim)
        row.update({f"Persistence_{k}": v for k, v in m_pers.items()})
        row["Persistence_ACC"] = acc_pers
        row["Persistence_train_time_s"] = round(t_train, 4)
        row["Persistence_infer_time_s"] = round(t_infer, 4)
        self.metrics.print_metrics("Persistence", m_pers, acc=acc_pers)

        # ── Climatology ──────────────────────────────────────────────
        m_clim, pred_clim, t_train, t_infer = self.climatology_model.run(train_s, test_s)
        acc_clim = self.metrics.compute_acc(test_s.values, pred_clim, test_s.index, doy_clim)
        row.update({f"Climatology_{k}": v for k, v in m_clim.items()})
        row["Climatology_ACC"] = acc_clim
        row["Climatology_train_time_s"] = round(t_train, 4)
        row["Climatology_infer_time_s"] = round(t_infer, 4)
        self.metrics.print_metrics("Climatology", m_clim, acc=acc_clim)

        # ── GFS ──────────────────────────────────────────────────────
        m_gfs, pred_gfs, t_train, t_infer = self.gfs_model.run(test_s, gfs_s)
        acc_gfs = self.metrics.compute_acc(test_s.values, pred_gfs, test_s.index, doy_clim)
        row.update({f"GFS_{k}": v for k, v in m_gfs.items()})
        row["GFS_ACC"] = acc_gfs
        row["GFS_train_time_s"] = round(t_train, 4)
        row["GFS_infer_time_s"] = round(t_infer, 4)
        self.metrics.print_metrics("GFS", m_gfs, acc=acc_gfs)

        # ── ARIMA ────────────────────────────────────────────────────
        # NOTE: t_train here includes AIC-based (p,1,q) order selection
        # plus the rolling refit — i.e. the full one-off cost to produce
        # a deployable model, not just a single .fit() call. This is
        # not directly comparable to the ANN/SVR/LSTM/CNN-LSTM training
        # times below, which exclude grid search (see those models'
        # docstrings) — worth noting explicitly if reporting timings
        # together in the paper.
        m_arima, pred_arima, t_train, t_infer = self.arima_model.run(train_s, test_s, sid, target)
        acc_arima = self.metrics.compute_acc(test_s.values, pred_arima, test_s.index, doy_clim)
        row.update({f"ARIMA_{k}": v for k, v in m_arima.items()})
        row["ARIMA_ACC"] = acc_arima
        row["ARIMA_train_time_s"] = round(t_train, 4)
        row["ARIMA_infer_time_s"] = round(t_infer, 4)
        self.metrics.print_metrics("ARIMA", m_arima, acc=acc_arima)

        # ── SARIMA ───────────────────────────────────────────────────
        m_sarima, pred_sarima, t_train, t_infer = self.sarima_model.run(train_s, test_s, sid, target)
        acc_sarima = self.metrics.compute_acc(test_s.values, pred_sarima, test_s.index, doy_clim)
        row.update({f"SARIMA_{k}": v for k, v in m_sarima.items()})
        row["SARIMA_ACC"] = acc_sarima
        row["SARIMA_train_time_s"] = round(t_train, 4)
        row["SARIMA_infer_time_s"] = round(t_infer, 4)
        self.metrics.print_metrics("SARIMA", m_sarima, acc=acc_sarima)

        # ── Shared ML/DL preprocessing (lag-only: 7 lagged temperature
        #    values, no month sin/cos or rolling std) ──────────────────
        (scaler, X_tr, y_tr, X_te, y_te,
         X_tr_dl, X_te_dl) = self.preprocessor.prepare(train_s, test_s, cfg.lag)

        # ── ANN ──────────────────────────────────────────────────────
        m_ann, pred_ann, _ = self.ann_model.run(
            X_tr, y_tr, X_te, y_te, scaler, sid, target, s_safe)
        row.update({f"ANN_{k}": v for k, v in m_ann.items()})
        self.metrics.print_metrics("ANN", m_ann, std_rmse=m_ann.get("RMSE_std"))

        # ── SVR ──────────────────────────────────────────────────────
        m_svr, pred_svr, _ = self.svr_model.run(
            X_tr, y_tr, X_te, y_te, scaler, sid, target)
        row.update({f"SVR_{k}": v for k, v in m_svr.items()})
        self.metrics.print_metrics("SVR", m_svr)

        # ── LSTM ─────────────────────────────────────────────────────
        m_lstm, pred_lstm, _ = self.lstm_model.run(
            X_tr_dl, y_tr, X_te_dl, y_te, scaler, sid, target, s_safe)
        row.update({f"LSTM_{k}": v for k, v in m_lstm.items()})
        self.metrics.print_metrics("LSTM", m_lstm, std_rmse=m_lstm.get("RMSE_std"))

        # ── CNN-LSTM ─────────────────────────────────────────────────
        m_cnnlstm, pred_cnnlstm, true_cnnlstm = self.cnn_lstm_model.run(
            X_tr_dl, y_tr, X_te_dl, y_te, scaler, sid, target, s_safe)
        row.update({f"CNN-LSTM_{k}": v for k, v in m_cnnlstm.items()})
        self.metrics.print_metrics("CNN-LSTM", m_cnnlstm, std_rmse=m_cnnlstm.get("RMSE_std"))

        # ── Align all model predictions to the full test window ──────
        # Every model's prediction array — baselines, classical, and
        # ML/DL — already covers the entire test period one-for-one
        # with test_s.index, starting at the FIRST test day. This is
        # because Preprocessor.prepare() builds ML/DL sequences across
        # the train/test boundary specifically so the first test day
        # already has a valid prediction (using the trailing training
        # values as lag context — see prepare()'s docstring). No
        # lag-based date offset is needed here. (Applying one
        # previously shifted the DL prediction values relative to
        # their true dates by `lag` days, corrupting the DM tests,
        # ACC values, seasonal breakdown, and prediction CSVs — fixed.)
        n_dl = min(len(test_s.index), len(true_cnnlstm), len(pred_ann),
                   len(pred_svr), len(pred_lstm), len(pred_cnnlstm),
                   len(pred_pers), len(pred_clim), len(pred_gfs),
                   len(pred_arima), len(pred_sarima))
        dl_dates = test_s.index[:n_dl]
        true_aligned = true_cnnlstm[:n_dl]

        predictions_dict = {
            "Persistence": pred_pers[:n_dl],
            "Climatology": pred_clim[:n_dl],
            "GFS":         pred_gfs[:n_dl],
            "ARIMA":       pred_arima[:n_dl],
            "SARIMA":      pred_sarima[:n_dl],
            "ANN":         pred_ann[:n_dl],
            "SVR":         pred_svr[:n_dl],
            "LSTM":        pred_lstm[:n_dl],
            "CNN-LSTM":    pred_cnnlstm[:n_dl],
        }

        # ── Save predictions CSV ──────────────────────────────────────
        pred_df = pd.DataFrame({"date": dl_dates, "observed": true_aligned})
        for mn, p_arr in predictions_dict.items():
            if p_arr is not None:
                pred_df[mn] = p_arr
        pred_df.to_csv(cfg.preds_dir / f"{target}_{sid}_{s_safe}_predictions.csv", index=False)

        # ── ACC (aligned window) ──────────────────────────────────────
        for mn, p_arr in predictions_dict.items():
            if p_arr is not None and not np.all(np.isnan(p_arr)):
                row[f"{mn}_ACC"] = self.metrics.compute_acc(
                    true_aligned, p_arr, dl_dates, doy_clim)

        # ── RMSE Skill Score ───────────────────────────────────────────
        ss_val = self.metrics.rmse_skill_score(float(m_cnnlstm["RMSE"]), float(m_gfs["RMSE"]))
        row["RMSE_SS_vs_GFS"] = ss_val
        ss_str = f"{ss_val:+.4f}" if not np.isnan(ss_val) else "—"
        cmp = "CNN-LSTM better" if ss_val > 0 else "GFS better"
        print(f"\n  RMSE Skill Score (CNN-LSTM vs GFS): {ss_str}  [{cmp}]")

        # ── DM tests ───────────────────────────────────────────────────
        df_dm = self.stats_tests.run_dm_tests(true_aligned, predictions_dict)
        if not df_dm.empty:
            df_dm.insert(0, "station", s_name)
            df_dm.insert(1, "zone", zone)
            df_dm.insert(2, "target", target)
            dm_rows.append(df_dm)
            self._print_dm(df_dm)

        # ── Seasonal ───────────────────────────────────────────────────
        sea = self.metrics.seasonal_metrics(true_aligned, pred_cnnlstm[:n_dl], dl_dates)
        seasonal_rows.append({"station": s_name, "zone": zone, **sea})
        print("\n  Seasonal RMSE (CNN-LSTM):  " +
              "  ".join(f"{s}={sea.get(f'{s}_RMSE', '—')}"
                          for s in ["Winter", "Spring", "Summer", "Autumn"]))

        # ── Forecast plot ────────────────────────────────────────────
        self.plotter.forecast_all(dl_dates, true_aligned, pred_cnnlstm[:n_dl],
                                     predictions_dict["LSTM"], predictions_dict["ANN"],
                                     predictions_dict["SVR"], predictions_dict["GFS"],
                                     s_name, zone, target, sid)

        return row

    @staticmethod
    def _print_dm(df_dm: pd.DataFrame) -> None:
        print("\n  DM tests (CNN-LSTM vs all) — with BH-FDR:")
        print(f"    {'Comparison':<30s}  {'DM':>7s}  {'p_raw':>8s}  {'p_adj':>8s}  "
              f"{'Sig_raw':>7s}  {'Sig_FDR':>7s}  {'Better':>10s}")
        print(f"    {'─' * 30}  {'─' * 7}  {'─' * 8}  {'─' * 8}  "
              f"{'─' * 7}  {'─' * 7}  {'─' * 10}")
        for _, r in df_dm.iterrows():
            print(f"    {r['comparison']:<30s}  {r['DM_stat']:>7.4f}  "
                  f"{r['p_value']:>8.4f}  {r['p_value_adj']:>8.4f}  "
                  f"{'✓' if r['significant_raw'] else '—':>7s}  "
                  f"{'✓' if r['significant_fdr'] else '—':>7s}  "
                  f"{r['better']:>10s}")

    # ── Cross-target (Tmax vs Tmin) comparison ──────────────────────
    def compare_targets(self, df_tmax: pd.DataFrame, df_tmin: pd.DataFrame) -> None:
        self.plotter.compare_targets(df_tmax, df_tmin)

    def compare_dm(self, df_dm_tmax: pd.DataFrame, df_dm_tmin: pd.DataFrame) -> None:
        self.plotter.compare_dm(df_dm_tmax, df_dm_tmin)
