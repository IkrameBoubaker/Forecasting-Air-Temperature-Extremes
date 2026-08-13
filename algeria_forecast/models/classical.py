"""Classical statistical time-series models: ARIMA and SARIMA."""

import json
import time
import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ..config import Config
from ..metrics import Metrics


class ARIMAModel:
    """ARIMA with AIC-based (p, 1, q) order search and rolling refit."""

    def __init__(self, config: Config, metrics: Metrics):
        self.config = config
        self.metrics = metrics

    def _select_order(self, train_vals):
        max_pq = self.config.arima_max_pq
        best_aic, best_order = np.inf, (1, 1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for p in range(0, max_pq + 1):
                for q in range(0, max_pq + 1):
                    try:
                        res = ARIMA(train_vals, order=(p, 1, q)).fit()
                        if res.aic < best_aic:
                            best_aic, best_order = res.aic, (p, 1, q)
                    except Exception:
                        pass
        return best_order

    def run(self, train_s, test_s, sid, target):
        cfg = self.config
        print(f"      ARIMA: selecting order (max p,q={cfg.arima_max_pq}) ...")
        t0 = time.perf_counter()
        order = self._select_order(train_s.values)
        print(f"      ARIMA: best order = {order}")

        with open(cfg.hp_dir / f"{target}_{sid}_arima_best_hyperparams.json", "w") as f:
            json.dump({"model": "ARIMA", "station_id": sid,
                       "target": target, "order": list(order)}, f, indent=2)

        history = list(train_s.values)
        preds = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(history, order=order).fit()
            for i, obs in enumerate(test_s.values):
                preds.append(float(model.forecast(steps=1)[0]))
                history.append(obs)
                if (i + 1) % 30 == 0:
                    try:
                        model = ARIMA(history, order=order).fit()
                    except Exception:
                        pass

        t_train = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = np.array(preds)
        m = self.metrics.compute(test_s.values, pred)
        return m, pred, t_train, time.perf_counter() - t0


class SARIMAModel:
    """Seasonal ARIMA with AIC-based order search (fixed weekly seasonality)."""

    def __init__(self, config: Config, metrics: Metrics):
        self.config = config
        self.metrics = metrics

    def _select_order(self, train_vals):
        max_pq, s = self.config.sarima_max_pq, self.config.sarima_s
        best_aic = np.inf
        best_order = ((1, 1, 1), (1, 0, 1, s))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for p in range(0, max_pq + 1):
                for q in range(0, max_pq + 1):
                    try:
                        res = SARIMAX(train_vals, order=(p, 1, q),
                                        seasonal_order=(1, 0, 1, s),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False).fit(disp=False)
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best_order = ((p, 1, q), (1, 0, 1, s))
                    except Exception:
                        pass
        return best_order

    def run(self, train_s, test_s, sid, target):
        cfg = self.config
        print("      SARIMA: selecting order ...")
        t0 = time.perf_counter()
        order, seasonal_order = self._select_order(train_s.values)
        print(f"      SARIMA: order={order}, seasonal={seasonal_order}")

        with open(cfg.hp_dir / f"{target}_{sid}_sarima_best_hyperparams.json", "w") as f:
            json.dump({"model": "SARIMA", "station_id": sid, "target": target,
                       "order": list(order),
                       "seasonal_order": list(seasonal_order)}, f, indent=2)

        history = list(train_s.values)
        preds = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(history, order=order, seasonal_order=seasonal_order,
                              enforce_stationarity=False,
                              enforce_invertibility=False).fit(disp=False)
            for i, obs in enumerate(test_s.values):
                preds.append(float(model.forecast(steps=1)[0]))
                history.append(obs)
                if (i + 1) % 30 == 0:
                    try:
                        model = SARIMAX(history, order=order,
                                          seasonal_order=seasonal_order,
                                          enforce_stationarity=False,
                                          enforce_invertibility=False).fit(disp=False)
                    except Exception:
                        pass

        t_train = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = np.array(preds)
        m = self.metrics.compute(test_s.values, pred)
        return m, pred, t_train, time.perf_counter() - t0
