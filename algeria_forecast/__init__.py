"""
algeria_forecast
================
Univariate daily temperature (Tmax / Tmin) forecasting study for 9
Algerian meteorological stations, comparing naive baselines, an NWP
baseline (GFS), classical statistical models (ARIMA/SARIMA), and
machine-learning / deep-learning models (SVR, ANN, LSTM, CNN-LSTM).

Quick start
-----------
    from algeria_forecast import Config, ForecastPipeline

    config = Config()
    pipeline = ForecastPipeline(config)
    pipeline.log_environment()
    df_tmax, df_tmax_sea, df_dm_tmax = pipeline.run("tmax")
    df_tmin, df_tmin_sea, df_dm_tmin = pipeline.run("tmin")
    pipeline.compare_targets(df_tmax, df_tmin)
    pipeline.compare_dm(df_dm_tmax, df_dm_tmin)
"""

from .config import Config, StationInfo
from .pipeline import ForecastPipeline

__all__ = ["Config", "StationInfo", "ForecastPipeline"]
__version__ = "1.0.0"
