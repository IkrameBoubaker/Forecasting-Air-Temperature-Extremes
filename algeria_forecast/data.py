"""Data loading and train/test splitting for station observations and GFS forecasts."""

import pandas as pd

from .config import Config


class DataLoader:
    """Loads observed station data and GFS forecasts, and splits train/test."""

    def __init__(self, config: Config):
        self.config = config

    def load_station(self, station_id: int, target: str) -> pd.Series:
        """
        Load a single target column from the unified station CSV.

        File   : <stations_path>/station_{id}.csv
        Columns: date, tmax, tmin

        No NaN filling is performed — the data is assumed complete; a
        warning is printed if unexpected NaNs are found.
        """
        fpath = self.config.stations_path / f"station_{station_id}.csv"
        df = pd.read_csv(fpath, parse_dates=["date"], index_col="date",
                          usecols=["date", target])
        series = df[target].sort_index()

        n_nan = series.isna().sum()
        if n_nan > 0:
            print(f"      WARNING: {n_nan} NaN(s) found in station "
                  f"{station_id} {target} — please inspect the raw data.")
        return series

    def load_gfs(self, station_id: int, target: str):
        """
        Load the GFS (NWP) forecast series for a station, or None if the
        file doesn't exist. Also checks coverage of the test period.
        """
        fpath = self.config.gfs_path / f"gfs_station_{station_id}.csv"
        if not fpath.exists():
            print(f"      WARNING: GFS file not found: {fpath.name}")
            return None

        df = pd.read_csv(fpath, parse_dates=["date"], index_col="date",
                          usecols=["date", target])
        series = df[target].sort_index()

        cfg = self.config
        gfs_test = series.loc[cfg.test_start:cfg.test_end]
        expected_days = (cfg.test_end - cfg.test_start).days + 1
        coverage = len(gfs_test) / expected_days * 100
        if coverage < 90:
            print(f"      WARNING: GFS coverage over test period is only "
                  f"{coverage:.1f}% ({len(gfs_test)}/{expected_days} days) "
                  f"for station {station_id}.")
        return series

    def load_all(self, target: str) -> dict:
        """Load the target series for every configured station."""
        data = {}
        print(f"\n  Loading {target.upper()} datasets ...")
        for sid, info in self.config.stations.items():
            try:
                s = self.load_station(sid, target)
                data[sid] = s
                print(f"    {info.name:15s} ({sid})  n={len(s)}  "
                      f"{s.index[0].date()} → {s.index[-1].date()}")
            except FileNotFoundError:
                print(f"    WARNING: file not found for station {sid}")
        return data

    def split(self, series: pd.Series):
        """
        Train : everything strictly before config.test_start
        Test  : config.test_start .. config.test_end (inclusive)
        """
        cfg = self.config
        train = series[series.index < cfg.test_start]
        test = series[(series.index >= cfg.test_start) &
                      (series.index <= cfg.test_end)]
        return train, test
