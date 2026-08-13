"""
Central configuration for the Algeria temperature forecasting study.

Every other class in this package receives a single `Config` instance
instead of relying on module-level globals, which makes the pipeline
testable and lets multiple configurations (e.g. different lags or
output directories) coexist in the same process.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass(frozen=True)
class StationInfo:
    """Metadata for a single meteorological station."""
    name: str
    zone: str


@dataclass
class Config:
    # ── Reproducibility ─────────────────────────────────────────────
    global_seed: int = 42
    run_seeds: List[int] = field(
        default_factory=lambda: [42, 123, 456, 789, 2024])
    n_runs: int = 5

    # ── Experiment settings ──────────────────────────────────────────
    lag: int = 7
    epochs: int = 100
    batch_size: int = 32
    patience: int = 15
    acc_window: int = 15  # days either side of DOY for climatology smoothing

    # ── Train / test split (train: everything strictly before
    #    test_start; test: test_start..test_end inclusive).
    #    test_start is 2024-01-20 (not 2024-01-19) because the GFS
    #    data for every station starts at 2024-01-20 — using
    #    2024-01-19 left a 1-day gap that GFS couldn't cover. ────────
    test_start: pd.Timestamp = pd.Timestamp("2024-07-21")
    test_end: pd.Timestamp = pd.Timestamp("2026-07-21")

    # ── Stations : 9 Algerian met stations across 3 climate zones ────
    stations: Dict[int, StationInfo] = field(default_factory=lambda: {
        60360: StationInfo("Annaba",        "Mediterranean"),
        60390: StationInfo("Dar El Beida",  "Mediterranean"),
        60490: StationInfo("Oran/Es Senia", "Mediterranean"),
        60419: StationInfo("Constantine",   "Semi-arid"),
        60475: StationInfo("Tebessa",       "Semi-arid"),
        60535: StationInfo("Djelfa",        "Semi-arid"),
        60566: StationInfo("Ghardaia",      "Saharan"),
        60571: StationInfo("Bechar",        "Saharan"),
        60590: StationInfo("El Golea",      "Saharan"),
    })

    # ── Grid search spaces ─────────────────────────────────────────
    dl_grid: Dict[str, list] = field(default_factory=lambda: {
        "filters":     [32, 64],
        "lstm_units":  [32, 64],
        "dropout":     [0.1, 0.2],
        "lr":          [1e-3, 5e-4],
        "kernel_size": [3, 5],
    })
    svr_grid: Dict[str, list] = field(default_factory=lambda: {
        "C":       [0.1, 1.0, 10.0, 100.0],
        "epsilon": [0.01, 0.05, 0.1, 0.5],
        "kernel":  ["rbf", "linear"],
    })

    arima_max_pq: int = 3
    sarima_s: int = 7
    sarima_max_pq: int = 2

    model_order: List[str] = field(default_factory=lambda: [
        "Persistence", "Climatology", "GFS",
        "ARIMA", "SARIMA", "ANN", "SVR",
        "LSTM", "CNN-LSTM",
    ])

    zone_colors: Dict[str, str] = field(default_factory=lambda: {
        "Mediterranean": "#2196F3",
        "Semi-arid":     "#FF9800",
        "Saharan":       "#F44336",
    })

    # ── Paths (overridable via ALGERIA_DATA / ALGERIA_OUT env vars) ──
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get(
        "ALGERIA_DATA",
        r"C:\Users\USER\PycharmProjects\GitHubProject\Datasets")))
    out_dir: Path = field(default_factory=lambda: Path(os.environ.get(
        "ALGERIA_OUT",
        r"C:\Users\USER\PycharmProjects\GitHubProject\Results\Univariate_v10")))

    def __post_init__(self):
        # Input paths
        self.stations_path = self.data_dir / "Selected Stations"  # station_{id}.csv
        self.gfs_path = self.data_dir / "GFS_Data"                # gfs_station_{id}.csv

        # Output paths
        self.models_dir = self.out_dir / "saved_models"
        self.history_dir = self.out_dir / "training_history"
        self.gs_dir = self.out_dir / "grid_search"
        self.preds_dir = self.out_dir / "predictions"
        self.hp_dir = self.out_dir / "hyperparameters"

        for d in (self.out_dir, self.models_dir, self.history_dir,
                  self.gs_dir, self.preds_dir, self.hp_dir):
            d.mkdir(parents=True, exist_ok=True)
