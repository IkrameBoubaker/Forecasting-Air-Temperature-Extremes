# Algeria Daily Temperature Forecasting — Univariate Study

Forecasts daily **Tmax** and **Tmin** at 9 Algerian meteorological stations
using naive baselines, an NWP baseline (GFS), classical statistical models
(ARIMA/SARIMA), and machine-learning / deep-learning models (SVR, ANN,
LSTM, CNN-LSTM).

## Features

- **Baselines**: Persistence, Climatology, GFS (numerical weather prediction)
- **Classical**: ARIMA and SARIMA with AIC-based order search
- **ML/DL**: SVR, ANN, LSTM, and CNN-LSTM, each with grid search over
  hyperparameters; neural models are trained 5 times with different seeds
  and reported as mean ± std
- **Evaluation**: RMSE, MAE, R², Anomaly Correlation Coefficient (ACC),
  and RMSE Skill Score vs. GFS
- **Statistical testing**: Diebold-Mariano test (CNN-LSTM vs every other
  model) with Benjamini-Hochberg FDR correction, both per-station and
  globally across all stations/targets
- **Seasonal breakdown** and a full set of diagnostic plots (heatmaps,
  forecast overlays, DM significance maps, training curves)

## Project layout

```
algeria-temp-forecast/
├── main.py                        # entry point: python main.py
├── check_gfs_gap.py               # checks a station's GFS CSV for missing/duplicate/NaN dates in the test period 
├── algeria_forecast/
    ├── config.py                  # Config dataclass: paths, stations, hyperparameters
    ├── data.py                    # DataLoader: reads CSVs, train/test split
    ├── preprocessing.py           # Preprocessor: scaling, lag sequences
    ├── metrics.py                 # Metrics: RMSE/MAE/R2, ACC, skill score
    ├── stats_tests.py             # StatisticalTests: Diebold-Mariano + BH-FDR
    ├── plotting.py                # Plotter: all matplotlib charts
    ├── pipeline.py                # ForecastPipeline: orchestrates everything
    ├── utils.py                   # seeding, safe filenames, time formatting
    └── models/
        ├── dl_builders.py         # Keras architectures (ANN/LSTM/CNN-LSTM)
        ├── baselines.py           # PersistenceModel, ClimatologyModel, GFSModel
        ├── classical.py           # ARIMAModel, SARIMAModel
        ├── ann_model.py           # ANNModel
        ├── svr_model.py           # SVRModel
        └── deep_learning.py       # DeepLearningModel (shared LSTM/CNN-LSTM logic)
├── Collecting Data/
│   ├── 1_fetch_algeria_stations.py        # Retrieve list of all Algerian weather stations
│   ├── 2_download_daily_weather_data.py   # Download raw daily observations per station
│   ├── 3_analyze_station_data.py          # QC/completeness analysis of downloaded data
│   ├── 4_generate_selected_datasets.py    # Filter and build final selected-station datasets
│   └── 5_get_gfs_data.py                  # Fetch GFS operational forecast data per station
├── Datasets/
│   ├── GFS_Data/                          # GFS forecast series per station (benchmark)
│   ├── Meteostat Datasets/                # Raw daily observations for all candidate stations
│   └── Selected Stations/                 # Final cleaned datasets for the 9 stations used
└── Files/
    ├── all_algerian_stations.csv                      # Full list of candidate stations
    ├── meteostat_station_data_quality_summary.csv      # Per-station data-quality summary
    ├── stations_with_7years_continuous_data.csv        # Stations with ≥7 years continuous data
    └── selected_stations_summary.csv                   # Summary of the 9 stations selected
```

## Data layout expected

```
<ALGERIA_DATA>/
├── Selected Stations/
│   └── station_{id}.csv           # columns: date, tmax, tmin
└── GFS_Data/
    └── gfs_station_{id}.csv       # columns: date, tmax, tmin
```

## Setup

```bash
pip install -r requirements.txt
```

Set the input/output paths via environment variables (defaults point to
placeholder Windows paths in `config.py`):

```bash
# Windows (cmd)
set ALGERIA_DATA=C:\path\to\Datasets
set ALGERIA_OUT=C:\path\to\Results

# Windows (PowerShell)
$env:ALGERIA_DATA = "C:\path\to\Datasets"
$env:ALGERIA_OUT = "C:\path\to\Results"

# macOS / Linux
export ALGERIA_DATA=/path/to/Datasets
export ALGERIA_OUT=/path/to/Results
```

## Run

```bash
python main.py
```

This runs the full study for both `tmax` and `tmin`, saving:

- `*_results_lag7.csv` — per-station metrics for every model
- `*_dm_tests_lag7.csv` — Diebold-Mariano test results
- `*_seasonal_cnnlstm_lag7.csv` — seasonal RMSE breakdown
- `saved_models/` — best CNN-LSTM/LSTM Keras models per station
- `hyperparameters/` — best hyperparameters and multi-run summaries (JSON)
- `predictions/` — per-station prediction CSVs
- `grid_search/` — grid search logs (CSV)
- assorted `.png` diagnostic plots

## Programmatic use

```python
from algeria_forecast import Config, ForecastPipeline

config = Config()                     # or Config(lag=14, n_runs=3, ...)
pipeline = ForecastPipeline(config)
pipeline.log_environment()

df_tmax, df_tmax_sea, df_dm_tmax = pipeline.run("tmax")
df_tmin, df_tmin_sea, df_dm_tmin = pipeline.run("tmin")

pipeline.compare_targets(df_tmax, df_tmin)
pipeline.compare_dm(df_dm_tmax, df_dm_tmin)
```

## Hyperparameter Search Spaces

Grid search is performed independently for each model, station, and target
variable. The search spaces are:

| Model      | Hyperparameters |
|------------|------------------|
| ANN        | Hidden units: {32, 64}; Dropout: {0.1, 0.2}; Learning rate: {1×10⁻³, 5×10⁻⁴} |
| LSTM       | LSTM units: {32, 64}; Dropout: {0.1, 0.2}; Learning rate: {1×10⁻³, 5×10⁻⁴} |
| CNN-LSTM   | Filters: {32, 64}; Kernel size: {3, 5}; LSTM units: {32, 64}; Dropout: {0.1, 0.2}; Learning rate: {1×10⁻³, 5×10⁻⁴} |
| SVR        | C: {0.1, 1, 10, 100}; ε: {0.01, 0.05, 0.1, 0.5}; Kernel: {RBF, Linear} |
| ARIMA      | p, q ∈ {0, 1, 2, 3}, d = 1 (order selected by AIC) |
| SARIMA     | p, q ∈ {0, 1, 2}, d = 1; seasonal order fixed at (1, 0, 1, 7) |

The configuration minimizing validation RMSE (last 10% of the training
sequence) is selected and retrained on the full training set. Final selected
values per station/variable are saved in `hyperparameters/`
A consolidated table of final hyperparameters across all stations and variables
is also provided in `Files/final_hyperparameters.xlsx` for quick reference
without needing to parse the JSON files.
## Notes

- MAPE is intentionally excluded from the metrics: Algeria's winter Tmin
  can sit near 0 °C, which makes percentage errors unstable and misleading.
- Scaling is fit on the training set only; test-set sequences are built
  across the train/test boundary so the first test-day prediction has
  the correct lag context, without leaking future test values.
- All random seeds (Python, NumPy, TensorFlow) are fixed for reproducibility;
  neural models additionally run 5 independent trainings per station/target
  and report mean ± std.
