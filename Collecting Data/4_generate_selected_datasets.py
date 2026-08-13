'''
This script selects the Algerian weather stations best suited for analysis by finding,
for each station, the most recent continuous (gap-free) block of tmin and tmax data.
Stations with more than 7 years of continuous data (ending July 21, 2026) are kept.

It then aligns all selected stations to a shared date range (the latest common start
date among them through July 21, 2026), trims each station's data to that range, drops
any remaining incomplete rows, and saves:
1. One CSV per selected station (date, tmin, tmax) in "Datasets/Selected Stations".
2. A summary CSV listing all selected stations with their row counts and date ranges.
'''

import pandas as pd
from pathlib import Path

# ==========================================================
# Configuration
# ==========================================================
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent

# File locations
stations_file = base_dir / "Files" / "all_algerian_stations.csv"
dataset_dir = base_dir / "Datasets" / "Meteostat Datasets"
output_dir = base_dir / "Datasets" / "Selected Stations"
files_dir = base_dir / "Files"

# Create output directories if they don't exist
output_dir.mkdir(parents=True, exist_ok=True)

min_years = 7
common_end = pd.Timestamp("2026-07-21")

print(f"Output directory for station files: {output_dir}")
print(f"Summary file will be saved in: {files_dir}\n")


# ==========================================================
# Function: Continuous data start
# ==========================================================
def continuous_start(series):
    """
    Returns the earliest date of the continuous non-NaN block
    ending on the last day of the study period.
    Returns NaT if the last day is NaN.
    """
    if pd.isna(series.iloc[-1]):
        return pd.NaT

    for i in range(len(series) - 2, -1, -1):
        if pd.isna(series.iloc[i]):
            return series.index[i + 1]

    return series.index[0]


# ==========================================================
# Read stations (use 'wmo' as station identifier)
# ==========================================================
stations = pd.read_csv(stations_file, dtype={"wmo": str})

# ==========================================================
# First pass: Find stations with continuous data for tmin and tmax
# ==========================================================
print("First pass: Finding stations with continuous data...")
station_continuous_start = {}

for idx, row in stations.iterrows():
    station_id = row["wmo"]          # WMO code is the station identifier
    station_name = row["name"]

    csv_file = dataset_dir / f"station_{station_id}_daily.csv"

    if not csv_file.exists():
        continue

    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            continue

        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

        # Check if tmin and tmax columns exist
        if "tmin" not in df.columns or "tmax" not in df.columns:
            continue

        # Filter to the study period
        df = df.loc["1996-01-01":common_end]

        # Ensure every day exists
        expected_dates = pd.date_range(start="1996-01-01", end=common_end, freq="D")
        df = df.reindex(expected_dates)

        # Get continuous start for tmin
        tmin_start = continuous_start(df["tmin"])
        # Get continuous start for tmax
        tmax_start = continuous_start(df["tmax"])

        if pd.notna(tmin_start) and pd.notna(tmax_start):
            # Take the later start date (when both are available)
            start_date = max(tmin_start, tmax_start)

            # Calculate years of continuous data
            years = (common_end - start_date).days / 365.25

            if years > min_years:
                station_continuous_start[station_id] = start_date
                print(f"   {station_id} - {station_name}: {years:.1f} years (from {start_date.strftime('%Y-%m-%d')})")

    except Exception as e:
        continue

# ==========================================================
# Check if any stations found
# ==========================================================
if not station_continuous_start:
    print(f"\nNo stations found with more than {min_years} years of continuous data.")
    exit(1)

# Find the common start date (latest start date among all selected stations)
common_start = max(station_continuous_start.values())

print(f"\nFound {len(station_continuous_start)} stations with > {min_years} years of continuous data")
print(f"Common start date: {common_start.strftime('%Y-%m-%d')}")
print(f"Common end date: {common_end.strftime('%Y-%m-%d')}")
print(f"Total days in common period: {(common_end - common_start).days + 1:,}\n")

# ==========================================================
# Second pass: Generate final datasets
# ==========================================================
print("Second pass: Generating final datasets...")
expected_dates = pd.date_range(start=common_start, end=common_end, freq="D")
processed_count = 0
station_info = []

for station_id, start_date in station_continuous_start.items():
    # Get station info
    station_row = stations[stations["wmo"] == station_id]
    if station_row.empty:
        continue

    station_name = station_row.iloc[0]["name"]
    latitude = station_row.iloc[0]["latitude"]
    longitude = station_row.iloc[0]["longitude"]

    print(f"Processing {station_id} - {station_name}")

    csv_file = dataset_dir / f"station_{station_id}_daily.csv"

    try:
        df = pd.read_csv(csv_file)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

        # Filter to the common period
        df = df.loc[common_start:common_end]

        # Ensure every day exists
        df = df.reindex(expected_dates)

        # Select only: tmin, and tmax columns
        df_final = df[["tmin", "tmax"]].copy()

        # Remove rows with any NaN
        nan_count = df_final.isna().any(axis=1).sum()
        if nan_count > 0:
            df_final = df_final.dropna()
            print(f"   Removed {nan_count} rows with NaN values")

        if len(df_final) == 0:
            print("   No valid data after removing NaN values.\n")
            continue

        # Reset index to make 'date' a column
        df_final = df_final.reset_index()
        df_final = df_final.rename(columns={"index": "date"})

        # Save to CSV with just the station number as filename
        output_file = output_dir / f"station_{station_id}.csv"
        df_final.to_csv(output_file, index=False)

        print(f"   Saved: {output_file}")
        print(f"   Rows: {len(df_final):,}")
        print(
            f"   Period: {df_final['date'].iloc[0].strftime('%Y-%m-%d')} to {df_final['date'].iloc[-1].strftime('%Y-%m-%d')}")
        print()

        processed_count += 1
        station_info.append({
            "station_id": station_id,
            "station_name": station_name,
            "latitude": latitude,
            "longitude": longitude,
            "rows": len(df_final),
            "start_date": df_final['date'].iloc[0].strftime("%Y-%m-%d"),
            "end_date": df_final['date'].iloc[-1].strftime("%Y-%m-%d"),
        })

    except Exception as e:
        print(f"   Error: {e}\n")

# ==========================================================
# Save summary in Files folder
# ==========================================================
if station_info:
    summary_df = pd.DataFrame(station_info)
    summary_file = files_dir / "selected_stations_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print("=" * 70)
    print(f"Completed! Processed {processed_count} stations")
    print(f"Station files saved in: {output_dir}")
    print(f"Summary file saved in: {summary_file}")
    print("=" * 70)
else:
    print("No stations were processed.")