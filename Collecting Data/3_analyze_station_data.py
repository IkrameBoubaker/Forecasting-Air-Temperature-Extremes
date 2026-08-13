'''
This script analyzes the daily weather data files downloaded for each Algerian station
(from January 1, 1996 to July 21, 2026), checking data completeness and identifying
the most recent continuous (gap-free) block of data for each weather variable.

It produces two outputs:
1. A full data-quality summary CSV for all stations with usable data files.
2. A filtered CSV listing only stations with at least 7 years of continuous data
   for at least one variable, ending on the last available date.
'''

import pandas as pd
from pathlib import Path

# ==========================================================
# Configuration
# ==========================================================
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent

stations_file = base_dir / "Files" / "all_algerian_stations.csv"
dataset_dir = base_dir / "Datasets" / "Meteostat Datasets"  # Where station data files are
output_dir = base_dir / "Files"  # Where output CSV files will be saved

output_csv = output_dir / "meteostat_station_data_quality_summary.csv"
output_quality_csv = output_dir / "stations_with_7years_continuous_data.csv"

start_date = "1996-01-01"
end_date = "2026-07-21"
expected_dates = pd.date_range(start=start_date, end=end_date, freq="D")
total_days = len(expected_dates)
min_years = 7


# ==========================================================
# Function: Continuous data start
# ==========================================================
def continuous_start(series):
    #Returns earliest date of continuous non-NaN block ending on last day.
    if pd.isna(series.iloc[-1]):
        return pd.NaT
    for i in range(len(series) - 2, -1, -1):
        if pd.isna(series.iloc[i]):
            return series.index[i + 1]
    return series.index[0]


# ==========================================================
# Read stations and analyze
# ==========================================================
# Use 'wmo' as station ID column (it contains the WMO code)
stations = pd.read_csv(stations_file, dtype={"wmo": str})
summary = []

print(f"Looking for station data files in: {dataset_dir}")
print(f"Output files will be saved to: {output_dir}")
print(f"Total stations: {len(stations)}\n")

for idx, row in stations.iterrows():
    station_id = row["wmo"]          # WMO code is the station identifier
    station_name = row["name"]
    print(f"Analyzing {station_id} ({idx + 1}/{len(stations)}) - {station_name}")

    csv_file = dataset_dir / f"station_{station_id}_daily.csv"

    if not csv_file.exists():
        print("   File not found\n")
        continue

    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            print("   Empty file.\n")
            continue

        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").loc[start_date:end_date].reindex(expected_dates)

        missing_days = df.isna().all(axis=1).sum()
        print(f"   Total days: {total_days:,}")
        print(f"   Missing days: {missing_days:,}")
        print("   Continuous data until 2026-07-21")

        result = {"station_id": station_id, "station_name": station_name,
                  "total_days": total_days, "missing_days": missing_days}

        for col in df.columns:
            start = continuous_start(df[col])
            start_str = start.strftime("%Y-%m-%d") if pd.notna(start) else ""
            print(f"   {col:<5}: {start_str}")
            result[f"{col}_continuous_from"] = start_str

        summary.append(result)
        print()
    except Exception as e:
        print(f"   Error: {e}\n")

# ==========================================================
# Check if any data was processed
# ==========================================================
if not summary:
    print("\n" + "=" * 70)
    print("ERROR: No station files were found or processed.")
    print(f"Please check that the files exist in: {dataset_dir}")
    print("Expected files like: station_60351_daily.csv")
    print("=" * 70)
    exit(1)

# ==========================================================
# Save summary
# ==========================================================
summary_df = pd.DataFrame(summary)
base_cols = ["station_id", "station_name", "total_days", "missing_days"]
var_cols = [col for col in summary_df.columns if col.endswith("_continuous_from")]
summary_df[base_cols + var_cols].to_csv(output_csv, index=False)

print(f"\n{'=' * 70}\nAnalysis completed.\nSummary saved to:\n{output_csv}")

# ==========================================================
# Stations with >= 7 years continuous data
# ==========================================================
end = pd.Timestamp(end_date)
quality_rows = []

for _, station in stations.iterrows():
    # Use wmo as station_id
    station_id = station["wmo"]
    # Find matching row in summary
    summary_row = summary_df[summary_df["station_id"] == station_id]
    if summary_row.empty:
        continue

    row = {"station_id": station_id,
           "station_name": station["name"],
           "latitude": station["latitude"],
           "longitude": station["longitude"]}

    has_variable = False
    for col in summary_df.columns:
        if col.endswith("_continuous_from"):
            var_name = col.replace("_continuous_from", "")
            if pd.notna(summary_row.iloc[0][col]) and summary_row.iloc[0][col] != "":
                start = pd.to_datetime(summary_row.iloc[0][col])
                if (end - start).days / 365.25 >= min_years:
                    row[var_name.upper()] = start.strftime("%Y-%m-%d")
                    has_variable = True
                    continue
            row[var_name.upper()] = ""

    if has_variable:
        quality_rows.append(row)

if quality_rows:
    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(output_quality_csv, index=False)
    print(f"\nSaved:\n{output_quality_csv}")
    print(f"\nTotal stations: {len(quality_df)}")
else:
    print(f"\nNo stations with >= {min_years} years of continuous data found.")