'''
This script downloads GFS (Global Forecast System) day-ahead weather forecast data
for each of the previously selected Algerian weather stations, using the Open-Meteo
"Previous Runs" API.

For each station, it retrieves the hourly temperature that GFS predicted 24 hours in
advance (temperature_2m_previous_day1) from January 1, 2024 to July 21, 2026, then
aggregates it into daily maximum and minimum temperatures (tmax, tmin).

Only days with a FULL 24 hours of non-null data are kept - any day with even a
single missing hour is excluded from the output (both the per-station CSV and
the start_date/end_date/rows reported in the summary).

The output is one CSV file per station saved in "Datasets/GFS_Data", along with a
summary CSV listing all successfully processed stations and their date ranges.
'''

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import time
from datetime import datetime
from pathlib import Path

# ==========================================================
# Configuration
# ==========================================================
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent

# File locations
stations_summary_file = base_dir / "Files" / "selected_stations_summary.csv"
gfs_output_dir = base_dir / "Datasets" / "GFS_Data"

# Create output directory if it doesn't exist
gfs_output_dir.mkdir(parents=True, exist_ok=True)

print(f"GFS data will be saved to: {gfs_output_dir}\n")

# ==========================================================
# Setup Open-Meteo client with retry logic
# ==========================================================
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# ==========================================================
# Read selected stations
# ==========================================================
stations = pd.read_csv(stations_summary_file)

HOURS_PER_DAY = 24  # a day is only kept if it has this many valid (non-null) hours

# The actual period you want in the final output
REQUESTED_START_DATE = "2024-07-21"
REQUESTED_END_DATE = "2026-07-21"

# The API is queried with one extra day at each end. This is because
# "timezone": "auto" converts the UTC-based hourly data to local time, and the
# last (and first) local day can otherwise be truncated to fewer than 24 hours
# near the edges of the requested UTC window. Requesting a one-day buffer gives
# enough raw UTC hours to fully cover the first/last local calendar day, and the
# extra buffer days are trimmed off afterwards.
API_START_DATE = (pd.Timestamp(REQUESTED_START_DATE) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
API_END_DATE = (pd.Timestamp(REQUESTED_END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

print(f"Processing {len(stations)} stations")
print(f"Period: {REQUESTED_START_DATE} to {REQUESTED_END_DATE}\n")
print("=" * 70)


# ==========================================================
# Function to get GFS data for a station
# ==========================================================
def get_gfs_daily_tmax_tmin(latitude, longitude, station_id, station_name):
    url = "https://previous-runs-api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": API_START_DATE,
        "end_date": API_END_DATE,
        "hourly": ["temperature_2m_previous_day1"],
        "models": "gfs_seamless",
        "timezone": "auto"
    }

    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        # Extract hourly data
        hourly = response.Hourly()
        hourly_data = hourly.Variables(0).ValuesAsNumpy()

        # Create date range
        dates = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )

        # Create DataFrame with hourly data (may contain NaNs for missing archive hours)
        df = pd.DataFrame({"date": dates, "temp": hourly_data})
        df["date"] = df["date"].dt.date

        # Count how many NON-NULL hours each day actually has
        valid_hours_per_day = df.groupby("date")["temp"].apply(lambda s: s.notna().sum())

        # Keep only days with a full 24 hours of valid data
        complete_days = valid_hours_per_day[valid_hours_per_day == HOURS_PER_DAY].index

        # Drop NaN hours before aggregating max/min
        df_valid = df.dropna(subset=["temp"])

        # Aggregate to daily Tmax and Tmin
        gfs_daily = df_valid.groupby(df_valid["date"])["temp"].agg(["max", "min"])
        gfs_daily.columns = ["tmax", "tmin"]

        # Restrict to complete days only
        gfs_daily = gfs_daily.loc[gfs_daily.index.isin(complete_days)]

        # Convert index to datetime
        gfs_daily.index = pd.to_datetime(gfs_daily.index)
        gfs_daily = gfs_daily.sort_index()

        # Trim back down to the originally requested date range (drops the
        # one-day buffer added at each end to fix the local-time truncation issue)
        gfs_daily = gfs_daily.loc[
            (gfs_daily.index >= pd.Timestamp(REQUESTED_START_DATE)) &
            (gfs_daily.index <= pd.Timestamp(REQUESTED_END_DATE))
        ]

        return gfs_daily

    except Exception as e:
        print(f"   ERROR: {e}")
        return None

# ==========================================================
# Process each station
# ==========================================================
processed_count = 0
station_info = []

for idx, row in stations.iterrows():
    station_id = row["station_id"]
    station_name = row["station_name"]
    latitude = row["latitude"]
    longitude = row["longitude"]

    print(f"Processing {station_id} - {station_name} ({idx + 1}/{len(stations)})")

    # Get GFS data
    gfs_data = get_gfs_daily_tmax_tmin(latitude, longitude, station_id, station_name)

    if gfs_data is not None and len(gfs_data) > 0:
        # Save to CSV
        output_file = gfs_output_dir / f"gfs_station_{station_id}.csv"
        gfs_data.to_csv(output_file)

        print(f"   Saved: {output_file}")
        print(f"   Rows (complete 24h days): {len(gfs_data):,}")
        print(f"   Period: {gfs_data.index[0].strftime('%Y-%m-%d')} to {gfs_data.index[-1].strftime('%Y-%m-%d')}")
        print()

        processed_count += 1
        station_info.append({
            "station_id": station_id,
            "station_name": station_name,
            "latitude": latitude,
            "longitude": longitude,
            "rows": len(gfs_data),
            "start_date": gfs_data.index[0].strftime("%Y-%m-%d"),
            "end_date": gfs_data.index[-1].strftime("%Y-%m-%d"),
        })
    else:
        print("   No complete-day data retrieved.\n")

    # Small delay to avoid rate limiting
    time.sleep(0.5)

# ==========================================================
# Save summary
# ==========================================================
if station_info:
    summary_df = pd.DataFrame(station_info)
    gfs_summary_file = gfs_output_dir / "gfs_stations_summary.csv"

    try:
        summary_df.to_csv(gfs_summary_file, index=False)
    except PermissionError:
        # File is likely open in Excel/another program, or locked by sync/AV software.
        # Fall back to a timestamped filename so the summary is never lost.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_file = gfs_output_dir / f"gfs_stations_summary_{timestamp}.csv"
        print(f"   WARNING: Could not write to {gfs_summary_file} (file may be open elsewhere).")
        print(f"   Saving summary to fallback file instead: {fallback_file}")
        summary_df.to_csv(fallback_file, index=False)
        gfs_summary_file = fallback_file

    print("=" * 70)
    print(f"Completed! Processed {processed_count} stations")
    print(f"GFS data saved in: {gfs_output_dir}")
    print(f"Summary file: {gfs_summary_file}")
    print("=" * 70)
else:
    print("No stations were processed.")