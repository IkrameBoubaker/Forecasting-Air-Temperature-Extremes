'''
This script loads the list of Algerian weather stations, loops through each station ID,
and downloads the daily historical weather data from January 1, 1996 to July 21, 2026.
The output is a separate CSV file for each station.
'''

from datetime import datetime
import meteostat as ms
import pandas as pd
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

input_path = project_root / 'Files' / 'all_algerian_stations.csv'
output_dir = project_root / 'Datasets' / 'Meteostat Datasets'
output_dir.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(input_path)
print(data.columns)

# Use 'wmo' as the station ID (it holds the WMO code used by Meteostat)
ids = data['wmo']
names = data['name']

for station_id, name in zip(ids, names):
    # Skip if wmo is missing (should not happen, but safe)
    if pd.isna(station_id):
        continue

    station_id = str(int(station_id))  # ensure it's a string (Meteostat expects str)
    print(station_id, '=====================> ', name)

    start = datetime(1996, 1, 1)
    end = datetime(2026, 7, 21)

    daily_data = ms.Daily(station_id, start, end).fetch()

    daily_data.to_csv(output_dir / f'station_{station_id}_daily.csv')

    print(f"Saved daily data for station {station_id} to CSV.")