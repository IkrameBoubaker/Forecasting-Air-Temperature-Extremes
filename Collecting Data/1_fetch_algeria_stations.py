'''
This script retrieves the complete list of weather stations located in Algeria using the Meteostat API
and exports the data to a CSV file.
'''

# pip install meteostat==1.7.6

from meteostat import Stations
from pathlib import Path

# --------------------------------------------------------------------
# Portable path: script is in "Collecting Data/", we save to "../Files/"
# --------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
output_path = project_root / 'Files' / 'all_algerian_stations.csv'

# Create the Files/ directory if it doesn't exist
output_path.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# Fetch and save
# --------------------------------------------------------------------
stations = Stations()
algerian_stations = stations.region(country='DZ')
station_data = algerian_stations.fetch()

print(station_data.head())

station_data.to_csv(output_path, index=False)
print(f"Data saved to: {output_path}")