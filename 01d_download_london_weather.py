"""Download historical weather (TEMP, PRES, WIND, DEWP) for each London station
via Open-Meteo Historical Weather API (free, no key required).
Saves one CSV per station: data/london_laqn/WEATHER_<SITECODE>.csv"""
import requests
import pandas as pd
import time
from pathlib import Path

OUT = Path("data/london_laqn")
OUT.mkdir(parents=True, exist_ok=True)

# Charger les coordonnées des stations
coords_df = pd.read_csv(OUT / "station_coords.csv")
print(f"Loaded {len(coords_df)} station coordinates.\n")

START = "2020-01-01"
END   = "2023-12-31"

# Variables Open-Meteo correspondant aux features Beijing
# Beijing : ['PM2.5', 'NO2', 'TEMP', 'PRES', 'DEWP', 'WSPM']
HOURLY_VARS = [
    "temperature_2m",          # -> TEMP
    "surface_pressure",        # -> PRES  
    "dew_point_2m",            # -> DEWP
    "wind_speed_10m",          # -> WSPM
]

base = "https://archive-api.open-meteo.com/v1/archive"

print(f"Downloading weather for {len(coords_df)} stations from {START} to {END}...")
print(f"Variables: {HOURLY_VARS}\n")

success = 0
for _, row in coords_df.iterrows():
    code = row["station"]
    lat, lon = row["lat"], row["lon"]
    
    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": START,
        "end_date":   END,
        "hourly":     ",".join(HOURLY_VARS),
        "timezone":   "UTC",
    }
    
    print(f"  {code} ({lat:.3f}, {lon:.3f})...", end=" ", flush=True)
    try:
        r = requests.get(base, params=params, timeout=120)
        if r.status_code == 200:
            data = r.json()
            hourly = data["hourly"]
            df = pd.DataFrame({
                "datetime": pd.to_datetime(hourly["time"]),
                "TEMP": hourly["temperature_2m"],
                "PRES": hourly["surface_pressure"],
                "DEWP": hourly["dew_point_2m"],
                "WSPM": hourly["wind_speed_10m"],
            })
            df.to_csv(OUT / f"WEATHER_{code}.csv", index=False)
            print(f"OK ({len(df):,} rows)")
            success += 1
        else:
            print(f"FAIL (status={r.status_code})")
    except Exception as e:
        print(f"ERROR: {e}")
    time.sleep(1)  # Open-Meteo rate limit

print(f"\nDownloaded weather for {success}/{len(coords_df)} stations.")
