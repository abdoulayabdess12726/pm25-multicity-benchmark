"""Download historical weather (TEMP, PRES, WIND, DEWP) for each Madrid station
via Open-Meteo Historical Weather API (free, no key required)."""
import requests
import pandas as pd
import time
from pathlib import Path

OUT = Path("data/madrid_openaq")
OUT.mkdir(parents=True, exist_ok=True)

coords_df = pd.read_csv(OUT / "station_coords.csv")
print(f"Loaded {len(coords_df)} station coordinates.\n")

START = "2020-01-01"
END   = "2023-12-31"

HOURLY_VARS = [
    "temperature_2m",
    "surface_pressure",
    "dew_point_2m",
    "wind_speed_10m",
]

base = "https://archive-api.open-meteo.com/v1/archive"

print(f"Downloading weather for {len(coords_df)} Madrid stations from {START} to {END}...\n")

success = 0
for _, row in coords_df.iterrows():
    code = row["station"]
    lat, lon = row["lat"], row["lon"]
    
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": START, "end_date": END,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "UTC",
    }
    
    safe_code = code.replace(" ", "_").replace("/", "_")[:30]
    print(f"  {safe_code} ({lat:.3f}, {lon:.3f})...", end=" ", flush=True)
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
            df.to_csv(OUT / f"WEATHER_{safe_code}.csv", index=False)
            print(f"OK ({len(df):,} rows)")
            success += 1
        else:
            print(f"FAIL (status={r.status_code})")
    except Exception as e:
        print(f"ERROR: {e}")
    time.sleep(1)

print(f"\nDownloaded weather for {success}/{len(coords_df)} stations.")
