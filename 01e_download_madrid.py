"""Download PM2.5 hourly data from Madrid via OpenAQ API.
Madrid serves as the third city to validate the heterogeneity-vs-GCN-performance relationship.

Madrid is expected to fall between Beijing (homogeneous, basin) and London (heterogeneous, mixed)
in terms of spatial heterogeneity, providing a useful intermediate point for the benchmark.

Usage:
    pip install openaq
    python 01e_download_madrid.py
"""
import os
import time
import pandas as pd
import requests
from pathlib import Path

OUT = Path("data/madrid_openaq")
OUT.mkdir(parents=True, exist_ok=True)

# Approach 1: OpenAQ direct API (no Python client needed)
# Madrid coordinates and 25 km radius
LAT, LON = 40.4168, -3.7038
RADIUS_M = 25000
START = "2020-01-01T00:00:00Z"
END   = "2023-12-31T23:00:00Z"

# Step 1: Find PM2.5 sensors in Madrid via OpenAQ v3 API
print("Step 1: Finding PM2.5 sensors in Madrid via OpenAQ...")
locations_url = "https://api.openaq.org/v3/locations"
params = {
    "coordinates": f"{LAT},{LON}",
    "radius": RADIUS_M,
    "parameters_id": 2,  # 2 = PM2.5
    "limit": 50,
}

# OpenAQ v3 needs API key (free tier)
# Sign up at https://explore.openaq.org/login then get key from /account
API_KEY = os.environ.get("OPENAQ_API_KEY", "")
headers = {"X-API-Key": API_KEY} if API_KEY else {}

if not API_KEY:
    print("\nWARNING: OPENAQ_API_KEY not set. OpenAQ v3 requires a free API key.")
    print("Sign up at https://explore.openaq.org/account to get one.")
    print("Then run: export OPENAQ_API_KEY='your_key_here'")
    print("\nProceeding without key (may rate-limit fast)...\n")

try:
    r = requests.get(locations_url, params=params, headers=headers, timeout=60)
    print(f"  HTTP status: {r.status_code}")
    if r.status_code == 401:
        print("  ERROR: Invalid or missing API key. Set OPENAQ_API_KEY env variable.")
        print("\n  Alternative: use European Environment Agency (EEA) data directly:")
        print("  https://discomap.eea.europa.eu/Map/UI/MapViewer?serviceID=AirQualityViewerHome")
        raise SystemExit(1)
    if r.status_code != 200:
        print(f"  ERROR: {r.text[:300]}")
        raise SystemExit(1)
    
    data = r.json()
    locations = data.get("results", [])
    print(f"  Found {len(locations)} location(s) with PM2.5")
except Exception as e:
    print(f"  ERROR: {e}")
    raise SystemExit(1)

# Filter Madrid locations (within radius)
madrid_locations = []
for loc in locations:
    coords = loc.get("coordinates")
    if coords and coords.get("latitude") and coords.get("longitude"):
        madrid_locations.append({
            "id": loc["id"],
            "name": loc.get("name", f"loc_{loc['id']}"),
            "lat": coords["latitude"],
            "lon": coords["longitude"],
            "sensors": [s for s in loc.get("sensors", []) if s.get("parameter", {}).get("id") == 2],
        })

# Keep only locations with active PM2.5 sensors
madrid_locations = [l for l in madrid_locations if l["sensors"]]
print(f"\n  Locations with PM2.5 sensors: {len(madrid_locations)}")
print(f"  Selecting top 12 closest to center...")

# Sort by distance to Madrid center
import math
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

for loc in madrid_locations:
    loc["distance_km"] = haversine(LAT, LON, loc["lat"], loc["lon"])
madrid_locations.sort(key=lambda x: x["distance_km"])
selected = madrid_locations[:12]

print(f"\n  Selected {len(selected)} stations:")
for loc in selected:
    print(f"    {loc['name']} ({loc['lat']:.3f}, {loc['lon']:.3f}, {loc['distance_km']:.1f} km)")

# Save coordinates
coords_df = pd.DataFrame([
    {"station": loc["name"][:20], "name": loc["name"],
     "lat": loc["lat"], "lon": loc["lon"], "id": loc["id"]}
    for loc in selected
])
coords_df.to_csv(OUT / "station_coords.csv", index=False)
print(f"\nSaved coordinates to {OUT}/station_coords.csv")

# Step 2: Download hourly measurements per sensor
print(f"\nStep 2: Downloading hourly PM2.5 from {START} to {END}...")
measurements_url = "https://api.openaq.org/v3/sensors/{sensor_id}/measurements/hourly"

success = 0
for loc in selected:
    sensor_id = loc["sensors"][0]["id"]
    print(f"  {loc['name'][:30]}... (sensor {sensor_id})", end=" ", flush=True)
    
    all_records = []
    page = 1
    try:
        while True:
            url = measurements_url.format(sensor_id=sensor_id)
            params_m = {
                "datetime_from": START,
                "datetime_to": END,
                "limit": 1000,
                "page": page,
            }
            r = requests.get(url, params=params_m, headers=headers, timeout=120)
            if r.status_code != 200:
                break
            d = r.json()
            results = d.get("results", [])
            if not results:
                break
            all_records.extend(results)
            if len(results) < 1000:
                break
            page += 1
            time.sleep(0.5)
        
        if all_records:
            df = pd.DataFrame([{
                "datetime": rec["period"]["datetimeFrom"]["utc"],
                "value": rec["value"],
            } for rec in all_records])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.to_csv(OUT / f"OPENAQ_{loc['name'][:20]}.csv", index=False)
            print(f"OK ({len(df):,} rows)")
            success += 1
        else:
            print("EMPTY")
    except Exception as e:
        print(f"ERROR: {e}")
    time.sleep(1)

print(f"\nDownloaded {success}/{len(selected)} stations.")
print(f"\nIf success < 8, consider:")
print(f"  - Getting an OPENAQ_API_KEY (free, immediate)")
print(f"  - Trying European Environment Agency (EEA) data instead")
print(f"  - Selecting a different city with better OpenAQ coverage")
