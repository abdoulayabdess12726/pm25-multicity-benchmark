"""Download PM2.5 AND NO2 hourly data from London Air Quality Network (LAQN).
v2: ajoute NO2 pour comparabilité features avec Beijing."""
import requests
import pandas as pd
import time
from pathlib import Path

OUT = Path("data/london_laqn")
OUT.mkdir(parents=True, exist_ok=True)

# 12 stations sélectionnées (les 9 confirmées valides + 3 backups)
STATIONS = {
    "KC1": "North Kensington",
    "BL0": "Bloomsbury",
    "BT4": "Brent Ikea",
    "GN0": "Greenwich Plumstead",
    "WM0": "Westminster",
    "TH4": "Tower Hamlets Roadside",
    "CT3": "Camden Swiss Cottage",
    "LB4": "Lewisham Honor Oak Park",
    "LH0": "Hillingdon Harlington",
    "RB7": "Redbridge Gardner Close",
    "EA8": "Ealing Horn Lane",
    "ST6": "Sutton Wallington",
}

SPECIES = ["PM25", "NO2"]  # 2 polluants
START, END = "2020-01-01", "2023-12-31"

base = "https://api.erg.ic.ac.uk/AirQuality/Data/SiteSpecies"

print(f"Downloading {len(STATIONS)} stations x {len(SPECIES)} species ({START} to {END})...\n")

success_count = 0
total = len(STATIONS) * len(SPECIES)

for code, name in STATIONS.items():
    for species in SPECIES:
        url = f"{base}/SiteCode={code}/SpeciesCode={species}/StartDate={START}/EndDate={END}/csv"
        print(f"  {code} ({name}) {species}...", end=" ", flush=True)
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200 and len(r.text) > 200:
                with open(OUT / f"LAQN_{code}_{species}.csv", "w") as f:
                    f.write(r.text)
                print(f"OK ({len(r.text):,} bytes)")
                success_count += 1
            else:
                print(f"FAIL (status={r.status_code}, size={len(r.text)})")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.5)

print(f"\nDownloaded {success_count}/{total} files.")

# Métadonnées coordonnées
print("\nFetching station metadata...")
meta_url = "https://api.erg.ic.ac.uk/AirQuality/Information/MonitoringSites/GroupName=London/Json"
try:
    meta = requests.get(meta_url, timeout=60).json()
    sites = meta["Sites"]["Site"]
    coords = []
    for s in sites:
        if s["@SiteCode"] in STATIONS:
            coords.append({
                "station": s["@SiteCode"],
                "name": s["@SiteName"],
                "lat": float(s["@Latitude"]),
                "lon": float(s["@Longitude"]),
            })
    pd.DataFrame(coords).to_csv(OUT / "station_coords.csv", index=False)
    print(f"Saved {len(coords)} station coordinates.")
except Exception as e:
    print(f"Metadata fetch failed: {e}")
