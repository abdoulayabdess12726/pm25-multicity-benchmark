"""Combine Madrid OpenAQ PM2.5 + Open-Meteo weather into Beijing/London-compatible format.
v2: handles truncated filenames and special characters from OpenAQ download."""
import pandas as pd
import numpy as np
from pathlib import Path

IN = Path("data/madrid_openaq")
OUT = Path("data/madrid_processed")
OUT.mkdir(parents=True, exist_ok=True)

FEATURES_ORDER = ["PM2.5", "NO2", "TEMP", "PRES", "DEWP", "WSPM"]

PHYSICAL_LIMITS = {
    "PM2.5": (0, 500),
    "NO2":   (0, 400),
    "TEMP":  (-30, 50),
    "PRES":  (950, 1060),
    "DEWP":  (-30, 30),
    "WSPM":  (0, 50),
}

coords_df = pd.read_csv(IN / "station_coords.csv")
all_codes = coords_df["station"].tolist()
print(f"Processing {len(all_codes)} stations from coords...\n")

# Lister tous les CSV OpenAQ effectivement présents
openaq_files = list(IN.glob("OPENAQ_*.csv"))
print(f"Found {len(openaq_files)} OpenAQ CSV files")

def find_openaq_file(station_name, openaq_files):
    """Match station_name to actual file (handle truncation + special chars)."""
    # Try exact match first
    candidates = [f for f in openaq_files if f.stem == f"OPENAQ_{station_name}"]
    if candidates:
        return candidates[0]
    
    # Try truncated match: filename starts with first 15 chars of station
    prefix = station_name[:15]
    candidates = [f for f in openaq_files 
                  if f.stem.startswith(f"OPENAQ_{prefix}")]
    if candidates:
        return candidates[0]
    
    # Try fuzzy: first word match
    first_word = station_name.split()[0][:10]
    candidates = [f for f in openaq_files 
                  if first_word.lower() in f.stem.lower()]
    if candidates:
        return candidates[0]
    
    return None

def find_weather_file(station_name):
    """Match weather file (uses safe_code from download)."""
    safe_code = station_name.replace(" ", "_").replace("/", "_")[:30]
    p = IN / f"WEATHER_{safe_code}.csv"
    if p.exists():
        return p
    # Try shorter
    for n in range(30, 5, -1):
        safe_code = station_name.replace(" ", "_").replace("/", "_")[:n]
        p = IN / f"WEATHER_{safe_code}.csv"
        if p.exists():
            return p
    return None

def clip_outliers(df, features_limits):
    for feat, (lo, hi) in features_limits.items():
        if feat in df.columns:
            df[feat] = df[feat].clip(lower=lo, upper=hi)
            p995 = df[feat].quantile(0.995)
            if pd.notna(p995):
                df[feat] = df[feat].clip(upper=p995)
    return df

station_dfs = {}
station_pm25_coverage = {}

for code in all_codes:
    pm_path = find_openaq_file(code, openaq_files)
    if pm_path is None:
        print(f"  {code}: PM2.5 file not found, SKIP")
        station_pm25_coverage[code] = 0.0
        continue
    
    pm25 = pd.read_csv(pm_path)
    pm25["datetime"] = pd.to_datetime(pm25["datetime"], utc=True).dt.tz_localize(None)
    pm25 = pm25.set_index("datetime").rename(columns={"value": "PM2.5"})
    pm25 = pm25.resample("1h").mean()
    pm25_cov_raw = pm25["PM2.5"].notna().mean()
    station_pm25_coverage[code] = pm25_cov_raw
    
    weather_path = find_weather_file(code)
    if weather_path is None:
        print(f"  {code}: WEATHER not found (looked for safe_code variants), SKIP")
        station_pm25_coverage[code] = 0.0
        continue
    
    weather = pd.read_csv(weather_path, parse_dates=["datetime"]).set_index("datetime")
    
    combined = pd.concat([pm25[["PM2.5"]], weather], axis=1)
    
    # Madrid n'a pas de NO2 en accès gratuit OpenAQ. On met une valeur constante par station
    # (médiane PM2.5) pour préserver la dimension feature, mais ça ne porte pas de signal inter-station
    no2_proxy = combined["PM2.5"].median()
    combined["NO2"] = no2_proxy
    combined = combined[FEATURES_ORDER]
    combined = clip_outliers(combined, PHYSICAL_LIMITS)
    
    # Interpolation
    combined = combined.interpolate(method="linear", limit=6).ffill().bfill()
    combined = combined.resample("1h").mean()
    combined = combined.dropna(subset=["PM2.5"])
    
    station_dfs[code] = combined
    print(f"  {code}: {len(combined):,} rows, raw cov {pm25_cov_raw:.1%}")

# Filter low-coverage
print(f"\n=== Filtering stations with PM2.5 raw coverage < 30% ===")
# Note: OpenAQ Madrid has lower density than LAQN (4-hourly typical), so threshold lowered to 30%
valid_codes = [c for c, cov in station_pm25_coverage.items() if cov >= 0.30]
dropped = [c for c, cov in station_pm25_coverage.items() if cov < 0.30]
if dropped:
    print(f"Dropped (low coverage): {[(c, f'{station_pm25_coverage[c]:.1%}') for c in dropped]}")

# Filter constant test
print(f"\n=== Filtering stations with constant test PM2.5 (std < 0.5) or too short ===")
constant_codes = []
for code in list(valid_codes):
    df = station_dfs[code]
    T = len(df)
    if T < 5000:
        constant_codes.append((code, f"T={T}_too_short"))
        valid_codes.remove(code)
        continue
    t2 = int(0.85 * T)
    test_std = df["PM2.5"].iloc[t2:].std()
    if test_std < 0.5:
        constant_codes.append((code, f"std={test_std:.2f}"))
        valid_codes.remove(code)
if constant_codes:
    print(f"Dropped: {constant_codes}")

print(f"Kept: {valid_codes} ({len(valid_codes)} stations)")

if len(valid_codes) < 5:
    raise SystemExit(f"ERREUR: trop peu de stations ({len(valid_codes)}).")

# Aligner timestamps
common_idx = None
for code in valid_codes:
    idx = station_dfs[code].index
    common_idx = idx if common_idx is None else common_idx.intersection(idx)
print(f"\nCommon timestamps: {len(common_idx):,}")

if len(common_idx) < 1000:
    raise SystemExit(f"ERREUR: trop peu de timestamps communs ({len(common_idx)}).")

# Save PM2.5 wide
pm25_wide = pd.DataFrame({
    code: station_dfs[code].loc[common_idx, "PM2.5"].values
    for code in valid_codes
}, index=common_idx)
pm25_wide.to_csv(OUT / "madrid_pm25_hourly.csv")
print(f"Saved: madrid_pm25_hourly.csv ({pm25_wide.shape})")

# Save parquet
combined_3d = []
for code in valid_codes:
    df = station_dfs[code].loc[common_idx, FEATURES_ORDER].copy()
    df["station"] = code
    combined_3d.append(df.reset_index())

full_df = pd.concat(combined_3d, ignore_index=True)
full_df.to_parquet(OUT / "madrid_full_hourly.parquet", index=False)
print(f"Saved: madrid_full_hourly.parquet ({full_df.shape})")

# Save valid coords
valid_coords = coords_df[coords_df["station"].isin(valid_codes)]
valid_coords.to_csv(OUT / "station_coords_valid.csv", index=False)
print(f"Saved: station_coords_valid.csv ({len(valid_coords)} stations)")

print(f"\n=== SUMMARY ===")
print(f"Stations: {len(valid_codes)}")
print(f"Timestamps: {len(common_idx):,}")
print(f"Date range: {common_idx.min()} -> {common_idx.max()}")
print(f"Final coverage: {full_df[FEATURES_ORDER].notna().mean().mean():.1%}")
print(f"\nPM2.5 stats per station:")
print(full_df.groupby('station')['PM2.5'].describe()[['mean','std','min','max']])

# Verification: NO2 must NOT vary across rows within a station (it's a per-station constant)
print(f"\nNO2 placeholder check (should be 0 std within station):")
print(full_df.groupby('station')['NO2'].std().head())
