"""Combine LAQN PM2.5 + NO2 + Open-Meteo weather into Beijing-compatible format.
v4: clip valeurs aberrantes (PM2.5 négatif, outliers > 99.5e percentile)
"""
import pandas as pd
import numpy as np
from pathlib import Path

IN = Path("data/london_laqn")
OUT = Path("data/london_processed")
OUT.mkdir(parents=True, exist_ok=True)

FEATURES_ORDER = ["PM2.5", "NO2", "TEMP", "PRES", "DEWP", "WSPM"]

# Limites physiques par feature (clip pour éviter outliers d'instrumentation)
PHYSICAL_LIMITS = {
    "PM2.5": (0, 500),       # PM2.5 toujours >= 0, max physique ~500 µg/m³
    "NO2":   (0, 400),       # NO2 toujours >= 0, max ~400 µg/m³
    "TEMP":  (-30, 50),      # Température raisonnable London
    "PRES":  (950, 1060),    # Pression atmosphérique raisonnable
    "DEWP":  (-30, 30),      # Point de rosée
    "WSPM":  (0, 50),        # Vitesse vent en m/s
}

coords_df = pd.read_csv(IN / "station_coords.csv")
all_codes = coords_df["station"].tolist()
print(f"Processing {len(all_codes)} stations...\n")

def load_pollutant_csv(path, target_col_name):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    date_col = df.columns[0]
    val_col  = df.columns[1]
    df = df[[date_col, val_col]].rename(columns={date_col: "datetime", val_col: target_col_name})
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime")
    df[target_col_name] = pd.to_numeric(df[target_col_name], errors="coerce")
    return df

def clip_outliers(df, features_limits):
    """Clip values to physical limits + cap at 99.5th percentile per feature."""
    for feat, (lo, hi) in features_limits.items():
        if feat in df.columns:
            # Clip aux limites physiques
            n_below = (df[feat] < lo).sum()
            n_above = (df[feat] > hi).sum()
            if n_below > 0 or n_above > 0:
                pass  # logged later
            df[feat] = df[feat].clip(lower=lo, upper=hi)
            # Cap à 99.5e percentile pour les outliers extrêmes restants
            p995 = df[feat].quantile(0.995)
            df[feat] = df[feat].clip(upper=p995)
    return df

station_dfs = {}
station_pm25_coverage = {}

for code in all_codes:
    parts = []
    
    pm25_path = IN / f"LAQN_{code}_PM25.csv"
    if pm25_path.exists():
        pm25 = load_pollutant_csv(pm25_path, "PM2.5")
        pm25_cov = pm25["PM2.5"].notna().mean()
        station_pm25_coverage[code] = pm25_cov
        parts.append(pm25)
    else:
        print(f"  {code}: PM2.5 missing, SKIP")
        station_pm25_coverage[code] = 0.0
        continue
    
    no2_path = IN / f"LAQN_{code}_NO2.csv"
    if no2_path.exists():
        no2 = load_pollutant_csv(no2_path, "NO2")
        parts.append(no2)
    else:
        print(f"  {code}: NO2 missing, SKIP")
        station_pm25_coverage[code] = 0.0
        continue
    
    weather_path = IN / f"WEATHER_{code}.csv"
    if weather_path.exists():
        weather = pd.read_csv(weather_path, parse_dates=["datetime"]).set_index("datetime")
        parts.append(weather)
    else:
        print(f"  {code}: WEATHER missing, SKIP")
        station_pm25_coverage[code] = 0.0
        continue
    
    combined = pd.concat(parts, axis=1)
    missing = [f for f in FEATURES_ORDER if f not in combined.columns]
    if missing:
        print(f"  {code}: features manquantes {missing}, SKIP")
        station_pm25_coverage[code] = 0.0
        continue
    
    combined = combined[FEATURES_ORDER]
    
    # Compter les valeurs aberrantes AVANT clipping
    n_negative_pm25 = (combined["PM2.5"] < 0).sum()
    n_negative_no2  = (combined["NO2"] < 0).sum()
    
    # Clipping (avant interpolation pour ne pas propager des valeurs aberrantes)
    combined = clip_outliers(combined, PHYSICAL_LIMITS)
    
    # Interpolation + resample
    combined = combined.interpolate(method="linear", limit=6).ffill().bfill()
    combined = combined.resample("1h").mean()
    combined = combined.dropna(subset=["PM2.5"])
    
    station_dfs[code] = combined
    aberrant_msg = ""
    if n_negative_pm25 > 0 or n_negative_no2 > 0:
        aberrant_msg = f" [clipped: {n_negative_pm25} neg PM2.5, {n_negative_no2} neg NO2]"
    print(f"  {code}: {len(combined):,} rows, PM2.5 raw cov {pm25_cov:.1%}{aberrant_msg}")

# Filtrer stations avec PM2.5 raw coverage < 50%
print(f"\n=== Filtering stations with PM2.5 raw coverage < 50% ===")
valid_codes = [c for c, cov in station_pm25_coverage.items() if cov >= 0.5]
dropped = [c for c, cov in station_pm25_coverage.items() if cov < 0.5]
if dropped:
    dropped_info = [(c, f'{station_pm25_coverage[c]:.1%}') for c in dropped]
    print(f"Dropped (low coverage): {dropped_info}")

# Filtre supplémentaire : exclure stations avec test set quasi-constant
# (capteur cassé qui renvoie valeur par défaut)
print(f"\n=== Filtering stations with constant test PM2.5 (std < 0.5) ===")
constant_codes = []
for code in list(valid_codes):
    df = station_dfs[code]
    T = len(df)
    t2 = int(0.85 * T)
    test_std = df["PM2.5"].iloc[t2:].std()
    if test_std < 0.5:
        constant_codes.append((code, test_std))
        valid_codes.remove(code)
if constant_codes:
    print(f"Dropped (constant test): {[(c, f'std={s:.2f}') for c, s in constant_codes]}")

print(f"Kept: {valid_codes} ({len(valid_codes)} stations)")

if len(valid_codes) < 5:
    raise SystemExit(f"ERREUR: trop peu de stations ({len(valid_codes)}).")

# Intersection temporelle
print(f"\n=== Aligning timestamps across stations ===")
common_idx = None
for code in valid_codes:
    idx = station_dfs[code].index
    common_idx = idx if common_idx is None else common_idx.intersection(idx)
print(f"Common timestamps: {len(common_idx):,}")

if len(common_idx) < 1000:
    raise SystemExit(f"ERREUR: trop peu de timestamps ({len(common_idx)}).")

# Save PM2.5 wide
pm25_wide = pd.DataFrame({
    code: station_dfs[code].loc[common_idx, "PM2.5"].values
    for code in valid_codes
}, index=common_idx)
pm25_wide.to_csv(OUT / "london_pm25_hourly.csv")
print(f"\nSaved: london_pm25_hourly.csv ({pm25_wide.shape})")

# Save parquet
combined_3d = []
for code in valid_codes:
    df = station_dfs[code].loc[common_idx, FEATURES_ORDER].copy()
    df["station"] = code
    combined_3d.append(df.reset_index())

full_df = pd.concat(combined_3d, ignore_index=True)
full_df.to_parquet(OUT / "london_full_hourly.parquet", index=False)
print(f"Saved: london_full_hourly.parquet ({full_df.shape})")

# Summary + verification clipping
print(f"\n=== SUMMARY ===")
print(f"Stations: {len(valid_codes)}")
print(f"Timestamps: {len(common_idx):,}")
print(f"Features per station: {FEATURES_ORDER}")
print(f"Date range: {common_idx.min()} -> {common_idx.max()}")
print(f"Final coverage: {full_df[FEATURES_ORDER].notna().mean().mean():.1%}")
print(f"\n=== POST-CLIP VALIDATION (PM2.5 par station) ===")
print(full_df.groupby('station')['PM2.5'].describe()[['mean','std','min','max']])

# Vérifier qu'il n'y a plus de valeurs négatives ou Inf
n_neg = (full_df['PM2.5'] < 0).sum()
n_inf = np.isinf(full_df[FEATURES_ORDER].values).sum()
n_nan = full_df[FEATURES_ORDER].isna().sum().sum()
print(f"\nNégatifs PM2.5 restants: {n_neg} (doit être 0)")
print(f"Inf restants: {n_inf} (doit être 0)")
print(f"NaN restants: {n_nan} (doit être 0)")
