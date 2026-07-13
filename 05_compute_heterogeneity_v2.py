"""Compute composite spatial heterogeneity index h(D) for multiple cities.
v2: CORRECT CV computation (was buggy: z-score forced CV=1.0 by construction).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from libpysal.weights import KNN
from esda.moran import Moran

BEIJING_COORDS = {
    "Aotizhongxin":  (39.982, 116.397),
    "Changping":     (40.218, 116.231),
    "Dingling":      (40.292, 116.220),
    "Dongsi":        (39.929, 116.417),
    "Guanyuan":      (39.929, 116.339),
    "Gucheng":       (39.914, 116.184),
    "Huairou":       (40.328, 116.628),
    "Nongzhanguan":  (39.937, 116.461),
    "Shunyi":        (40.127, 116.655),
    "Tiantan":       (39.886, 116.407),
    "Wanliu":        (39.987, 116.287),
    "Wanshouxigong": (39.878, 116.352),
}

def heterogeneity(df_wide, coords_df, label=""):
    """df_wide: index=datetime, columns=stations
       coords_df: columns=[station, lat, lon]"""
    
    # 1. Mean inter-station Pearson correlation
    corr = df_wide.corr().values
    iu = np.triu_indices_from(corr, k=1)
    r_bar = np.nanmean(corr[iu])

    # 2. Moran's I on time-averaged station values
    means = df_wide.mean(axis=0)
    coords_df = coords_df.set_index("station").reindex(means.index).dropna()
    means_aligned = means.loc[coords_df.index]
    pts = coords_df[["lat", "lon"]].values
    w = KNN.from_array(pts, k=min(8, len(pts) - 1))
    w.transform = "r"
    moran_I = Moran(means_aligned.values, w).I

    # 3. CORRECT Coefficient of Variation: std/mean PER STATION on raw values
    # (previous bug: z-scoring before CV computation forced CV=1.0)
    station_means = df_wide.mean(axis=0)
    station_stds  = df_wide.std(axis=0)
    # Avoid division by zero/near-zero means
    valid_mask = np.abs(station_means) > 1e-3
    cv_per_station = (station_stds[valid_mask] / station_means[valid_mask]).abs()
    cv = cv_per_station.mean()

    # Composite : need to normalize CV to [0, 1] range for combination
    # Empirical: typical CV for air quality stations ranges 0.3 to 1.5
    # We cap at 2.0 for normalization
    cv_normalized = min(cv / 2.0, 1.0)
    
    h = ((1 - r_bar) + (1 - moran_I) + cv_normalized) / 3

    print(f"\n=== {label} ===")
    print(f"  Stations: {len(df_wide.columns)}")
    print(f"  r_bar          = {r_bar:.3f}  (lower => more heterogeneous)")
    print(f"  Moran I        = {moran_I:.3f}  (lower => less spatial clustering)")
    print(f"  CV (raw)       = {cv:.3f}  (higher => more inter-station variability)")
    print(f"  CV (normalized)= {cv_normalized:.3f}")
    print(f"  h(D)           = {h:.3f}  (higher => more heterogeneous)")
    
    return {"city": label, "n_stations": len(df_wide.columns),
            "r_bar": r_bar, "moran_I": moran_I, 
            "cv_raw": cv, "cv_normalized": cv_normalized, "h": h}

# === BEIJING ===
print("Loading Beijing data...")
beijing_dir = Path("data/beijing_real/PRSA_Data_20130301-20170228")
csv_files = sorted(beijing_dir.glob("PRSA_Data_*.csv"))
print(f"Found {len(csv_files)} Beijing station CSVs")

if len(csv_files) == 0:
    print("ERROR: No Beijing per-station CSVs found.")
    raise SystemExit(1)

b_frames = []
for csv in csv_files:
    name = csv.stem.split("_")[2]
    df = pd.read_csv(csv)
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.set_index("datetime")[["PM2.5"]].rename(columns={"PM2.5": name})
    b_frames.append(df)

beijing_wide = pd.concat(b_frames, axis=1).interpolate(limit=3)
beijing_wide = beijing_wide.dropna(thresh=int(0.7 * len(beijing_wide.columns)))
beijing_coords = pd.DataFrame([
    {"station": k, "lat": v[0], "lon": v[1]} for k, v in BEIJING_COORDS.items()
])

h_beijing = heterogeneity(beijing_wide, beijing_coords, label="Beijing")

# === LONDON ===
print("\nLoading London data...")
london_wide = pd.read_csv("data/london_processed/london_pm25_hourly.csv",
                          index_col=0, parse_dates=True)
london_coords = pd.read_csv("data/london_laqn/station_coords.csv")
h_london = heterogeneity(london_wide, london_coords, label="London")

# === MADRID (placeholder if data exists) ===
madrid_csv = Path("data/madrid_processed/madrid_pm25_hourly.csv")
madrid_coords_csv = Path("data/madrid_openaq/station_coords.csv")
h_madrid = None
if madrid_csv.exists() and madrid_coords_csv.exists():
    print("\nLoading Madrid data...")
    madrid_wide = pd.read_csv(madrid_csv, index_col=0, parse_dates=True)
    madrid_coords = pd.read_csv(madrid_coords_csv)
    h_madrid = heterogeneity(madrid_wide, madrid_coords, label="Madrid")
else:
    print("\n(Madrid data not found, skipping)")

# === COMPARAISON ===
results_list = [h_beijing, h_london]
if h_madrid:
    results_list.append(h_madrid)

print(f"\n{'='*60}")
print(f"SUMMARY OF HETEROGENEITY INDICES")
print(f"{'='*60}")
print(f"{'City':<10} {'r_bar':>8} {'Moran I':>10} {'CV':>8} {'h(D)':>8}")
for r in results_list:
    print(f"{r['city']:<10} {r['r_bar']:>8.3f} {r['moran_I']:>10.3f} "
          f"{r['cv_raw']:>8.3f} {r['h']:>8.3f}")

if len(results_list) >= 2:
    gap = results_list[-1]["h"] - results_list[0]["h"]
    print(f"\nGap range = {gap:+.3f}")

# Save
Path("results").mkdir(exist_ok=True)
results = pd.DataFrame(results_list)
results.to_csv("results/heterogeneity_index_v2.csv", index=False)
print(f"\nSaved to results/heterogeneity_index_v2.csv")
