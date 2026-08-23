"""01i_preprocess_czt.py — E16 (P5) : prétraitement Chang-Zhu-Tan
==========================================================================
Calqué sur 01c_preprocess_london.py / 01g_preprocess_madrid.py : PM2.5 brut
+ jonction météo, filtre §3.2 sur couverture NON imputée (seuil réel 50 %,
comme Londres — pas le 30 % exceptionnel de Madrid, motivé par la faible
densité OpenAQ, non pertinente ici), sortie au format harmonisé
['PM2.5','TEMP','PRES','DEWP','WSPM'].

Différence avec Beijing/London/Madrid : la météo Open-Meteo est récupérée
UNE FOIS PAR VILLE (Changsha/Zhuzhou/Xiangtan — 3 requêtes), pas par
station (20), chaque station étant rattachée au point météo de sa ville
(champ `area` de configs/stations/czt.yaml, lui-même issu de l'API CNEMC
temps réel, cf. commit czt.yaml). Le point météo par ville est le centroïde
(moyenne lat/lon) des stations qui lui sont rattachées — aucune coordonnée
"centre-ville" officielle utilisée, choix documenté ici. Repli dans un
script séparé jugé disproportionné vu le volume (3 requêtes vs les dizaines
par station pour les autres villes) — fondu dans ce script.

Source PM2.5 : data/czt_raw/daily/czt_YYYYMMDD.csv (CNEMC historique
2020-2023, 01h_download_czt.py, déjà téléchargé). Lignes type=="PM2.5"
uniquement.

Sortie : data/czt_processed/czt_pm25_hourly.csv, czt_full_hourly.parquet,
station_coverage.csv (couverture par station, diagnostic variance train/test).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "czt_raw" / "daily"
OUT = ROOT / "data" / "czt_processed"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES_ORDER = ["PM2.5", "TEMP", "PRES", "DEWP", "WSPM"]
COVERAGE_THRESHOLD = 0.50          # §3.2 réel (identique à Londres, pas l'exception Madrid 30%)
CONST_TEST_STD_THRESHOLD = 0.5     # même seuil que Londres/Madrid (station-level, pas train)
START, END = "2020-01-01", "2023-12-31"
HOURLY_VARS = ["temperature_2m", "surface_pressure", "dew_point_2m", "wind_speed_10m"]


def load_station_config():
    with open(ROOT / "configs" / "stations" / "czt.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["stations"]


def city_centroids(stations):
    by_area = {}
    for s in stations:
        by_area.setdefault(s["area"], []).append((s["lat"], s["lon"]))
    centroids = {area: (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts])))
                for area, pts in by_area.items()}
    return centroids


def fetch_weather(area, lat, lon):
    cache = OUT / f"WEATHER_{area}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["datetime"]).set_index("datetime")
    print(f"  météo {area} ({lat:.4f}, {lon:.4f})...", end=" ", flush=True)
    params = dict(latitude=lat, longitude=lon, start_date=START, end_date=END,
                 hourly=",".join(HOURLY_VARS), timezone="UTC")
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=120)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "TEMP": hourly["temperature_2m"],
        "PRES": hourly["surface_pressure"],
        "DEWP": hourly["dew_point_2m"],
        "WSPM": hourly["wind_speed_10m"],
    }).set_index("datetime")
    df.to_csv(cache)
    print(f"OK ({len(df)} lignes)")
    time.sleep(1)  # courtoisie API, 3 requêtes seulement
    return df


def load_pm25_wide():
    files = sorted(RAW_DIR.glob("czt_*.csv"))
    print(f"{len(files)} fichiers journaliers CNEMC à charger...")
    frames = []
    for i, fp in enumerate(files):
        df = pd.read_csv(fp)
        df = df[df["type"] == "PM2.5"].copy()
        df["datetime"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d") + pd.to_timedelta(df["hour"], unit="h")
        frames.append(df.drop(columns=["date", "hour", "type"]).set_index("datetime"))
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(files)}...")
    pm25 = pd.concat(frames).sort_index()
    pm25 = pm25[~pm25.index.duplicated(keep="first")]
    return pm25


def main():
    stations = load_station_config()
    codes = [s["name"] for s in stations]
    area_of = {s["name"]: s["area"] for s in stations}
    centroids = city_centroids(stations)
    print(f"{len(codes)} stations, {len(centroids)} villes : {list(centroids.keys())}")

    print("\n=== Météo Open-Meteo (3 points ville, centroïdes) ===")
    weather_by_area = {area: fetch_weather(area, lat, lon) for area, (lat, lon) in centroids.items()}

    print("\n=== Chargement PM2.5 CNEMC 2020-2023 (brut, non imputé) ===")
    pm25_wide = load_pm25_wide()
    print(f"PM2.5 brut : {pm25_wide.shape}, période {pm25_wide.index.min()} -> {pm25_wide.index.max()}")

    station_dfs = {}
    coverage_rows = []
    print("\n=== Jonction météo + calcul de couverture brute (avant tout filtre/imputation) ===")
    for code in codes:
        if code not in pm25_wide.columns:
            print(f"  {code}: absente du CNEMC brut, SKIP")
            coverage_rows.append(dict(station=code, area=area_of[code], raw_coverage=0.0,
                                      kept=False, reason="absente_source"))
            continue
        pm = pm25_wide[[code]].rename(columns={code: "PM2.5"})
        raw_cov = pm["PM2.5"].notna().mean()
        weather = weather_by_area[area_of[code]]
        combined = pd.concat([pm, weather], axis=1)[FEATURES_ORDER]
        station_dfs[code] = combined
        coverage_rows.append(dict(station=code, area=area_of[code], raw_coverage=raw_cov,
                                  kept=None, reason=""))
        print(f"  {code} ({area_of[code]}): couverture brute PM2.5 = {raw_cov:.1%}")

    print(f"\n=== Filtre §3.2 : couverture PM2.5 brute >= {COVERAGE_THRESHOLD:.0%} (sur données NON imputées) ===")
    cov_map = {r["station"]: r["raw_coverage"] for r in coverage_rows}
    valid_codes = [c for c in station_dfs if cov_map[c] >= COVERAGE_THRESHOLD]
    dropped_cov = [c for c in station_dfs if cov_map[c] < COVERAGE_THRESHOLD]
    if dropped_cov:
        print(f"  Écartées (couverture < {COVERAGE_THRESHOLD:.0%}) : "
              f"{[(c, f'{cov_map[c]:.1%}') for c in dropped_cov]}")
        for r in coverage_rows:
            if r["station"] in dropped_cov:
                r["kept"], r["reason"] = False, "couverture_insuffisante"

    print(f"\n=== Filtre : test PM2.5 quasi-constant (std < {CONST_TEST_STD_THRESHOLD}) ===")
    print("=== Diagnostic : mode variance-train-faible / variance-test-normale (cf. MENDEZ ALVARO, Madrid P1) ===")
    dropped_const = []
    train_test_notes = {}
    for code in list(valid_codes):
        df = station_dfs[code].interpolate(method="linear", limit=6).ffill().bfill()
        df = df.dropna(subset=["PM2.5"])
        T = len(df)
        if T < 5000:
            dropped_const.append((code, f"T={T}_trop_court"))
            valid_codes.remove(code)
            continue
        t1, t2 = int(0.70 * T), int(0.85 * T)
        train_std = df["PM2.5"].iloc[:t1].std()
        test_std = df["PM2.5"].iloc[t2:].std()
        train_test_notes[code] = (train_std, test_std)
        if test_std < CONST_TEST_STD_THRESHOLD:
            dropped_const.append((code, f"test_std={test_std:.3f}"))
            valid_codes.remove(code)
        elif train_std < 0.5:
            print(f"  ATTENTION {code} : train_std={train_std:.3f} (faible) mais test_std={test_std:.3f} "
                  f"(normal) — mode MENDEZ ALVARO potentiel, à vérifier avant tout entraînement")
    if dropped_const:
        print(f"  Écartées (test quasi-constant ou trop court) : {dropped_const}")
        for r in coverage_rows:
            if r["station"] in [c for c, _ in dropped_const]:
                r["kept"], r["reason"] = False, "test_constant_ou_court"

    for r in coverage_rows:
        if r["kept"] is None:
            r["kept"] = r["station"] in valid_codes
            if r["kept"]:
                r["reason"] = "ok"

    print(f"\n=== RÉSULTAT : {len(valid_codes)}/{len(codes)} stations retenues ===")
    print(valid_codes)
    if len(valid_codes) < 5:
        raise SystemExit(f"ERREUR : trop peu de stations ({len(valid_codes)}).")

    common_idx = None
    for code in valid_codes:
        idx = station_dfs[code].dropna(subset=["PM2.5"]).index
        common_idx = idx if common_idx is None else common_idx.union(idx)
    # fenêtre horaire complète sur la période commune (comme Beijing/London/Madrid : resample 1h)
    full_range = pd.date_range(pm25_wide.index.min(), pm25_wide.index.max(), freq="1h")

    combined_3d = []
    for code in valid_codes:
        df = station_dfs[code].reindex(full_range)
        df = df.interpolate(method="linear", limit=6).ffill().bfill()
        df["station"] = code
        combined_3d.append(df.reset_index().rename(columns={"index": "datetime"}))

    full_df = pd.concat(combined_3d, ignore_index=True)
    full_df.to_parquet(OUT / "czt_full_hourly.parquet", index=False)
    print(f"\nSauvegardé : czt_full_hourly.parquet {full_df.shape}")

    pm25_out = pd.DataFrame({
        code: full_df[full_df.station == code].set_index("datetime")["PM2.5"].values
        for code in valid_codes
    }, index=full_range)
    pm25_out.to_csv(OUT / "czt_pm25_hourly.csv")
    print(f"Sauvegardé : czt_pm25_hourly.csv {pm25_out.shape}")

    cov_df = pd.DataFrame(coverage_rows)
    cov_df["train_std"] = cov_df["station"].map(lambda c: train_test_notes.get(c, (None, None))[0])
    cov_df["test_std"] = cov_df["station"].map(lambda c: train_test_notes.get(c, (None, None))[1])
    cov_df.to_csv(OUT / "station_coverage.csv", index=False)
    print(f"Sauvegardé : station_coverage.csv ({len(cov_df)} lignes)")

    valid_coords = pd.DataFrame([
        dict(station=s["name"], lat=s["lat"], lon=s["lon"], area=s["area"])
        for s in stations if s["name"] in valid_codes
    ])
    valid_coords.to_csv(OUT / "station_coords_valid.csv", index=False)

    print("\n=== RÉCAPITULATIF COUVERTURE (toutes stations) ===")
    print(cov_df[["station", "area", "raw_coverage", "kept", "reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
