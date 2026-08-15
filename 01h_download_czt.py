"""
01h_download_czt.py — Collecte CNEMC brute, réseau Chang-Zhu-Tan (22 stations)
=============================================================================
Reconstruit depuis la source nationale ce que le .npz de MSDGNN (Lu et al.,
PLOS ONE 2025) fournit sans horodatage, avec 4 variables lissées (EWMA α=0.5)
et ~2 % d'heures supprimées non marquées. Voir PREREGISTRATION_CZT.md §5.

Source : https://quotsoft.net/air/data/china_sites_YYYYMMDD.csv
         (miroir public de la plateforme CNEMC, un fichier par jour,
          toutes stations nationales, résolution horaire)

Sortie  : data/czt_raw/daily/czt_YYYYMMDD.csv  (un fichier par jour, 22 colonnes)
          data/czt_raw/download_manifest.csv   (état par jour, reprise possible)

Reprise : relancer le script saute les jours déjà téléchargés.

Usage :
    python 01h_download_czt.py                    # 2020-01-01 -> 2023-12-31
    python 01h_download_czt.py --start 20200101 --end 20201231
"""

import argparse
import io
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Codes CNEMC identifiés par correspondance exacte des valeurs PM2.5 horaires
# du 2020-01-01 contre l'archive nationale (20/22 stations à 24/24 heures).
# Voir PREREGISTRATION_CZT.md §5.
CZT_CODES = [
    # Changsha
    "1335A", "1336A", "1337A", "1338A", "1339A",
    "1340A", "1341A", "1342A", "1343A", "1344A",
    # Zhuzhou / Xiangtan
    "1508A", "1511A", "1512A", "1513A", "1514A", "1515A",
    "1518A", "1519A", "1520A", "1524A",
    "1559A", "1562A",
]

# Ordre des variables du .npz MSDGNN, confirmé sur 1339A/2020-01-01.
# Les moyennes glissantes 24 h (_24h) et O3_8h sont écartées : dérivées.
TYPES = ["PM2.5", "PM10", "SO2", "NO2", "O3", "CO", "AQI"]

URL = "https://quotsoft.net/air/data/china_sites_{ymd}.csv"
OUT = Path("data/czt_raw")
DAILY = OUT / "daily"
MANIFEST = OUT / "download_manifest.csv"

SLEEP = 0.4        # politesse entre requêtes (s)
RETRIES = 3
TIMEOUT = 60


def fetch_day(ymd: str) -> pd.DataFrame | None:
    """Télécharge une journée nationale, renvoie les 22 colonnes CZT en format long.

    Renvoie None si le fichier est absent côté serveur (404) — certains jours
    manquent dans l'archive, c'est une donnée en soi (voir manifest).
    """
    url = URL.format(ymd=ymd)
    last_err = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research/pm25-benchmark"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                raw = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last_err = e
            time.sleep(2 * (attempt + 1))
        except Exception as e:                      # noqa: BLE001 — réseau
            last_err = e
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"{ymd}: {last_err}")

    df = pd.read_csv(io.BytesIO(raw))
    df = df[df["type"].isin(TYPES)]

    present = [c for c in CZT_CODES if c in df.columns]
    missing = [c for c in CZT_CODES if c not in df.columns]

    keep = ["date", "hour", "type"] + present
    out = df[keep].copy()
    for c in missing:                                # colonne absente ce jour-là
        out[c] = pd.NA
    return out[["date", "hour", "type"] + CZT_CODES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20200101")
    ap.add_argument("--end", default="20231231")
    args = ap.parse_args()

    DAILY.mkdir(parents=True, exist_ok=True)

    d0 = date(int(args.start[:4]), int(args.start[4:6]), int(args.start[6:]))
    d1 = date(int(args.end[:4]), int(args.end[4:6]), int(args.end[6:]))
    days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]

    print(f"Chang-Zhu-Tan : {len(CZT_CODES)} stations, {len(days)} jours "
          f"({d0} -> {d1})", flush=True)

    rows, n_new, n_skip, n_404 = [], 0, 0, 0
    t_start = time.time()

    for i, d in enumerate(days):
        ymd = d.strftime("%Y%m%d")
        dst = DAILY / f"czt_{ymd}.csv"

        if dst.exists():
            n_skip += 1
            rows.append({"date": ymd, "status": "cached", "n_hours": pd.NA})
            continue

        day = fetch_day(ymd)
        if day is None:
            n_404 += 1
            rows.append({"date": ymd, "status": "absent_404", "n_hours": 0})
            print(f"  {ymd}: absent de l'archive (404)", flush=True)
        else:
            day.to_csv(dst, index=False)
            n_new += 1
            n_hours = day["hour"].nunique()
            rows.append({"date": ymd, "status": "ok", "n_hours": n_hours})
            if n_hours < 24:
                print(f"  {ymd}: {n_hours}/24 heures disponibles", flush=True)

        time.sleep(SLEEP)

        if (i + 1) % 100 == 0:
            el = time.time() - t_start
            print(f"[{i+1}/{len(days)}] {ymd} — {n_new} téléchargés, "
                  f"{n_skip} en cache, {n_404} absents — {el/60:.1f} min",
                  flush=True)

    pd.DataFrame(rows).to_csv(MANIFEST, index=False)

    print(f"\nTerminé en {(time.time()-t_start)/60:.1f} min : "
          f"{n_new} nouveaux, {n_skip} en cache, {n_404} absents", flush=True)
    print(f"Manifeste : {MANIFEST}", flush=True)
    print(f"Fichiers  : {DAILY}/czt_YYYYMMDD.csv", flush=True)
    print("\nÉtape suivante : 01i_preprocess_czt.py "
          "(filtre §3.2 couverture 50 %, jonction Open-Meteo)", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
