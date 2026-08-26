#!/usr/bin/env python3
"""
p9_3_characterization.py — P9.3 (R2.7) : table de caractérisation des 4 réseaux
==========================================================================
Aucun entraînement. Métriques calculées depuis les données déjà chargées
(load_*_data) et raw_results.csv (Persistence pour Beijing/London/Madrid ;
recalculée pour CZT — c'est une baseline sans apprentissage, y_t = y_{t-1},
pas un entraînement).

Taux de manquants AVANT interpolation : calculé sur les fichiers source
BRUTS (avant tout traitement de 01*.py), pas sur les parquets déjà
interpolés par load_*_data — sinon le taux serait trivialement ~0.

Densité de graphe réalisée / degré effectif : depuis build_graph (distance)
et build_correlation_graph (correlation, version corrigée P5), k=5.

Sortie : analysis/p9_3_characterization.csv + .md
"""
import glob
import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.regenerate_tables import load as load_resolved_raw_results  # noqa: E402

PERIODS = {
    "beijing": "2013-03-01 -> 2017-02-28 (48 mois)",
    "london": "2020-01-01 -> 2023-12-31 (48 mois)",
    "madrid": "2020-01-01 -> 2023-12-31 (48 mois)",
    "czt": "2020-01-01 -> 2023-12-31 (48 mois)",
}
PROVIDERS = {
    "beijing": "UCI Multi-Site Air-Quality Data Set (#501)",
    "london": "London Air Quality Network (LAQN)",
    "madrid": "OpenAQ API v3",
    "czt": "CNEMC (historique horaire, cf. 01h_download_czt.py)",
}
WEATHER_SOURCES = {
    "beijing": "native (dataset UCI, TEMP/PRES/DEWP/WSPM déjà inclus)",
    "london": "Open-Meteo Historical Weather",
    "madrid": "Open-Meteo Historical Weather",
    "czt": "Open-Meteo Historical Weather (3 points ville, centroïdes)",
}


def load_bench():
    spec = importlib.util.spec_from_file_location("bench", str(ROOT / "06_train_multistation.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = mod
    with redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def load_city(b, city):
    with redirect_stdout(io.StringIO()):
        if city == "beijing":
            ret = b.load_beijing_data(str(ROOT / "data/beijing_real/PRSA_Data_20130301-20170228"))
        elif city == "london":
            ret = b.load_london_data()
        elif city == "madrid":
            ret = b.load_madrid_data()
        else:
            ret = b.load_czt_data()
    data = ret[0] if isinstance(ret, (tuple, list)) else ret
    return np.asarray(data, dtype=np.float32)


def raw_missing_rate_beijing():
    files = glob.glob(str(ROOT / "data/beijing_real/PRSA_Data_20130301-20170228/*.csv"))
    rates = []
    for f in files:
        df = pd.read_csv(f)
        rates.append(df["PM2.5"].isna().mean())
    return float(np.mean(rates)), len(files)


def raw_missing_rate_london():
    files = glob.glob(str(ROOT / "data/london_laqn/LAQN_*_PM25.csv"))
    rates = []
    for f in files:
        df = pd.read_csv(f)
        val_col = df.columns[1]
        full_range = pd.date_range(pd.to_datetime(df.iloc[:, 0]).min(),
                                   pd.to_datetime(df.iloc[:, 0]).max(), freq="1h")
        n_present = df[val_col].notna().sum()
        rates.append(1 - n_present / len(full_range))
    return float(np.mean(rates)), len(files)


def raw_missing_rate_madrid():
    files = glob.glob(str(ROOT / "data/madrid_openaq/OPENAQ_*.csv"))
    files = [f for f in files if "WEATHER" not in f]
    rates = []
    for f in files:
        df = pd.read_csv(f)
        if "value" not in df.columns:
            continue
        dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce").dt.tz_localize(None)
        full_range = pd.date_range(dt.min(), dt.max(), freq="1h")
        n_present = df["value"].notna().sum()
        rates.append(1 - n_present / len(full_range))
    return float(np.mean(rates)), len(files)


def raw_missing_rate_czt():
    cov = pd.read_csv(ROOT / "data/czt_processed/station_coverage.csv")
    return float(1 - cov["raw_coverage"].mean()), len(cov)


RAW_MISSING = {
    "beijing": raw_missing_rate_beijing,
    "london": raw_missing_rate_london,
    "madrid": raw_missing_rate_madrid,
    "czt": raw_missing_rate_czt,
}


def persistence_r2_czt(data, pm_idx):
    """y_t = y_{t-1}, R² sur le test (15% finaux) — baseline sans apprentissage."""
    from sklearn.metrics import r2_score
    T = data.shape[0]
    t2 = int(0.85 * T)
    pm = data[:, :, pm_idx]
    y_true = pm[t2 + 1:].flatten()
    y_pred = pm[t2:-1].flatten()
    return r2_score(y_true, y_pred)


def main():
    b = load_bench()
    rows = []
    het_details = {}
    # scripts.regenerate_tables.load() — SOURCE UNIQUE. Une lecture
    # indépendante ici (pd.read_csv brut, .iloc[0] sur un résultat non
    # dédupliqué) a fait piocher R² persistance Madrid = 0.7986
    # (SUSPECT_6STATION, pré-correctif MENDEZ ALVARO) plutôt que 0.7961
    # (résolu) — valeur fausse déjà propagée dans Table 1 (caractérisation)
    # avant ce correctif (trouvé 2026-08-2x, cf. CHANGELOG_TABLES.md).
    df_rr = load_resolved_raw_results()

    for city in ["beijing", "london", "madrid", "czt"]:
        print(f"[{city}] chargement...", file=sys.stderr)
        data = load_city(b, city)
        names = list(b.STATION_NAMES)
        n = len(names)
        pm_idx = b.FEATURES.index("PM2.5")
        T = data.shape[0]
        train_len = int(0.70 * T)

        pm_full = data[:, :, pm_idx]
        pm_train = data[:train_len, :, pm_idx]

        pm_mean = float(np.nanmean(pm_full))
        pm_var = float(np.nanvar(pm_full, ddof=1))

        train_stds = np.nanstd(pm_train, axis=0, ddof=1)
        train_var_mean = float(np.nanmean(train_stds ** 2))
        train_var_min = float(np.nanmin(train_stds ** 2))
        train_var_max = float(np.nanmax(train_stds ** 2))

        # autocorrélation lag-1, moyenne par station
        lag1s = []
        for j in range(n):
            s = pm_full[:, j]
            if np.nanstd(s) < 1e-9:
                continue
            valid = ~np.isnan(s)
            s0, s1 = s[:-1][valid[:-1] & valid[1:]], s[1:][valid[:-1] & valid[1:]]
            if len(s0) > 2:
                lag1s.append(np.corrcoef(s0, s1)[0, 1])
        lag1_mean = float(np.mean(lag1s))

        # R² persistance
        if city == "czt":
            r2_persist = persistence_r2_czt(data, pm_idx)
        else:
            sub = df_rr[(df_rr.city == city) & (df_rr.model == "Persistence") & (df_rr.station == "__aggregate__")]
            r2_persist = float(sub.r2.iloc[0]) if len(sub) else np.nan

        # manquants bruts avant interpolation
        raw_miss, n_raw_files = RAW_MISSING[city]()

        # graphe : densité réalisée + degré effectif, distance et correlation, k=5
        b.N_STATIONS = n
        with redirect_stdout(io.StringIO()):
            ei_d, _ = b.build_graph(k=5)
        with redirect_stdout(io.StringIO()):
            data3d = pm_train[:, :, None]
            ei_c, _ = b.build_correlation_graph(data3d, k=5)
        max_edges = n * (n - 1)
        density_d = ei_d.shape[1] / max_edges
        density_c = ei_c.shape[1] / max_edges
        deg_eff_d = ei_d.shape[1] / n
        deg_eff_c = ei_c.shape[1] / n

        # r̄ inter-stations (train, toutes paires, comme 05_compute_heterogeneity_v2.py)
        corr = np.corrcoef(pm_train.T)
        iu = np.triu_indices_from(corr, k=1)
        r_bar = float(np.nanmean(corr[iu]))

        rows.append(dict(
            city=city, period=PERIODS[city], n_stations=n, provider=PROVIDERS[city],
            weather_source=WEATHER_SOURCES[city],
            pm25_mean=round(pm_mean, 2), pm25_var=round(pm_var, 2),
            train_var_mean=round(train_var_mean, 2), train_var_min=round(train_var_min, 2),
            train_var_max=round(train_var_max, 2), lag1_autocorr=round(lag1_mean, 4),
            r2_persistence=round(r2_persist, 4) if pd.notna(r2_persist) else None,
            raw_missing_rate=round(raw_miss, 4), n_raw_files_checked=n_raw_files,
            density_distance=round(density_d, 4), density_correlation=round(density_c, 4),
            degree_eff_distance=round(deg_eff_d, 2), degree_eff_correlation=round(deg_eff_c, 2),
            r_bar=round(r_bar, 4),
        ))
        print(f"  OK : n={n}, pm_mean={pm_mean:.2f}, r_bar={r_bar:.3f}, "
              f"train_var min/mean/max={train_var_min:.2f}/{train_var_mean:.2f}/{train_var_max:.2f}",
              file=sys.stderr)

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "analysis" / "p9_3_characterization.csv", index=False)

    md = ["# P9.3 (R2.7) — Table de caractérisation des 4 réseaux\n",
         "Aucun entraînement — métriques calculées depuis les données prétraitées "
         "(load_*_data) et raw_results.csv (Persistence Beijing/London/Madrid ; "
         "recalculée pour CZT, baseline sans apprentissage). Taux de manquants "
         "calculé sur les fichiers SOURCE bruts, avant toute interpolation.\n"]
    md.append("| " + " | ".join(out.columns) + " |")
    md.append("|" + "|".join(["---"] * len(out.columns)) + "|")
    for _, r in out.iterrows():
        md.append("| " + " | ".join(str(v) for v in r.values) + " |")
    (ROOT / "analysis" / "p9_3_characterization.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nSauvegardé : analysis/p9_3_characterization.csv + .md", file=sys.stderr)
    print(out.T.to_string())


if __name__ == "__main__":
    main()
