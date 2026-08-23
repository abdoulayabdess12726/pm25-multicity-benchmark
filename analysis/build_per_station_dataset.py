#!/usr/bin/env python3
"""
build_per_station_dataset.py — P9, base commune pour R2.1/R2.4/R2.7
==========================================================================
Reconstruit le jeu de données per-station (h_i, ΔR², métadonnées) pour les
4 réseaux (Beijing, London, Madrid, CZT), en repartant de raw_results.csv
(source unique, seed 42) plutôt que de l'ancien
results/per_station_seed_topology.csv (généré en Étape 2, antérieur à tous
les correctifs P5 — Madrid 6→7 stations, correctif build_correlation_graph,
CZT inexistant à l'époque).

h_i : calcul IDENTIQUE à 12_per_station_heterophily.py (voisinage top-k=5
par corrélation train, corrélations non-finies masquées avant le tri —
c'est le motif SÛR, celui-là même généralisé à build_correlation_graph lors
du correctif P5, cf. CHANGELOG_TABLES.md). Aucun entraînement : calcul pur
sur les séries PM2.5 déjà prétraitées (load_*_data), lecture de
raw_results.csv pour ΔR².

Sortie : analysis/per_station_dataset.csv
  (city, station, h_i, n_neighbors, delta_r2_distance, delta_r2_correlation)
"""
import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
K = 5
CITIES = ["beijing", "london", "madrid", "czt"]


def load_bench():
    spec = importlib.util.spec_from_file_location("bench", str(ROOT / "06_train_multistation.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = mod
    with redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def heterophily_for_city(b, city):
    """Identique à 12_per_station_heterophily.py::heterophily_for_city,
    étendu à CZT."""
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
        data = np.asarray(data, dtype=np.float32)

    names = list(b.STATION_NAMES)
    train_len = int(0.70 * len(data))
    feat_idx = b.FEATURES.index("PM2.5")
    pm25 = data[:train_len][:, :, feat_idx]

    corr = np.corrcoef(pm25.T)
    np.fill_diagonal(corr, -np.inf)
    k_eff = min(K, len(names) - 1)

    result = {}
    for i, name in enumerate(names):
        row = corr[i]
        valid = np.isfinite(row)
        if valid.sum() == 0:
            result[name] = (float("nan"), 0)
            continue
        masked = np.where(valid, row, -np.inf)
        neighbors = np.argsort(masked)[::-1][:min(k_eff, int(valid.sum()))]
        vals = row[neighbors]
        result[name] = (1.0 - float(vals.mean()), int(len(neighbors)))
    return result


def main():
    b = load_bench()

    het, ncount = {}, {}
    for city in CITIES:
        print(f"[{city}] calcul h_i (voisinage corrélation k={K}, train)...", file=sys.stderr)
        for name, (h_i, nn) in heterophily_for_city(b, city).items():
            het[(city, name)] = h_i
            ncount[(city, name)] = nn

    df = pd.read_csv(ROOT / "results" / "raw_results.csv", dtype=str)
    df["r2"] = pd.to_numeric(df["r2"], errors="coerce")
    df["variant"] = df["variant"].fillna("")
    df["k"] = pd.to_numeric(df["k"], errors="coerce")

    rows = []
    for city in CITIES:
        sub = df[(df.city == city) & (df.model.isin(["GCN-Transformer", "Linear-Transformer"]))
                & (df.variant == "") & (df.seed == "42") & (df.station != "__aggregate__")]
        # Linear-Transformer : indépendant de la topologie (dupliqué distance/correlation
        # dans raw_results.csv, cf. migrate_canonical — on prend n'importe laquelle).
        lin = sub[sub.model == "Linear-Transformer"].drop_duplicates(subset=["station"]).set_index("station")["r2"]
        for topo in ["distance", "correlation"]:
            gcn = sub[(sub.model == "GCN-Transformer") & (sub.topology == topo)
                     & (sub.k == 5)].set_index("station")["r2"]
            for station in gcn.index:
                if station not in lin.index:
                    continue
                key = (city, station)
                rows.append(dict(city=city, station=station, topology=topo,
                                 h_i=het.get(key, np.nan), n_neighbors=ncount.get(key, 0),
                                 delta_r2=float(gcn[station] - lin[station])))

    long_df = pd.DataFrame(rows)
    # merge plutôt que pivot_table : pivot_table avec dropna=False (nécessaire
    # pour ne pas perdre MENDEZ ALVARO, h_i indéfini) provoque une explosion
    # cartésienne sur un index multi-colonnes hétérogène entre villes (bug
    # pandas connu — 46 lignes attendues, 17672 obtenues, détecté avant ce
    # correctif) ; un merge explicite sur les clés (city, station) n'a pas ce
    # problème et gère nativement les h_i NaN sans les perdre.
    dist = long_df[long_df.topology == "distance"][["city", "station", "h_i", "n_neighbors", "delta_r2"]]
    dist = dist.rename(columns={"delta_r2": "delta_r2_distance"})
    corr = long_df[long_df.topology == "correlation"][["city", "station", "delta_r2"]]
    corr = corr.rename(columns={"delta_r2": "delta_r2_correlation"})
    wide = dist.merge(corr, on=["city", "station"], how="outer")
    wide = wide[["city", "station", "h_i", "n_neighbors", "delta_r2_distance", "delta_r2_correlation"]]
    wide = wide.sort_values(["city", "station"]).reset_index(drop=True)

    out_path = ROOT / "analysis" / "per_station_dataset.csv"
    wide.to_csv(out_path, index=False)
    print(f"\n{len(wide)} lignes -> {out_path}", file=sys.stderr)
    print(wide.groupby("city").size(), file=sys.stderr)
    n_nan = wide.h_i.isna().sum()
    if n_nan:
        print(f"\nh_i indéfini (corrélations train non-finies) : {n_nan} station(s) :", file=sys.stderr)
        print(wide[wide.h_i.isna()][["city", "station"]].to_string(index=False), file=sys.stderr)


if __name__ == "__main__":
    main()
