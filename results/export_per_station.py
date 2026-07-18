#!/usr/bin/env python3
"""
results/export_per_station.py
==========================================================================
Produit results/per_station_seed_topology.csv
    colonnes : city, station, seed, topology, r2_linear, r2_gcn, delta_r2

Source : results/{city}/multistation_results.json, clé
    graphs[topology]["per_station_all_seeds"][model][seed][station]["R2"]
peuplée par 06_train_multistation.py (--graph both, seeds 42/123/777).
Aucun ré-entraînement ici : pur export des résultats persistés.

Attendu : 162 lignes = 27 stations x 3 seeds x 2 topologies.

Vérifications imprimées :
  * seed 42 / distance  -> nb de delta_r2 < 0   (attendu 26/27)
  * seed 42 / 54 paires -> nb de delta_r2 < 0   (attendu 53/54)
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CITIES = ["beijing", "london", "madrid"]
TOPOLOGIES = ["distance", "correlation"]
GCN, LIN = "GCN+Transformer", "Linear+Transformer"


def main():
    rows = []
    for city in CITIES:
        path = ROOT / f"results/{city}/multistation_results.json"
        if not path.exists():
            sys.exit(f"MANQUANT : {path} — lancer 06_train_multistation.py --city {city} --graph both")
        d = json.loads(path.read_text())
        for topo in TOPOLOGIES:
            if topo not in d["graphs"]:
                sys.exit(f"{city}: topologie '{topo}' absente — relancer 06 avec --graph both")
            psa = d["graphs"][topo].get("per_station_all_seeds")
            if not psa:
                sys.exit(f"{city}/{topo}: 'per_station_all_seeds' absent — relancer 06 (version à jour)")
            for seed in sorted(psa[LIN].keys(), key=int):
                lin_st, gcn_st = psa[LIN][seed], psa[GCN][seed]
                for station in lin_st:
                    r2l = lin_st[station]["R2"]
                    r2g = gcn_st[station]["R2"]
                    rows.append(dict(city=city, station=station, seed=int(seed),
                                     topology=topo, r2_linear=r2l, r2_gcn=r2g,
                                     delta_r2=r2g - r2l))

    df = pd.DataFrame(rows, columns=["city", "station", "seed", "topology",
                                     "r2_linear", "r2_gcn", "delta_r2"])
    out = ROOT / "results" / "per_station_seed_topology.csv"
    df.to_csv(out, index=False)

    # ── Rapport / vérifications ──
    print("=" * 78)
    print("  EXPORT PER-STATION — results/per_station_seed_topology.csv")
    print("=" * 78)
    print(f"  Lignes : {len(df)}   (attendu 162 = 27 stations x 3 seeds x 2 topologies)")
    print(f"  Seeds  : {sorted(df.seed.unique())} | topologies : {sorted(df.topology.unique())}")
    print(f"  Stations par ville : "
          + ", ".join(f"{c}={df[df.city == c].station.nunique()}" for c in CITIES))

    s42 = df[df.seed == 42]
    d42 = s42[s42.topology == "distance"]
    nd = int((d42.delta_r2 < 0).sum())
    na = int((s42.delta_r2 < 0).sum())
    print(f"\n  seed 42 / DISTANCE  : {nd}/{len(d42)} stations GCN<Linear   (attendu 26/27)")
    print(f"  seed 42 / 54 PAIRES : {na}/{len(s42)} paires GCN<Linear       (attendu 53/54)")

    print("\n  ΔR² moyen par ville/topologie/seed :")
    for city in CITIES:
        for topo in TOPOLOGIES:
            sub = df[(df.city == city) & (df.topology == topo)]
            per_seed = "  ".join(
                f"s{s}={sub[sub.seed == s].delta_r2.mean():+.4f}"
                for s in sorted(sub.seed.unique()))
            neg = int((sub[sub.seed == 42].delta_r2 < 0).sum())
            n = sub[sub.seed == 42].shape[0]
            print(f"    {city:8s} {topo:12s}: {per_seed}   | seed42 GCN<Lin {neg}/{n}")

    pos = s42[s42.delta_r2 >= 0]
    if len(pos):
        print("\n  Station(s) seed 42 où GCN >= Linear :")
        for _, r in pos.iterrows():
            print(f"    {r.city} / {r.station} / {r.topology} : ΔR²={r.delta_r2:+.4f}")


if __name__ == "__main__":
    main()
