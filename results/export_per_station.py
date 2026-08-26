#!/usr/bin/env python3
"""
results/export_per_station.py
==========================================================================
Produit results/per_station_seed_topology.csv
    colonnes : city, station, seed, topology, r2_linear, r2_gcn, delta_r2

Source : scripts.regenerate_tables.load() — vue résolue de raw_results.csv
(GCN-Transformer k=5, Linear-Transformer, seeds 42/123/777, par station).
Aucun ré-entraînement ici : pur export des résultats persistés.

Auparavant, ce script lisait results/{city}/multistation_results.json
directement (clé graphs[topology]["per_station_all_seeds"]) — une SECONDE
source parallèle à raw_results.csv, jamais garantie synchronisée avec elle.
Le fichier produit (per_station_seed_topology.csv) était resté figé au
18 juillet, avant tous les correctifs 2026-08 (MENDEZ ALVARO, tri NaN
build_correlation_graph). Même famille de risque que les 3 bugs déjà
trouvés (Figure 3, P9.2, P9.3 caractérisation) — signalé par l'utilisateur,
corrigé ici en convergeant sur la source unique raw_results.csv via
load() plutôt que de maintenir un second chemin de lecture JSON.

Attendu : 162 lignes = 27 stations x 3 seeds x 2 topologies.

Vérifications imprimées :
  * seed 42 / distance  -> nb de delta_r2 < 0   (attendu 26/27)
  * seed 42 / 54 paires -> nb de delta_r2 < 0   (attendu 53/54)
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.regenerate_tables import load as load_resolved_raw_results  # noqa: E402

CITIES = ["beijing", "london", "madrid"]
TOPOLOGIES = ["distance", "correlation"]
SEEDS = ["42", "123", "777"]


def main():
    df = load_resolved_raw_results()
    df = df[(df.city.isin(CITIES)) & (df.station != "__aggregate__")
            & (df.seed.isin(SEEDS))]

    gcn = df[(df.model == "GCN-Transformer") & (df.k == 5) & (df.variant == "")]
    lin = df[(df.model == "Linear-Transformer") & (df.variant == "")]

    rows = []
    for city in CITIES:
        for topo in TOPOLOGIES:
            g = gcn[(gcn.city == city) & (gcn.topology == topo)].set_index(["seed", "station"])["r2"]
            l = lin[(lin.city == city) & (lin.topology == topo)].set_index(["seed", "station"])["r2"]
            common = g.index.intersection(l.index)
            for seed, station in sorted(common):
                r2g, r2l = float(g.loc[(seed, station)]), float(l.loc[(seed, station)])
                rows.append(dict(city=city, station=station, seed=int(seed),
                                 topology=topo, r2_linear=r2l, r2_gcn=r2g,
                                 delta_r2=r2g - r2l))

    df_out = pd.DataFrame(rows, columns=["city", "station", "seed", "topology",
                                         "r2_linear", "r2_gcn", "delta_r2"])
    out = ROOT / "results" / "per_station_seed_topology.csv"
    df_out.to_csv(out, index=False)

    # ── Rapport / vérifications ──
    print("=" * 78)
    print("  EXPORT PER-STATION — results/per_station_seed_topology.csv")
    print("=" * 78)
    print(f"  Lignes : {len(df_out)}   (attendu 162 = 27 stations x 3 seeds x 2 topologies)")
    print(f"  Seeds  : {sorted(df_out.seed.unique())} | topologies : {sorted(df_out.topology.unique())}")
    print(f"  Stations par ville : "
          + ", ".join(f"{c}={df_out[df_out.city == c].station.nunique()}" for c in CITIES))

    s42 = df_out[df_out.seed == 42]
    d42 = s42[s42.topology == "distance"]
    nd = int((d42.delta_r2 < 0).sum())
    na = int((s42.delta_r2 < 0).sum())
    print(f"\n  seed 42 / DISTANCE  : {nd}/{len(d42)} stations GCN<Linear   (attendu 26/27)")
    print(f"  seed 42 / 54 PAIRES : {na}/{len(s42)} paires GCN<Linear       (attendu 53/54)")

    print("\n  ΔR² moyen par ville/topologie/seed :")
    for city in CITIES:
        for topo in TOPOLOGIES:
            sub = df_out[(df_out.city == city) & (df_out.topology == topo)]
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
