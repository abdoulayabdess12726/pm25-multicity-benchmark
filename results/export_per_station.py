#!/usr/bin/env python3
"""
results/export_per_station.py
==========================================================================
Produit results/per_station_seed_topology.csv
    colonnes : city, station, seed, topology, r2_linear, r2_gcn, delta_r2, source

MODE seed 42 (par défaut) — 54 lignes = 27 stations x 2 topologies :
  * topology=distance    : lu depuis les per-station DÉJÀ PERSISTÉS dans
    results/{city}/multistation_results.json (chiffres exacts du papier).
  * topology=correlation : RÉ-ENTRAÎNÉ ici (Linear + GCN2, seed 42) en
    réutilisant EXACTEMENT les fonctions de 06_train_multistation.py
    (build_correlation_graph, split_and_scale, SpatioTemporalModel,
    train_model, evaluate) — car le per-station corrélation n'est persisté
    nulle part.

Vérifications imprimées :
  * seed 42 / distance     -> nb de delta_r2 < 0   (attendu 26/27)
  * seed 42 / 54 paires    -> nb de delta_r2 < 0   (attendu 53/54)

NB : les seeds 123/777 ne sont pas persistés côté per-station ; les inclure
exigerait un re-run complet (~2-3 h). Ce script couvre le seed primaire 42,
suffisant pour les deux vérifications du papier.
"""
import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
VARIANTS = [("Linear+Transformer", "linear"), ("GCN+Transformer", "gcn2")]


def load_bench():
    spec = importlib.util.spec_from_file_location(
        "bench", str(ROOT / "06_train_multistation.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = mod
    spec.loader.exec_module(mod)
    return mod


def rerun_correlation_perstation(b, data, device):
    """Ré-entraîne Linear+GCN2 (seed 42) sur la topologie CORRÉLATION et
    renvoie {model_name: {station: R2}}. Protocole identique à 06."""
    train_len = int(0.70 * len(data))
    ei, ew = b.build_correlation_graph(data[:train_len], k=b.K_NEIGHBORS)
    out = {}
    for model_name, enc_type in VARIANTS:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        train_d, val_d, test_d, scaler = b.split_and_scale(data)
        train_loader = b.DataLoader(b.MultiStationDataset(train_d),
                                    batch_size=b.BATCH_SIZE, shuffle=True)
        val_loader = b.DataLoader(b.MultiStationDataset(val_d),
                                  batch_size=b.BATCH_SIZE)
        test_loader = b.DataLoader(b.MultiStationDataset(test_d),
                                   batch_size=b.BATCH_SIZE)
        model = b.SpatioTemporalModel(
            in_features=len(b.FEATURES), d_model=b.D_MODEL,
            n_heads=b.N_HEADS, n_layers=b.N_LAYERS, dropout=b.DROPOUT,
            n_nodes=b.N_STATIONS, encoder_type=enc_type).to(device)
        model = b.train_model(model, train_loader, val_loader, ei, ew, device,
                              max_epochs=b.MAX_EPOCHS, patience=b.PATIENCE)
        _, _, _, per_st = b.evaluate(model, test_loader, ei, ew, scaler, device)
        out[model_name] = {s: per_st[s]["R2"] for s in per_st}
    return out


def main():
    b = load_bench()
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    loaders = {
        "beijing": lambda: b.load_beijing_data(
            str(ROOT / "data/beijing_real/PRSA_Data_20130301-20170228")),
        "london": b.load_london_data,
        "madrid": b.load_madrid_data,
    }

    rows = []
    log = io.StringIO()
    for city, loader in loaders.items():
        print(f"[{city}] chargement + re-run corrélation (seed {SEED})...",
              file=sys.stderr)
        with redirect_stdout(log):                    # loaders/graphes/train verbeux
            ret = loader()
            data = ret[0] if isinstance(ret, (tuple, list)) else ret
            data = np.asarray(data, dtype=np.float32)
            stations = list(b.STATION_NAMES)

            # --- distance : per-station persisté (chiffres du papier) ---
            j = json.load(open(ROOT / f"results/{city}/multistation_results.json"))
            ps = j["graphs"]["distance"]["per_station"]
            lin_d, gcn_d = ps["Linear+Transformer"], ps["GCN+Transformer"]

            # --- correlation : re-run ---
            corr = rerun_correlation_perstation(b, data, device)

        for s in stations:
            r2l, r2g = lin_d[s]["R2"], gcn_d[s]["R2"]
            rows.append(dict(city=city, station=s, seed=SEED, topology="distance",
                             r2_linear=r2l, r2_gcn=r2g, delta_r2=r2g - r2l,
                             source="persisted_json"))
        for s in stations:
            r2l, r2g = corr["Linear+Transformer"][s], corr["GCN+Transformer"][s]
            rows.append(dict(city=city, station=s, seed=SEED, topology="correlation",
                             r2_linear=r2l, r2_gcn=r2g, delta_r2=r2g - r2l,
                             source="rerun"))

    df = pd.DataFrame(rows, columns=["city", "station", "seed", "topology",
                                     "r2_linear", "r2_gcn", "delta_r2", "source"])
    out = ROOT / "results" / "per_station_seed_topology.csv"
    df.to_csv(out, index=False)

    # ── Rapport / vérifications ──
    print("=" * 78)
    print("  EXPORT PER-STATION — results/per_station_seed_topology.csv")
    print("=" * 78)
    print(f"  Lignes : {len(df)}  (attendu 54 = 27 stations x 2 topologies, seed {SEED})")
    dist = df[df.topology == "distance"]
    corr = df[df.topology == "correlation"]
    nd = int((dist.delta_r2 < 0).sum())
    nc = int((corr.delta_r2 < 0).sum())
    na = int((df.delta_r2 < 0).sum())
    print(f"\n  seed 42 / DISTANCE     : {nd}/{len(dist)} stations GCN<Linear   (attendu 26/27)")
    print(f"  seed 42 / CORRELATION  : {nc}/{len(corr)} stations GCN<Linear")
    print(f"  seed 42 / 54 PAIRES    : {na}/{len(df)} paires GCN<Linear         (attendu 53/54)")
    print("\n  Détail par ville/topologie (nb GCN<Linear / n_stations, ΔR² moyen) :")
    for city in ["beijing", "london", "madrid"]:
        for topo in ["distance", "correlation"]:
            sub = df[(df.city == city) & (df.topology == topo)]
            neg = int((sub.delta_r2 < 0).sum())
            print(f"    {city:8s} {topo:12s}: {neg}/{len(sub)}  ΔR²_moy={sub.delta_r2.mean():+.4f}")
    # station GCN>Linear (le/les positif(s))
    pos = df[df.delta_r2 >= 0]
    if len(pos):
        print("\n  Station(s) où GCN >= Linear :")
        for _, r in pos.iterrows():
            print(f"    {r.city} / {r.station} / {r.topology} : ΔR²={r.delta_r2:+.4f}")


if __name__ == "__main__":
    main()
