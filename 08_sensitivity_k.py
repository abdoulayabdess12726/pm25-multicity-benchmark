#!/usr/bin/env python3
"""
08_sensitivity_k.py  —  Étape 6 (E3) : k-sensitivity CANONIQUE
==========================================================================
Sensibilité à la densité du graphe (k voisins) sous le PROTOCOLE COMPLET de 06
(3 seeds 42/123/777, entraînement complet MAX_EPOCHS=50 / PATIENCE=8, modèle
plein D_MODEL=64 — AUCUN --quick / schedule réduit). k ∈ {3,5,8} plafonné à N−1.

ΔR² = R²(GCN-Transformer) − R²(Linear-Transformer), agrégat global (comme 06),
moyenne ± SD sur les 3 seeds, format Table 6.

Réutilise les résultats CANONIQUES déjà persistés dans
results/{city}/multistation_results.json :
  - Linear-Transformer (indépendant de k et de la topologie),
  - GCN-Transformer à k=5 (le benchmark canonique).
Recalcule uniquement le GCN à k=3 et k=8 (deux topologies, 3 seeds).

NB : agrégat sur TOUTES les stations (7 pour Madrid), cohérent avec le k=5
canonique et l'ancienne Table 6 (l'exclusion MENDEZ ALVARO de E1/Exp. A ne
s'applique pas ici, pour garder Table 6 comparable en interne).

Sortie : results/sensitivity_k_canonical.csv
Usage : python 08_sensitivity_k.py --cities beijing [--k 3 8]
"""
import argparse
import gc
import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
import pandas as pd


def _free_mps():
    """Libère la mémoire MPS entre entraînements (sinon accumulation -> swap)."""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

ROOT = Path(__file__).resolve().parent
KS = [3, 5, 8]
SEEDS = [42, 123, 777]
TOPOS = ["distance", "correlation"]
CSV = ROOT / "results" / "sensitivity_k_canonical.csv"


def load_bench():
    spec = importlib.util.spec_from_file_location(
        "bench", str(ROOT / "06_train_multistation.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = mod
    spec.loader.exec_module(mod)
    return mod


def get_city(b, city):
    with redirect_stdout(io.StringIO()):
        if city == "beijing":
            ret = b.load_beijing_data(
                str(ROOT / "data/beijing_real/PRSA_Data_20130301-20170228"))
        elif city == "london":
            ret = b.load_london_data()
        else:
            ret = b.load_madrid_data()
        data = ret[0] if isinstance(ret, (tuple, list)) else ret
        data = np.asarray(data, dtype=np.float32)
        train_d, val_d, test_d, scaler = b.split_and_scale(data)
    return dict(city=city, data=data, n_nodes=len(b.STATION_NAMES),
                train_d=train_d, val_d=val_d, test_d=test_d, scaler=scaler)


def build_graph_k(b, c, topo, k):
    """Graphe DISTANCE ou CORRÉLATION à k voisins (mêmes fonctions que 06)."""
    with redirect_stdout(io.StringIO()):
        if topo == "distance":
            ei, ew = b.build_graph(k=k)
        else:
            train_len = int(0.70 * len(c["data"]))
            ei, ew = b.build_correlation_graph(c["data"][:train_len], k=k)
    return ei, ew


def train_gcn_r2(b, c, ei, ew, seed, device):
    """Entraîne le GCN-Transformer (gcn2) au protocole complet -> R² agrégé."""
    torch.manual_seed(seed); np.random.seed(seed)
    tl = b.DataLoader(b.MultiStationDataset(c["train_d"]),
                      batch_size=b.BATCH_SIZE, shuffle=True)
    vl = b.DataLoader(b.MultiStationDataset(c["val_d"]), batch_size=b.BATCH_SIZE)
    te = b.DataLoader(b.MultiStationDataset(c["test_d"]), batch_size=b.BATCH_SIZE)
    model = b.SpatioTemporalModel(
        in_features=len(b.FEATURES), d_model=b.D_MODEL, n_heads=b.N_HEADS,
        n_layers=b.N_LAYERS, dropout=b.DROPOUT, n_nodes=c["n_nodes"],
        encoder_type="gcn2").to(device)
    with redirect_stdout(io.StringIO()):
        model = b.train_model(model, tl, vl, ei, ew, device,
                              max_epochs=b.MAX_EPOCHS, patience=b.PATIENCE)
        _, _, r2, _ = b.evaluate(model, te, ei, ew, c["scaler"], device)
    del model, tl, vl, te
    _free_mps()
    return float(r2)


def canonical_from_json(city):
    """Linear (par seed, par topo) et GCN k=5 (par seed, par topo) canoniques."""
    p = ROOT / f"results/{city}/multistation_results.json"
    g = json.loads(p.read_text())["graphs"]
    lin = {t: g[t]["Linear+Transformer"]["R2"] for t in TOPOS}
    gcn5 = {t: g[t]["GCN+Transformer"]["R2"] for t in TOPOS}
    return lin, gcn5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", required=True,
                    choices=["beijing", "london", "madrid"])
    ap.add_argument("--k", nargs="+", type=int, default=[3, 8],
                    help="k à recalculer (5 réutilisé du JSON par défaut)")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    b = load_bench()

    rows = []
    for city in args.cities:
        c = get_city(b, city)
        N = c["n_nodes"]
        lin_json, gcn5_json = canonical_from_json(city)
        print(f"\n[{city}] {N} nœuds", file=sys.stderr)
        for k in KS:
            k_eff = min(k, N - 1)
            for topo in TOPOS:
                lin = np.array(lin_json[topo])                 # canonique, par seed
                if k == 5:
                    gcn = np.array(gcn5_json[topo])            # réutilisé du JSON
                    src = "json(k5)"
                elif k in args.k:
                    ei, ew = build_graph_k(b, c, topo, k)
                    gcn = np.array([train_gcn_r2(b, c, ei, ew, s, device)
                                    for s in args.seeds])
                    src = "recompute"
                else:
                    continue
                delta = gcn - lin[:len(gcn)]
                rows.append(dict(city=city, k=k, k_eff=k_eff, topology=topo,
                                 gcn_r2_mean=gcn.mean(), gcn_r2_std=gcn.std(ddof=0),
                                 lin_r2_mean=lin.mean(),
                                 delta_r2_mean=delta.mean(), delta_r2_std=delta.std(ddof=0),
                                 n_seeds=len(gcn), source=src))
                print(f"  k={k} (eff {k_eff}) {topo:12s}: ΔR²={delta.mean():+.4f} "
                      f"± {delta.std(ddof=0):.4f}  [{src}]", file=sys.stderr)

    new = pd.DataFrame(rows)
    if CSV.exists():
        old = pd.read_csv(CSV)
        old = old[~old.city.isin(args.cities)]
        full = pd.concat([old, new], ignore_index=True)
    else:
        full = new
    full = full.sort_values(["city", "k", "topology"]).reset_index(drop=True)
    full.to_csv(CSV, index=False)
    print(f"\nCSV : {CSV}  ({len(full)} lignes)", file=sys.stderr)


if __name__ == "__main__":
    main()
