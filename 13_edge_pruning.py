#!/usr/bin/env python3
"""
13_edge_pruning.py  —  Étape 5 (B) : élagage des arêtes hétérophiles
==========================================================================
Hypothèse : la dégradation du GCN-Transformer vient des arêtes HÉTÉROPHILES.
On part du graphe DISTANCE (k=5, celui du GCN canonique) et on retire les arêtes
par hétérophilie décroissante, puis on ré-entraîne le GCN-Transformer et on
mesure la récupération de R².

Hétérophilie d'une arête (i, j) :   h_e = 1 − corr_train(i, j)
    corr = Pearson des séries PM2.5 sur le TRAIN (même base que l'Expérience A).
    Élaguer par hétérophilie décroissante = garder les arêtes de plus forte
    corrélation. Les arêtes vers une station à corr NaN (Madrid/MENDEZ ALVARO,
    PM2.5 constant sur le train) sont traitées comme les plus hétérophiles
    (élaguées en premier).

Niveaux d'arêtes conservées : 100 % (référence), 75 %, 50 %, 25 %, 0 % (vide).
    0 % : graphe vide -> le GCN ne fait plus d'agrégation spatiale (self-loops
    seulement) -> R² doit converger vers le niveau Linear-Transformer (sanity).

Protocole IDENTIQUE à 06 : mêmes splits/scaling/features, mêmes hyperparamètres
GCN-Transformer (gcn2), seeds 42/123/777, early-stopping de 06. Seul le graphe
change entre les runs. MENDEZ ALVARO : dans le graphe (7 nœuds) mais EXCLUE de
l'évaluation (agrégat Madrid sur 6 stations).

Sorties : results/edge_pruning.csv (ville × niveau × seed, per-station + agrégat).
Usage : python 13_edge_pruning.py --cities madrid [--seeds 42] [--levels 1.0 0.0]
"""
import argparse
import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
EXCLUDE = {"madrid": {"MENDEZ ALVARO"}}
LEVELS = [1.0, 0.75, 0.50, 0.25, 0.0]
CSV = ROOT / "results" / "edge_pruning.csv"


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
        ei, ew = b.build_graph(k=b.K_NEIGHBORS)              # graphe DISTANCE de 06
    names = list(b.STATION_NAMES)
    pm = b.FEATURES.index("PM2.5")
    train_len = int(0.70 * len(data))
    corr = np.corrcoef(data[:train_len, :, pm].T)            # comme build_correlation_graph
    return dict(city=city, names=names, pm=pm, data=data,
                train_d=train_d, val_d=val_d, test_d=test_d, scaler=scaler,
                edge_index=ei, edge_weight=ew, corr=corr, n_nodes=len(names))


def prune(edge_index, edge_weight, corr, keep_frac):
    """Garde la fraction keep_frac des arêtes de plus faible hétérophilie."""
    ei = edge_index.cpu().numpy()
    het = 1.0 - corr[ei[0], ei[1]]                           # hétérophilie par arête
    key = np.where(np.isnan(het), np.inf, het)               # NaN -> plus hétérophile
    order = np.argsort(key, kind="stable")                   # hétérophilie croissante
    n_keep = int(round(keep_frac * ei.shape[1]))
    keep = np.sort(order[:n_keep])
    if n_keep == 0:
        return (torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0,), dtype=torch.float32))
    return edge_index[:, keep], edge_weight[keep]


def predict_denorm(model, loader, ei, ew, scaler, pm, device):
    model.eval()
    P, T = [], []
    with torch.no_grad():
        for xb, yb in loader:
            P.append(model(xb.to(device), ei, ew).cpu().numpy())
            T.append(yb.numpy())
    P = np.concatenate(P); T = np.concatenate(T)             # (n, N) en espace scaled
    lo, hi = scaler.data_min_[pm], scaler.data_max_[pm]
    return T * (hi - lo) + lo, P * (hi - lo) + lo            # dénormalisé


def metrics_rows(city, level, seed, names, Y, P):
    excl = EXCLUDE.get(city, set())
    keep = [i for i, n in enumerate(names) if n not in excl]
    rows = []
    for i in keep:
        rows.append(dict(city=city, keep_frac=level, seed=seed, station=names[i],
                         MAE=mean_absolute_error(Y[:, i], P[:, i]),
                         RMSE=float(np.sqrt(mean_squared_error(Y[:, i], P[:, i]))),
                         R2=r2_score(Y[:, i], P[:, i])))
    yf, pf = Y[:, keep].reshape(-1), P[:, keep].reshape(-1)
    rows.append(dict(city=city, keep_frac=level, seed=seed, station="__aggregate__",
                     MAE=mean_absolute_error(yf, pf),
                     RMSE=float(np.sqrt(mean_squared_error(yf, pf))),
                     R2=r2_score(yf, pf)))
    return rows


def train_eval(b, c, ei_p, ew_p, seed, device):
    torch.manual_seed(seed); np.random.seed(seed)
    train_loader = b.DataLoader(b.MultiStationDataset(c["train_d"]),
                                batch_size=b.BATCH_SIZE, shuffle=True)
    val_loader = b.DataLoader(b.MultiStationDataset(c["val_d"]), batch_size=b.BATCH_SIZE)
    test_loader = b.DataLoader(b.MultiStationDataset(c["test_d"]), batch_size=b.BATCH_SIZE)
    model = b.SpatioTemporalModel(
        in_features=len(b.FEATURES), d_model=b.D_MODEL, n_heads=b.N_HEADS,
        n_layers=b.N_LAYERS, dropout=b.DROPOUT, n_nodes=c["n_nodes"],
        encoder_type="gcn2").to(device)
    with redirect_stdout(io.StringIO()):
        model = b.train_model(model, train_loader, val_loader, ei_p, ew_p, device,
                              max_epochs=b.MAX_EPOCHS, patience=b.PATIENCE)
    ei_d, ew_d = ei_p.to(device), ew_p.to(device)
    Y, P = predict_denorm(model, test_loader, ei_d, ew_d, c["scaler"], c["pm"], device)
    return Y, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", required=True,
                    choices=["beijing", "london", "madrid"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 777])
    ap.add_argument("--levels", nargs="+", type=float, default=LEVELS)
    args = ap.parse_args()
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    b = load_bench()

    all_rows = []
    for city in args.cities:
        c = get_city(b, city)
        E = c["edge_index"].shape[1]
        print(f"\n[{city}] {c['n_nodes']} nœuds, {E} arêtes (distance k={b.K_NEIGHBORS})",
              file=sys.stderr)
        for lvl in args.levels:
            ei_p, ew_p = prune(c["edge_index"], c["edge_weight"], c["corr"], lvl)
            for seed in args.seeds:
                Y, P = train_eval(b, c, ei_p, ew_p, seed, device)
                rows = metrics_rows(city, lvl, seed, c["names"], Y, P)
                agg = [r for r in rows if r["station"] == "__aggregate__"][0]
                print(f"  keep={lvl:>4.0%} ({ei_p.shape[1]:>3} arêtes) seed {seed}: "
                      f"R2={agg['R2']:.4f}", file=sys.stderr)
                all_rows += rows

    new = pd.DataFrame(all_rows, columns=["city", "keep_frac", "seed", "station",
                                          "MAE", "RMSE", "R2"])
    if CSV.exists():
        old = pd.read_csv(CSV)
        old = old[~old.city.isin(args.cities)]
        full = pd.concat([old, new], ignore_index=True)
    else:
        full = new
    full.to_csv(CSV, index=False)
    print(f"\nCSV : {CSV}  ({len(full)} lignes ; +{len(new)})", file=sys.stderr)


if __name__ == "__main__":
    main()
