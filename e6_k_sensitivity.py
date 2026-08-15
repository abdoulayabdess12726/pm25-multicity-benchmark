#!/usr/bin/env python3
"""
e6_k_sensitivity.py  —  R6 / Table 7 : robustesse de ΔR² au nombre de voisins k
(Table 7 sous la numérotation confirmée du cycle 20265149 — anciennement
« Table 6 » sous la numérotation du cycle 20264131 ; ne pas confondre avec
l'actuelle Table 6 = ΔR² par station, cf. REVISION_BRIEF.md)
==========================================================================
PÉRIMÈTRE STRICT : 3 villes × 2 topologies × k∈{3,8} × seed 42 = 12 entraînements
GCN-Transformer. k=5 et Linear-Transformer NE SONT PAS ré-entraînés : réutilisés
depuis les résultats sauvegardés (results/{city}/multistation_results.json,
per_station_all_seeds, seed 42).

ΔR² = R²(GCN-Transformer) − R²(Linear-Transformer), agrégat global dénormalisé,
stations retenues via src.stations.load_stations(city, "benchmark") (source
unique, cf. REVISION_BRIEF.md — Madrid = 7 stations, MENDEZ ALVARO incluse).
Protocole identique à la config principale (splits 70/15/15, 5 features,
SEQ_LEN=24, MinMax train, cibles test[24:], d_model/heads/layers/batch/lr/
epochs/early-stop inchangés). Le SEUL paramètre qui change est k.

⚠️ results/e6_k_sensitivity.csv EXISTANT (avant ce correctif) est marqué NON
UTILISABLE pour Madrid : l'ancien code excluait MENDEZ ALVARO de `kept` avant
même le calcul du R² agrégé, donc aucune métrique la concernant n'a jamais
été calculée ni stockée — contrairement aux baselines externes, il n'y a rien
à ré-agréger a posteriori. Un ré-entraînement complet (E9, cf.
REVISION_BRIEF.md P5) est nécessaire ; ce script, une fois relancé pour
Madrid, produira des lignes correctes (7 stations).

Sortie : results/e6_k_sensitivity.csv (18 lignes) + tableau markdown + durée.

RÈGLE D'ARRÊT : toute divergence (réf. introuvable, n_edges incohérent, station
manquante, non-convergence) → exception, on ne corrige rien silencieusement.
"""
import argparse
import gc
import importlib.util
import io
import json
import resource
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.stations import load_stations  # noqa: E402 — source unique des listes de stations

SEED = 42
KS = [3, 5, 8]
RECOMPUTE = [3, 8]
TOPOS = ["distance", "correlation"]
CSV = ROOT / "results" / "e6_k_sensitivity.csv"


def _free_mps():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_bench():
    spec = importlib.util.spec_from_file_location("bench", str(ROOT / "06_train_multistation.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = mod
    spec.loader.exec_module(mod)
    return mod


def get_city(b, city):
    with redirect_stdout(io.StringIO()):
        if city == "beijing":
            ret = b.load_beijing_data(str(ROOT / "data/beijing_real/PRSA_Data_20130301-20170228"))
        elif city == "london":
            ret = b.load_london_data()
        else:
            ret = b.load_madrid_data()
        data = ret[0] if isinstance(ret, (tuple, list)) else ret
        data = np.asarray(data, dtype=np.float32)
        train_d, val_d, test_d, scaler = b.split_and_scale(data)
    names = list(b.STATION_NAMES)
    js = json.loads((ROOT / f"results/{city}/multistation_results.json").read_text())
    if js["station_names"] != names:                      # STOP : cohérence stations
        raise RuntimeError(f"{city}: stations JSON != loader {js['station_names']} vs {names}")
    pm = b.FEATURES.index("PM2.5")
    allowed = set(load_stations(city, "benchmark"))
    kept = [i for i, s in enumerate(names) if s in allowed]
    if len(kept) != len(allowed):                          # STOP : incohérence config/loader
        raise RuntimeError(f"{city}: {len(allowed)} stations attendues (load_stations), "
                           f"{len(kept)} retrouvées dans le loader {names}")
    Y = data[int(0.85 * len(data)):][b.SEQ_LEN:, :, pm]   # cibles test dénormalisées
    n = Y.shape[0]
    Yk = Y[:, kept]
    ss_tot = float(((Yk - Yk.mean()) ** 2).sum())
    return dict(city=city, data=data, names=names, pm=pm, n_nodes=len(names),
                kept=kept, Y=Y, n=n, ss_tot=ss_tot, scaler=scaler,
                train_d=train_d, val_d=val_d, test_d=test_d, js=js)


def ref_r2_from_json(c, model_key, topo):
    """R² seed-42 agrégé sur stations conservées, reconstruit exactement depuis
    le per-station sauvegardé : SS_res=Σ RMSE_s²·n, SS_tot depuis les données."""
    psa = c["js"]["graphs"][topo].get("per_station_all_seeds", {}).get(model_key, {})
    per = psa.get("42")
    if per is None:                                       # STOP : réf introuvable
        raise RuntimeError(f"{c['city']}/{topo}: {model_key} seed 42 introuvable dans le JSON")
    ss_res = sum(per[c["names"][i]]["RMSE"] ** 2 * c["n"] for i in c["kept"])
    return 1.0 - ss_res / c["ss_tot"]


def build_graph_k(b, c, topo, k):
    with redirect_stdout(io.StringIO()):
        if topo == "distance":
            ei, ew = b.build_graph(k=k)
        else:
            ei, ew = b.build_correlation_graph(c["data"][:int(0.70 * len(c["data"]))], k=k)
    return ei, ew


def train_gcn_r2(b, c, ei, ew, device):
    torch.manual_seed(SEED); np.random.seed(SEED)
    tl = b.DataLoader(b.MultiStationDataset(c["train_d"]), batch_size=b.BATCH_SIZE, shuffle=True)
    vl = b.DataLoader(b.MultiStationDataset(c["val_d"]), batch_size=b.BATCH_SIZE)
    te = b.DataLoader(b.MultiStationDataset(c["test_d"]), batch_size=b.BATCH_SIZE)
    model = b.SpatioTemporalModel(
        in_features=len(b.FEATURES), d_model=b.D_MODEL, n_heads=b.N_HEADS,
        n_layers=b.N_LAYERS, dropout=b.DROPOUT, n_nodes=c["n_nodes"], encoder_type="gcn2").to(device)
    eid, ewd = ei.to(device), ew.to(device)
    with redirect_stdout(io.StringIO()):
        model = b.train_model(model, tl, vl, eid, ewd, device,
                              max_epochs=b.MAX_EPOCHS, patience=b.PATIENCE)
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in te:
                preds.append(model(xb.to(device), eid, ewd).cpu().numpy())
    P = np.concatenate(preds)                              # (n, N) scaled
    lo, hi = c["scaler"].data_min_[c["pm"]], c["scaler"].data_max_[c["pm"]]
    P = P * (hi - lo) + lo                                 # dénormalisé
    if P.shape != c["Y"].shape:                            # STOP : alignement cibles
        raise RuntimeError(f"{c['city']}: pred {P.shape} != cibles {c['Y'].shape}")
    kept = c["kept"]
    r2 = r2_score(c["Y"][:, kept].reshape(-1), P[:, kept].reshape(-1))
    del model, tl, vl, te
    _free_mps()
    return float(r2)


def upsert_row(row):
    """Écrit/actualise UNE ligne (city,topology,k) dans le CSV, sans toucher aux autres."""
    cols = ["city", "topology", "k", "n_edges", "R2_gcn", "R2_linear_ref", "delta_R2"]
    df = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame(columns=cols)
    m = (df.city == row["city"]) & (df.topology == row["topology"]) & (df.k == row["k"])
    df = df[~m]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values(["city", "topology", "k"]).reset_index(drop=True)
    df.to_csv(CSV, index=False)
    return len(df)


def run_single(city, topology, k, cpu=False):
    """UN entraînement isolé (sous-processus frais), upsert dans le CSV,
    report durée + RAM résidente max. Respecte la gate de timing."""
    t0 = time.time()
    device = "cpu" if cpu else ("mps" if torch.backends.mps.is_available()
                                else "cuda" if torch.cuda.is_available() else "cpu")
    b = load_bench()
    c = get_city(b, city)
    lin = ref_r2_from_json(c, "Linear+Transformer", topology)
    ei, ew = build_graph_k(b, c, topology, k)
    n_edges = int(ei.shape[1])
    exp_max = c["n_nodes"] * min(k, c["n_nodes"] - 1)
    if n_edges > exp_max or n_edges == 0:
        raise RuntimeError(f"{city}/{topology} k={k}: n_edges={n_edges} incohérent (max {exp_max})")
    r2_gcn = train_gcn_r2(b, c, ei, ew, device)
    row = dict(city=city, topology=topology, k=k, n_edges=n_edges,
               R2_gcn=round(r2_gcn, 4), R2_linear_ref=round(lin, 4),
               delta_R2=round(r2_gcn - lin, 4))
    ntot = upsert_row(row)
    dt = time.time() - t0
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)  # macOS: octets
    print(f"[{city}/{topology}/k={k}] device={device} edges={n_edges} "
          f"R2_gcn={r2_gcn:.4f} ΔR²={row['delta_R2']:+.4f}", file=sys.stderr)
    print(f"DUREE_S={dt:.0f} DUREE_MIN={dt/60:.1f} RSS_GB={rss_gb:.2f} CSV_LIGNES={ntot}",
          file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=["beijing", "london", "madrid"])
    ap.add_argument("--topology", choices=["distance", "correlation"])
    ap.add_argument("--k", type=int)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    if args.city and args.topology and args.k:
        run_single(args.city, args.topology, args.k, cpu=args.cpu)
        return

    t0 = time.time()
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    b = load_bench()
    rows = []
    for city in ["beijing", "london", "madrid"]:
        c = get_city(b, city)
        lin = {t: ref_r2_from_json(c, "Linear+Transformer", t) for t in TOPOS}
        print(f"\n[{city}] {c['n_nodes']} nœuds, {len(c['kept'])} stations conservées, "
              f"Linear s42 ref: dist={lin['distance']:.4f} corr={lin['correlation']:.4f}",
              file=sys.stderr)
        for topo in TOPOS:
            for k in KS:
                ei, ew = build_graph_k(b, c, topo, k)
                n_edges = int(ei.shape[1])
                exp_max = c["n_nodes"] * min(k, c["n_nodes"] - 1)
                if n_edges > exp_max or n_edges == 0:      # STOP : n_edges incohérent
                    raise RuntimeError(f"{city}/{topo} k={k}: n_edges={n_edges} incohérent (max {exp_max})")
                if k == 5:
                    r2_gcn = ref_r2_from_json(c, "GCN+Transformer", topo)
                    src = "json(k5)"
                elif k in RECOMPUTE:
                    r2_gcn = train_gcn_r2(b, c, ei, ew, device)
                    src = "recompute"
                else:
                    continue
                d = r2_gcn - lin[topo]
                rows.append(dict(city=city, topology=topo, k=k, n_edges=n_edges,
                                 R2_gcn=round(r2_gcn, 4), R2_linear_ref=round(lin[topo], 4),
                                 delta_R2=round(d, 4)))
                print(f"  k={k:>1} {topo:11s} edges={n_edges:>3} : R2_gcn={r2_gcn:.4f} "
                      f"ΔR²={d:+.4f}  [{src}]", file=sys.stderr)

    df = pd.DataFrame(rows, columns=["city", "topology", "k", "n_edges",
                                     "R2_gcn", "R2_linear_ref", "delta_R2"])
    df = df.sort_values(["city", "topology", "k"]).reset_index(drop=True)
    df.to_csv(CSV, index=False)

    # tableau markdown compact : ligne = ville×topo, colonnes k=3/5/8 (ΔR²)
    piv = df.pivot_table(index=["city", "topology"], columns="k", values="delta_R2")
    print("\n\n### Table 7 — ΔR² (GCN − Linear) vs k  [seed 42, stations via load_stations(city,'benchmark')]\n")
    print("| City | Topology | k=3 | k=5 | k=8 |")
    print("|---|---|---|---|---|")
    for (city, topo), r in piv.iterrows():
        print(f"| {city.capitalize()} | {topo} | {r[3]:+.4f} | {r[5]:+.4f} | {r[8]:+.4f} |")
    dt = time.time() - t0
    print(f"\nCSV : {CSV}  ({len(df)} lignes)")
    print(f"Durée totale : {dt/60:.1f} min ({dt:.0f} s)")


if __name__ == "__main__":
    main()
