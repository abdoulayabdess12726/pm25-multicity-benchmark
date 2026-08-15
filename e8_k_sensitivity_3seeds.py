#!/usr/bin/env python3
"""
e8_k_sensitivity_3seeds.py  —  R2.5 / Table 7 : k-sensitivity étendue à 3 seeds
================================================================================
PÉRIMÈTRE STRICT : 3 villes × 2 topologies × k∈{3,8} × seed∈{123,777} = 24
entraînements GCN-Transformer NOUVEAUX (device=cpu). k=5 (3 seeds, benchmark
principal) et k∈{3,8} seed=42 (déjà calculés via E6, archivés dans
results/e6_k_sensitivity_seed42_only.csv) NE SONT PAS ré-entraînés : réutilisés
tels quels.

Protocole IDENTIQUE à E6 / 06_train_multistation.py : splits chronologiques
70/15/15, 5 features, SEQ_LEN=24, horizon 1h, MinMax ajusté sur train, cibles
test[24:], métriques dénormalisées, stations via src.stations.load_stations
(city, "benchmark") — source unique, cf. REVISION_BRIEF.md — seul k (et le
seed, pour les runs nouveaux) change ; tous les autres hyperparamètres
proviennent de 06 (b.MAX_EPOCHS, b.PATIENCE, b.D_MODEL, ...).

⚠️ results/e6_k_sensitivity.csv / results/table7_k_sensitivity.csv EXISTANTS
(avant ce correctif) sont marqués NON UTILISABLES pour Madrid, y compris pour
les 3 seeds : le seed 42 est PULL depuis l'archive
`e6_k_sensitivity_seed42_only.csv`, elle-même produite par l'ancien E6
(MENDEZ ALVARO jamais évaluée) ; les seeds 123/777 étaient calculés ici avec
la même exclusion `EXCLUDE`/`kept`. Aucune métrique MENDEZ ALVARO n'a donc
jamais existé pour aucun des 3 seeds Madrid — irréparable sans ré-entraîner
les 18 cellules Madrid (E9, cf. REVISION_BRIEF.md P5).

Modes d'exécution :
  --city --topology --k --seed [--cpu]   : UNE cellule (sous-processus frais).
        k=5              -> pull JSON (aucun entraînement), tout seed.
        k∈{3,8} seed=42  -> pull depuis l'archive e6_k_sensitivity_seed42_only.csv
                             (aucun entraînement).
        k∈{3,8} seed∈{123,777} -> ENTRAÎNEMENT GCN réel (device=cpu imposé).
  --aggregate                            : recalcule results/e6_k_sensitivity.csv
        en 54 lignes (3 villes×2 topo×3 k×3 seeds) + Table 7 mean±std
        (results/table7_k_sensitivity.csv) + récap markdown.

RÈGLE D'ARRÊT : tout écart au protocole (réf. JSON/archive introuvable,
n_edges incohérent, NaN, station manquante) → exception, rien n'est corrigé
silencieusement.
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

SEEDS_ALL = [42, 123, 777]
SEEDS_NEW = [123, 777]
KS = [3, 5, 8]
RECOMPUTE_KS = [3, 8]
TOPOS = ["distance", "correlation"]
CSV = ROOT / "results" / "e6_k_sensitivity.csv"
BACKUP = ROOT / "results" / "e6_k_sensitivity_seed42_only.csv"
TABLE7_CSV = ROOT / "results" / "table7_k_sensitivity.csv"


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


def ref_r2_from_json(c, model_key, topo, seed):
    """R² seed-{seed} agrégé sur stations conservées, reconstruit depuis le
    per-station sauvegardé : SS_res=Σ RMSE_s²·n, SS_tot depuis les données."""
    psa = c["js"]["graphs"][topo].get("per_station_all_seeds", {}).get(model_key, {})
    per = psa.get(str(seed))
    if per is None:                                       # STOP : réf introuvable
        raise RuntimeError(f"{c['city']}/{topo}: {model_key} seed {seed} introuvable dans le JSON")
    ss_res = sum(per[c["names"][i]]["RMSE"] ** 2 * c["n"] for i in c["kept"])
    return 1.0 - ss_res / c["ss_tot"]


def ref_from_backup(city, topo, k):
    """Valeur seed=42 archivée (E6), pour k∈{3,8} — jamais ré-entraînée."""
    df = pd.read_csv(BACKUP)
    m = (df.city == city) & (df.topology == topo) & (df.k == k)
    if not m.any():                                       # STOP : archive incomplète
        raise RuntimeError(f"{city}/{topo} k={k}: absent de l'archive {BACKUP.name}")
    row = df[m].iloc[0]
    return dict(n_edges=int(row.n_edges), R2_gcn=float(row.R2_gcn), R2_linear_ref=float(row.R2_linear_ref))


def build_graph_k(b, c, topo, k):
    with redirect_stdout(io.StringIO()):
        if topo == "distance":
            ei, ew = b.build_graph(k=k)
        else:
            ei, ew = b.build_correlation_graph(c["data"][:int(0.70 * len(c["data"]))], k=k)
    return ei, ew


def train_gcn_r2(b, c, ei, ew, device, seed):
    torch.manual_seed(seed); np.random.seed(seed)
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
    if not np.isfinite(r2):                                # STOP : non-convergence / NaN
        raise RuntimeError(f"{c['city']}: R² non fini ({r2}) — seed={seed}")
    del model, tl, vl, te
    _free_mps()
    return float(r2)


def upsert_row(row):
    """Écrit/actualise UNE ligne (city,topology,k,seed) dans le CSV."""
    cols = ["city", "topology", "k", "seed", "n_edges", "R2_gcn", "R2_linear_ref", "delta_R2"]
    df = pd.read_csv(CSV) if CSV.exists() else pd.DataFrame(columns=cols)
    if "seed" not in df.columns:
        raise RuntimeError(f"{CSV.name}: colonne 'seed' absente — migration non faite")
    m = (df.city == row["city"]) & (df.topology == row["topology"]) & (df.k == row["k"]) & (df.seed == row["seed"])
    df = df[~m]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values(["city", "topology", "k", "seed"]).reset_index(drop=True)
    df = df[cols]
    df.to_csv(CSV, index=False)
    return len(df)


def run_single(city, topology, k, seed, cpu=True):
    t0 = time.time()
    device = "cpu" if cpu else ("mps" if torch.backends.mps.is_available()
                                else "cuda" if torch.cuda.is_available() else "cpu")
    b = load_bench()
    c = get_city(b, city)
    lin = ref_r2_from_json(c, "Linear+Transformer", topology, seed)
    ei, ew = build_graph_k(b, c, topology, k)
    n_edges = int(ei.shape[1])
    exp_max = c["n_nodes"] * min(k, c["n_nodes"] - 1)
    if n_edges > exp_max or n_edges == 0:
        raise RuntimeError(f"{city}/{topology} k={k}: n_edges={n_edges} incohérent (max {exp_max})")

    if k == 5:
        r2_gcn = ref_r2_from_json(c, "GCN+Transformer", topology, seed)
        src = "json(k5)"
    elif seed == 42:
        ref = ref_from_backup(city, topology, k)
        if ref["n_edges"] != n_edges:                     # STOP : graphe incohérent vs archive
            raise RuntimeError(f"{city}/{topology} k={k}: n_edges recalculé {n_edges} != archive {ref['n_edges']}")
        if abs(ref["R2_linear_ref"] - lin) > 1e-3:          # STOP : Linear-ref divergent
            raise RuntimeError(f"{city}/{topology}: Linear ref recalculé {lin:.4f} != archive {ref['R2_linear_ref']:.4f}")
        r2_gcn = ref["R2_gcn"]
        src = "archive(seed42)"
    else:
        r2_gcn = train_gcn_r2(b, c, ei, ew, device, seed)
        src = "recompute"

    row = dict(city=city, topology=topology, k=k, seed=seed, n_edges=n_edges,
               R2_gcn=round(r2_gcn, 4), R2_linear_ref=round(lin, 4),
               delta_R2=round(r2_gcn - lin, 4))
    ntot = upsert_row(row)
    dt = time.time() - t0
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)  # macOS: octets
    print(f"[{city}/{topology}/k={k}/seed={seed}] device={device} edges={n_edges} "
          f"R2_gcn={r2_gcn:.4f} ΔR²={row['delta_R2']:+.4f} [{src}]", file=sys.stderr)
    print(f"DUREE_S={dt:.0f} DUREE_MIN={dt/60:.1f} RSS_GB={rss_gb:.2f} CSV_LIGNES={ntot}",
          file=sys.stderr)


def migrate():
    """Ajoute la colonne seed=42 aux 18 lignes originales de e6_k_sensitivity.csv
    si elle n'existe pas déjà (idempotent)."""
    df = pd.read_csv(CSV)
    if "seed" in df.columns:
        print("Migration déjà faite (colonne 'seed' présente).", file=sys.stderr)
        return
    if len(df) != 18:                                     # STOP : état inattendu
        raise RuntimeError(f"{CSV.name}: {len(df)} lignes avant migration (attendu 18)")
    df["seed"] = 42
    df = df[["city", "topology", "k", "seed", "n_edges", "R2_gcn", "R2_linear_ref", "delta_R2"]]
    df.to_csv(CSV, index=False)
    print(f"Migration OK : {len(df)} lignes, colonne 'seed' ajoutée.", file=sys.stderr)


def aggregate():
    df = pd.read_csv(CSV)
    expected = len(TOPOS) * 3 * len(KS) * len(SEEDS_ALL)
    if len(df) != expected:                                # STOP : grille incomplète
        raise RuntimeError(f"{CSV.name}: {len(df)} lignes, attendu {expected} (3 villes×2 topo×3 k×3 seeds)")
    for city in ["beijing", "london", "madrid"]:
        for topo in TOPOS:
            for k in KS:
                sub = df[(df.city == city) & (df.topology == topo) & (df.k == k)]
                if sorted(sub.seed.tolist()) != SEEDS_ALL:
                    raise RuntimeError(f"{city}/{topo}/k={k}: seeds {sorted(sub.seed.tolist())} != {SEEDS_ALL}")

    table7 = (df.groupby(["city", "topology", "k"])["delta_R2"]
                .agg(delta_R2_mean="mean", delta_R2_std="std")
                .round(4).reset_index()
                .sort_values(["city", "topology", "k"]))
    table7.to_csv(TABLE7_CSV, index=False)

    print("\n### Table 7 — ΔR² (GCN − Linear) vs k, mean±std sur 3 seeds (42/123/777)\n")
    print("| City | Topology | k=3 | k=5 | k=8 |")
    print("|---|---|---|---|---|")
    piv_m = table7.pivot_table(index=["city", "topology"], columns="k", values="delta_R2_mean")
    piv_s = table7.pivot_table(index=["city", "topology"], columns="k", values="delta_R2_std")
    for (city, topo) in piv_m.index:
        cells = " | ".join(f"{piv_m.loc[(city, topo), kk]:+.4f}±{piv_s.loc[(city, topo), kk]:.4f}" for kk in KS)
        print(f"| {city.capitalize()} | {topo} | {cells} |")
    print(f"\nCSV complet : {CSV} ({len(df)} lignes)")
    print(f"Table 7     : {TABLE7_CSV} ({len(table7)} lignes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=["beijing", "london", "madrid"])
    ap.add_argument("--topology", choices=["distance", "correlation"])
    ap.add_argument("--k", type=int, choices=[3, 5, 8])
    ap.add_argument("--seed", type=int, choices=[42, 123, 777])
    ap.add_argument("--cpu", action="store_true", default=True)
    ap.add_argument("--migrate", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    if args.migrate:
        migrate()
        return
    if args.aggregate:
        aggregate()
        return
    if args.city and args.topology and args.k and args.seed:
        run_single(args.city, args.topology, args.k, args.seed, cpu=args.cpu)
        return
    ap.error("spécifier --migrate, --aggregate, ou --city --topology --k --seed")


if __name__ == "__main__":
    main()
