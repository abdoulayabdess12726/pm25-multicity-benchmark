#!/usr/bin/env python3
"""
15_pruning_controls.py — E12 (P5) : contrôles de pruning
==========================================================================
Complète l'expérience d'élagage d'arêtes (13_edge_pruning.py, §6.2.1) par
deux conditions de contrôle, à densité appariée avec le pruning guidé
(élagage par hétérophilie décroissante, cf. 13_edge_pruning.py) :

  (1) ALÉATOIRE À DENSITÉ APPARIÉE : à chaque niveau, retirer un ensemble
      d'arêtes de même cardinalité choisi au hasard (uniforme sans remise),
      moyenné sur >=5 tirages indépendants. Le seed d'un tirage contrôle À
      LA FOIS la sélection aléatoire des arêtes ET l'entraînement du modèle
      pour ce tirage (un tirage = un couple (sous-graphe, seed) cohérent),
      seeds hors bande [1001..1005] pour ne jamais entrer en collision avec
      les seeds canoniques (42/123/777) du pruning guidé/inverse.
  (2) INVERSE : retirer les arêtes par HOMOPHILIE décroissante (les plus
      corrélées d'abord) — l'exact opposé du guidé. Protocole standard
      3 seeds (42/123/777), pas de tirage aléatoire nécessaire (ordre
      déterministe).

Pour chaque condition (ville, stratégie, niveau) : degré effectif moyen
(arêtes conservées / n_noeuds) et magnitude moyenne des messages
(moyenne des edge_weight conservés — le coefficient par lequel le GCN
pondère chaque message avant agrégation) sont rapportés pour montrer ce
qui est tenu constant entre stratégies (densité, donc degré) et ce qui
diffère (quelles arêtes, donc quelle magnitude moyenne).

RÉDUCTION DE BUDGET (consigne explicite, P5, 2026-08-18) : "si E12 dépasse
largement le budget, réduire le nombre de niveaux plutôt que le nombre de
tirages". Niveaux 1.0 et 0.0 sont partagés avec le pruning guidé (à ces
deux extrêmes, guidé/aléatoire/inverse sont par construction IDENTIQUES —
graphe complet ou graphe vide, aucune donnée n'est réentraînée pour ces
2 points, réutilisés depuis results/edge_pruning.csv). Seuls les niveaux
intermédiaires nécessitent un réentraînement pour aléatoire/inverse ;
réduits à {0.75, 0.25} (au lieu des 5 niveaux canoniques) pour tenir le
budget — les >=5 tirages aléatoires ne sont PAS réduits.

Sorties : results/pruning_controls.csv (ville×stratégie×niveau×tirage/seed,
per-station + agrégat, degré effectif, magnitude moyenne des messages) +
raw_results.csv (variant="random_matched"|"inverse").
Usage : python 15_pruning_controls.py --cities madrid --strategies random inverse --levels 0.75 0.25
"""
import argparse
import importlib.util
import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.csv_upsert import upsert_rows  # noqa: E402
from src.results_io import (append_run, make_run_id, git_commit_hash,  # noqa: E402
                            compute_split_hash)

LEVELS_REDUCED = [0.75, 0.25]      # niveaux intermédiaires réentraînés (E12)
LEVELS_SHARED = [1.0, 0.0]         # partagés avec le pruning guidé (E10/E14), non réentraînés
N_RANDOM_DRAWS = 5
RANDOM_DRAW_SEEDS = [1001, 1002, 1003, 1004, 1005]   # hors bande [42,123,777]
CANONICAL_SEEDS = [42, 123, 777]
CSV = ROOT / "results" / "pruning_controls.csv"
GUIDED_CSV = ROOT / "results" / "edge_pruning.csv"


def load_edge_pruning_module():
    """Réutilise EXACTEMENT get_city/prune/train_eval/metrics_rows/predict_denorm
    de 13_edge_pruning.py (même graphe de base, même protocole)."""
    spec = importlib.util.spec_from_file_location("ep", str(ROOT / "13_edge_pruning.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ep"] = mod
    with redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def prune_random(edge_index, edge_weight, keep_frac, rng):
    """Sous-ensemble uniforme sans remise, cardinalité appariée au guidé."""
    E = edge_index.shape[1]
    n_keep = int(round(keep_frac * E))
    keep = np.sort(rng.choice(E, size=n_keep, replace=False)) if n_keep > 0 else np.array([], dtype=int)
    if n_keep == 0:
        return (torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0,), dtype=torch.float32))
    return edge_index[:, keep], edge_weight[keep]


def prune_inverse(edge_index, edge_weight, corr, keep_frac):
    """Garde la fraction keep_frac des arêtes de plus forte HÉTÉROPHILIE
    (= exact opposé de 13_edge_pruning.py::prune, qui garde la plus faible
    hétérophilie — même tri, ordre inversé). Équivaut à retirer d'abord les
    arêtes les plus homophiles (les plus corrélées), cf. consigne E12."""
    ei = edge_index.cpu().numpy()
    het = 1.0 - corr[ei[0], ei[1]]
    key = np.where(np.isnan(het), np.inf, het)
    order = np.argsort(-key, kind="stable")          # hétérophilie DÉCROISSANTE en premier -> garde les plus hétérophiles
    # équivalent : on retire d'abord les arêtes les plus corrélées (homophiles)
    n_keep = int(round(keep_frac * ei.shape[1]))
    keep = np.sort(order[:n_keep])
    if n_keep == 0:
        return (torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0,), dtype=torch.float32))
    return edge_index[:, keep], edge_weight[keep]


def effective_degree(edge_index, n_nodes):
    if edge_index.shape[1] == 0:
        return 0.0
    return float(edge_index.shape[1]) / float(n_nodes)


def mean_message_magnitude(edge_weight):
    if edge_weight.numel() == 0:
        return 0.0
    return float(edge_weight.abs().mean().item())


def run_condition(ep, b, c, strategy, level, seed, ei_p, ew_p, device):
    Y, P = ep.train_eval(b, c, ei_p, ew_p, seed, device)
    rows = ep.metrics_rows(c["city"], level, seed, c["names"], Y, P)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", required=True,
                    choices=["beijing", "london", "madrid"])
    ap.add_argument("--strategies", nargs="+", default=["random", "inverse"],
                    choices=["random", "inverse"])
    ap.add_argument("--levels", nargs="+", type=float, default=LEVELS_REDUCED)
    ap.add_argument("--n_draws", type=int, default=N_RANDOM_DRAWS)
    ap.add_argument("--cpu", action="store_true",
                    help="force device=cpu (cf. E3/NOTE.md)")
    args = ap.parse_args()
    device = "cpu" if args.cpu else ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    ep = load_edge_pruning_module()
    b = ep.load_bench()

    all_metric_rows = []
    condition_rows = []   # degré effectif / magnitude, un par (ville,strategie,niveau,[tirage])

    for city in args.cities:
        c = ep.get_city(b, city)
        c["city"] = city
        E_total = c["edge_index"].shape[1]
        print(f"\n[{city}] {c['n_nodes']} noeuds, {E_total} aretes de base (distance k=5)",
              file=sys.stderr)

        for lvl in args.levels:
            if "random" in args.strategies:
                for draw_idx, draw_seed in enumerate(RANDOM_DRAW_SEEDS[:args.n_draws]):
                    rng = np.random.RandomState(draw_seed)
                    ei_p, ew_p = prune_random(c["edge_index"], c["edge_weight"], lvl, rng)
                    deg = effective_degree(ei_p, c["n_nodes"])
                    mag = mean_message_magnitude(ew_p)
                    condition_rows.append(dict(city=city, strategy="random_matched",
                                               keep_frac=lvl, draw=draw_idx, seed=draw_seed,
                                               effective_degree=deg, mean_message_magnitude=mag,
                                               n_edges=ei_p.shape[1]))
                    rows = run_condition(ep, b, c, "random_matched", lvl, draw_seed, ei_p, ew_p, device)
                    agg = [r for r in rows if r["station"] == "__aggregate__"][0]
                    print(f"  [random] keep={lvl:>4.0%} tirage {draw_idx} (seed={draw_seed}) "
                          f"deg_eff={deg:.2f} msg_mag={mag:.3f} R2={agg['R2']:.4f}",
                          file=sys.stderr)
                    for r in rows:
                        r["strategy"] = "random_matched"; r["draw"] = draw_idx
                    all_metric_rows += rows

            if "inverse" in args.strategies:
                ei_p, ew_p = prune_inverse(c["edge_index"], c["edge_weight"], c["corr"], lvl)
                deg = effective_degree(ei_p, c["n_nodes"])
                mag = mean_message_magnitude(ew_p)
                condition_rows.append(dict(city=city, strategy="inverse", keep_frac=lvl,
                                           draw="", seed="", effective_degree=deg,
                                           mean_message_magnitude=mag, n_edges=ei_p.shape[1]))
                for seed in CANONICAL_SEEDS:
                    rows = run_condition(ep, b, c, "inverse", lvl, seed, ei_p, ew_p, device)
                    agg = [r for r in rows if r["station"] == "__aggregate__"][0]
                    print(f"  [inverse] keep={lvl:>4.0%} seed={seed} deg_eff={deg:.2f} "
                          f"msg_mag={mag:.3f} R2={agg['R2']:.4f}", file=sys.stderr)
                    for r in rows:
                        r["strategy"] = "inverse"; r["draw"] = ""
                    all_metric_rows += rows

    if not all_metric_rows:
        print("Aucune cellule exécutée.", file=sys.stderr)
        return

    metrics_df = pd.DataFrame(all_metric_rows,
                              columns=["city", "keep_frac", "seed", "station", "MAE", "RMSE", "R2",
                                       "strategy", "draw"])
    full = upsert_rows(CSV, metrics_df,
                       key_cols=["city", "strategy", "keep_frac", "draw", "seed", "station"])
    print(f"\nCSV : {CSV} ({len(full)} lignes ; +{len(metrics_df)})", file=sys.stderr)

    cond_df = pd.DataFrame(condition_rows)
    cond_path = ROOT / "results" / "pruning_controls_conditions.csv"
    cond_full = upsert_rows(cond_path, cond_df,
                            key_cols=["city", "strategy", "keep_frac", "draw"])
    print(f"CSV (degré/magnitude) : {cond_path} ({len(cond_full)} lignes)", file=sys.stderr)

    # ── raw_results.csv (P3) ──
    # variant distingue la stratégie (random_matched|inverse) du pruning
    # guidé (variant="", 13_edge_pruning.py) — jamais mêlé au nom du modèle.
    # Les tirages aléatoires portent leur seed hors bande [1001..1005] dans
    # la colonne seed elle-même (contrôle à la fois sélection d'arêtes et
    # entraînement pour ce tirage, cf. docstring) — PAS une colonne "draw"
    # dédiée dans raw_results.csv (schéma figé, REVISION_BRIEF.md) ; `draw`
    # reste seulement dans pruning_controls.csv pour la lisibilité humaine.
    run_id = make_run_id("15_pruning_controls")
    commit = git_commit_hash()
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    raw_rows = []
    for city in args.cities:
        c = ep.get_city(b, city)
        T = len(c["data"])
        t1, t2 = int(0.70 * T), int(0.85 * T)
        shash = compute_split_hash(T, t1, t2, b.SEQ_LEN)
        for r in all_metric_rows:
            if r["city"] != city:
                continue
            raw_rows.append(dict(
                city=city, model="GCN-Transformer", variant=r["strategy"], topology="distance",
                k="", keep_frac=r["keep_frac"], seed=r["seed"],
                checkpoint_id="no_checkpoint_saved", split_hash=shash,
                n_stations=c["n_nodes"], station=r["station"], split="test",
                rmse=r["RMSE"], mae=r["MAE"], r2=r["R2"], run_id=run_id,
                config_path="15_pruning_controls.py", git_commit=commit,
                timestamp=ts,
                provenance_note=(f"E12 (P5) ; tirage={r['draw']}" if r["strategy"] == "random_matched"
                                 else "E12 (P5)")))
    if raw_rows:
        append_run(raw_rows)
        print(f"raw_results.csv : +{len(raw_rows)} lignes (run_id={run_id})", file=sys.stderr)


if __name__ == "__main__":
    main()
