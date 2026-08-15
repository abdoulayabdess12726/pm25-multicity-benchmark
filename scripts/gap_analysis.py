#!/usr/bin/env python3
"""
scripts/gap_analysis.py — quelles conditions du manuscrit n'ont AUCUNE ligne
dans results/raw_results.csv (P3, REVISION_BRIEF.md).

Distingue explicitement deux catégories, jamais confondues :
  - MANQUANT  : aucune ligne du tout pour cette condition -> P4 ne pourra
    rien calculer, doit soit s'arrêter soit exclure explicitement.
  - PRÉSENT MAIS FLAGGÉ : une ligne existe (provenance_note non vide,
    SUSPECT_6STATION / UNRECOVERABLE) -> P4 peut la lire, mais ne doit pas
    la traiter comme fiable sans décision explicite.

La grille attendue par table est définie ici à partir du protocole documenté
dans REVISION_BRIEF.md / CLAUDE.md / les docstrings des scripts (SEEDS,
TOPOS, KS, LEVELS) — pas une supposition.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.results_io import load_results

CITIES = ["beijing", "london", "madrid"]
TOPOS = ["distance", "correlation"]
SEEDS = [42, 123, 777]
KS = [3, 5, 8]
LEVELS = [1.0, 0.75, 0.50, 0.25, 0.0]


def _agg(df, model, variant=None, topology=None, k=None, keep_frac=None, seed=None):
    """Sous-ensemble des lignes __aggregate__ pour un (model,[variant],[topology],
    [k],[keep_frac],[seed]). Un champ vide écrit "" se relit en NaN depuis le
    CSV (round-trip pandas) — normalisé ici plutôt que de laisser chaque
    appelant s'en souvenir."""
    m = (df.station == "__aggregate__") & (df.model == model)
    if variant is not None:
        m &= (df.variant.fillna("") == variant)
    if topology is not None:
        m &= (df.topology.fillna("") == topology)
    if k is not None:
        m &= (df.k.astype(str).fillna("") == str(k))
    if keep_frac is not None:
        m &= (df.keep_frac.astype(str).fillna("") == str(keep_frac))
    if seed is not None:
        m &= (df.seed.astype(str) == str(seed))
    return df[m]


def check(label, expected, present_keys):
    missing = expected - present_keys
    print(f"\n### {label}")
    print(f"  attendu : {len(expected)}  |  présent : {len(expected & present_keys)}  "
          f"|  MANQUANT : {len(missing)}")
    if missing:
        for k in sorted(missing, key=str):
            print(f"    MANQUANT: {k}")
    return missing


def main():
    df = load_results()
    print(f"raw_results.csv : {len(df)} lignes totales, "
          f"{(df.station == '__aggregate__').sum()} lignes agrégées")

    all_missing = {}

    # --- Table 2 : benchmark canonique -------------------------------------
    # Linear-Transformer dupliqué par topologie (comportement du script live).
    expected = set()
    for city in CITIES:
        for topo in TOPOS:
            for seed in SEEDS:
                expected.add(("Linear-Transformer", city, topo, "", seed))
                expected.add(("GCN-Transformer", city, topo, 5, seed))
    present = set()
    for _, r in _agg(df, "Linear-Transformer", variant="").iterrows():
        try:
            seed_int = int(float(r.seed))
        except ValueError:
            continue
        present.add(("Linear-Transformer", r.city, r.topology, "", seed_int))
    for _, r in _agg(df, "GCN-Transformer", variant="").iterrows():
        if str(r.k) not in ("5.0", "5"):
            continue
        try:
            seed_int = int(float(r.seed))
        except ValueError:
            continue
        present.add(("GCN-Transformer", r.city, r.topology, 5, seed_int))
    all_missing["Table 2 (benchmark canonique)"] = check("Table 2 — benchmark canonique", expected, present)

    # --- Table 3 : baselines externes + SOTA --------------------------------
    expected = set()
    for city in CITIES:
        for model in ["Persistence", "ARIMA"]:
            expected.add((model, city, "-"))        # déterministes, seed non applicable
        expected.add(("XGBoost", city, "42"))       # seed fixe 42 (pas "-")
        for seed in SEEDS:
            expected.add(("LSTM", city, str(seed)))
        for model in ["stgcn", "graphwavenet"]:
            expected.add((model, city, "42"))
    present = set()
    for model in ["Persistence", "ARIMA", "XGBoost", "LSTM", "stgcn", "graphwavenet"]:
        for _, r in _agg(df, model).iterrows():
            present.add((model, r.city, str(r.seed)))
    all_missing["Table 3 (baselines externes + SOTA)"] = check(
        "Table 3 — baselines externes + SOTA (STGCN/Graph WaveNet)", expected, present)

    # --- Table 7 : sensibilité k --------------------------------------------
    expected = {(city, topo, k, seed) for city in CITIES for topo in TOPOS
                for k in KS for seed in SEEDS}
    present = set()
    for _, r in _agg(df, "GCN-Transformer", variant="").iterrows():
        try:
            k_int = int(float(r.k))
        except (ValueError, TypeError):
            continue
        if k_int not in KS:
            continue
        seed_val = str(r.seed)
        if seed_val == "unknown_3seed_agg":
            continue  # agrégat non attribuable à un seed précis, pas compté "présent par seed"
        try:
            seed_int = int(float(seed_val))
        except ValueError:
            continue
        present.add((r.city, r.topology, k_int, seed_int))
    all_missing["Table 7 (sensibilité k)"] = check("Table 7 — sensibilité k", expected, present)

    # --- Table 8 : over-smoothing / GAT --------------------------------------
    # (model, variant) distincts de la Table 2 (Linear-Transformer/variant="")
    # -> jamais confondus même si topology coïncide (distance).
    variant_specs = [("Linear-Transformer", "1layer"), ("GCN-Transformer", "1layer"),
                     ("GCN-Transformer", "2layer"), ("GAT-Transformer", "2layer")]
    expected = {(city, "distance", model, variant, seed) for city in CITIES
                for model, variant in variant_specs for seed in SEEDS}
    present = set()
    for model, variant in variant_specs:
        for _, r in _agg(df, model, variant=variant, topology="distance").iterrows():
            try:
                seed_int = int(float(r.seed))
            except ValueError:
                continue
            present.add((r.city, "distance", model, variant, seed_int))
    all_missing["Table 8 (over-smoothing/GAT)"] = check(
        "Table 8 — over-smoothing / GAT (aucune ligne n'existait avant P3)", expected, present)

    # --- Table 9 : contrôles diagnostiques -----------------------------------
    expected = set()
    for city in CITIES:
        for topo in TOPOS:
            for exp in ["real", "shuffled_graph", "no_meteorology"]:
                expected.add((city, topo, "GCN-Transformer", exp, 42))
            expected.add((city, topo, "Linear-Transformer", "no_meteorology", 42))
    present = set()
    for model in ["GCN-Transformer", "Linear-Transformer"]:
        for _, r in _agg(df, model).iterrows():
            variant = str(r.variant) if pd.notna(r.variant) else ""
            if variant not in ("real", "shuffled_graph", "no_meteorology"):
                continue
            try:
                seed_int = int(float(r.seed))
            except ValueError:
                continue
            present.add((r.city, r.topology, model, variant, seed_int))
    all_missing["Table 9 (diagnostics)"] = check("Table 9 — contrôles diagnostiques", expected, present)

    # --- Édition d'arêtes (§6.2.1) -------------------------------------------
    expected = {(city, lvl, seed) for city in CITIES for lvl in LEVELS for seed in SEEDS}
    present = set()
    for _, r in _agg(df, "GCN-Transformer", variant="", topology="distance").iterrows():
        try:
            kf = float(r.keep_frac)
        except (ValueError, TypeError):
            continue
        if kf not in LEVELS:
            continue
        try:
            seed_int = int(float(r.seed))
        except ValueError:
            continue
        present.add((r.city, kf, seed_int))
    all_missing["Edge pruning (§6.2.1)"] = check("Édition d'arêtes (§6.2.1)", expected, present)

    # --- Récap ---------------------------------------------------------------
    print(f"\n{'='*60}\nRÉCAP MANQUANTS (aucune ligne, pas juste flaggé)\n{'='*60}")
    total_missing = 0
    for label, missing in all_missing.items():
        print(f"  {label}: {len(missing)} condition(s) manquante(s)")
        total_missing += len(missing)
    print(f"\nTOTAL : {total_missing} conditions sans aucune ligne dans raw_results.csv")

    # --- Lignes présentes mais flaggées (provenance incertaine) --------------
    flagged = df[df.provenance_note.fillna("") != ""]
    suspect = flagged[flagged.provenance_note.str.contains("SUSPECT_6STATION", na=False)]
    unrecoverable = flagged[flagged.provenance_note.str.contains("UNRECOVERABLE", na=False)]
    lean_only = flagged[~flagged.provenance_note.str.contains("SUSPECT_6STATION|UNRECOVERABLE", na=False)]
    print(f"\n{'='*60}\nLIGNES PRÉSENTES MAIS FLAGGÉES\n{'='*60}")
    print(f"  Total provenance_note non vide : {len(flagged)}")
    print(f"  dont SUSPECT_6STATION (Madrid, ancien protocole) : {len(suspect)}")
    print(f"  dont UNRECOVERABLE (aucune donnée brute) : {len(unrecoverable)}")
    print(f"  dont granularité seulement (pas de RMSE/MAE/per-station, mais fiable) : {len(lean_only)}")


if __name__ == "__main__":
    main()
