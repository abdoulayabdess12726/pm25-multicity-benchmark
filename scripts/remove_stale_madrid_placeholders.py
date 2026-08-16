#!/usr/bin/env python3
"""
remove_stale_madrid_placeholders.py — correctif ponctuel post-E9/E10 (P5)
==========================================================================
migrate_k_sensitivity() et migrate_edge_pruning() (scripts/migrate_raw_results.py)
avaient migré, avant E9/E10, des lignes placeholder Madrid marquées
UNRECOVERABLE_6STATION / SUSPECT_6STATION (n_stations=6, MENDEZ ALVARO absente),
sous les run_id "migrated_e6_k_sensitivity/e8_k_sensitivity_3seeds_madrid" (12
lignes, k∈{3,8}) et "migrated_13_edge_pruning_madrid" (105 lignes, 5 niveaux ×
3 seeds × (6 stations + agrégat)).

E9/E10 ont maintenant produit les données réelles 7-station pour CHACUNE de
ces conditions (même identité city/model/topology/k/keep_frac/seed/station,
run_id différent) — vérifié 1:1 avant exécution (aucune condition orpheline).
Les 117 lignes placeholder sont donc strictement redondantes : on les retire.

Ce n'est PAS un hand-edit de valeur (aucun chiffre n'est modifié) — seule la
suppression de lignes déjà invalidées par un ré-entraînement réel, comme pour
la ligne UNRECOVERABLE beijing/k=3 (cf. migrate_raw_results.py, source 7).
migrate_k_sensitivity()/migrate_edge_pruning() ont été corrigées en parallèle
pour ne plus jamais réintroduire ces placeholders lors d'une migration
complète future.

Usage : python3 scripts/remove_stale_madrid_placeholders.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "raw_results.csv"
STALE_RUN_IDS = [
    "migrated_e6_k_sensitivity/e8_k_sensitivity_3seeds_madrid",
    "migrated_13_edge_pruning_madrid",
]
EXPECTED_STALE_COUNT = 117


def main():
    df = pd.read_csv(CSV, dtype=str)
    n_before = len(df)
    stale_mask = df.run_id.isin(STALE_RUN_IDS)
    n_stale = int(stale_mask.sum())

    if n_stale != EXPECTED_STALE_COUNT:
        sys.exit(
            f"ATTENDU {EXPECTED_STALE_COUNT} lignes stale, trouvé {n_stale} — "
            "arrêt (ne pas supprimer à l'aveugle, revérifier)."
        )

    kept = df[~stale_mask].reset_index(drop=True)
    kept.to_csv(CSV, index=False)
    print(f"raw_results.csv : {n_before} -> {len(kept)} lignes "
          f"(-{n_stale} placeholders Madrid superseded par E9/E10, P5)")


if __name__ == "__main__":
    main()
