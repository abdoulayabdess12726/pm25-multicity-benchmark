"""
src/csv_upsert.py — fusion CSV par clé exacte, jamais par ville.

Cause de l'incident (cf. CHANGELOG_TABLES.md, P2) : `08_sensitivity_k.py`
fusionnait les résultats en supprimant TOUTES les lignes de chaque ville
listée dans `--cities` (`old[~old.city.isin(args.cities)]`) avant de
concaténer les nouvelles lignes — un ré-run ciblé sur une seule condition
(ex. k=5 seul) effaçait silencieusement les autres lignes déjà calculées
pour cette ville (ex. k=3 recompute canonique). Le même pattern existait
indépendamment dans `10_external_baselines.py` et `13_edge_pruning.py`
(non corrigés ici, signalés séparément).

`upsert_rows` ne supprime QUE les lignes dont la clé (`key_cols`) correspond
exactement à une ligne de `new_df` — toute ligne existante dont la clé n'est
pas dans `new_df` est conservée intacte, quelle que soit sa ville.
"""
from pathlib import Path

import pandas as pd


def upsert_rows(csv_path, new_df, key_cols):
    """Fusionne `new_df` dans le CSV à `csv_path` par clé exacte `key_cols`.

    - Les lignes existantes dont la clé (tuple des `key_cols`) apparaît dans
      `new_df` sont remplacées.
    - Toutes les autres lignes existantes sont conservées, peu importe leur
      ville/condition — jamais de suppression en bloc par ville.
    - Comparaison de clé en `str` : une colonne à valeurs mixtes (ex. `seed`
      qui vaut soit un entier soit "-" pour un modèle déterministe) se relit
      depuis le CSV en dtype `object`, ce qui casserait une comparaison de
      tuples typée — cf. incident documenté dans CHANGELOG_TABLES.md.
    - Écrit le résultat sur disque et le retourne.
    """
    csv_path = Path(csv_path)
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        old_keys = old[key_cols].astype(str).apply(tuple, axis=1)
        new_keys = set(new_df[key_cols].astype(str).apply(tuple, axis=1))
        mask_keep = ~old_keys.isin(new_keys)
        full = pd.concat([old[mask_keep], new_df], ignore_index=True)
    else:
        full = new_df.copy()
    full = full.sort_values(key_cols).reset_index(drop=True)
    full.to_csv(csv_path, index=False)
    return full
