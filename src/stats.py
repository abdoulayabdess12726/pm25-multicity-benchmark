"""
src/stats.py — convention ddof unique pour les agrégations inter-seeds.

Cause racine (voir REVISION_BRIEF.md) : Table 4 en ddof=0, Table 7 en
ddof=1 — pas un choix explicite incohérent, mais un défaut IMPLICITE
différent entre numpy (`.std()` → ddof=0) et pandas (`.std()` → ddof=1)
selon que le code agrège une liste/array ou un DataFrame. `agg_mean_std`
neutralise ce défaut : ddof=1 est toujours passé explicitement, quel que
soit le type d'entrée — aucun appel ici ne délègue à un défaut de
librairie.

Convention : ddof=1 (écart-type d'échantillon) sur les 3 seeds
(42, 123, 777). Sur un seul point (modèle déterministe : ARIMA, XGBoost,
Persistence), l'écart-type ddof=1 est mathématiquement indéfini
(dénominateur n-1=0) — convention conservée du code existant : renvoyer 0.0.
"""
import numpy as np


def agg_mean_std(values):
    """(mean, std) sur `values`, ddof=1 explicite. `values` : liste, array
    numpy ou Series pandas — le type d'entrée n'affecte jamais le ddof
    utilisé, contrairement à `.std()` appelé nu sur ces mêmes objets."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("agg_mean_std: séquence vide")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std
