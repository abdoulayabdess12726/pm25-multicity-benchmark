"""
src/results_io.py — accès unique à results/raw_results.csv.

Règle non négociable (REVISION_BRIEF.md) : raw_results.csv est écrit une
fois par les runs, jamais édité à la main. Ce module ne fournit
délibérément AUCUNE fonction de mise à jour ni de suppression — l'API
elle-même rend l'immuabilité structurelle, pas seulement documentée :
`append_run` écrit en mode ajout pur (`mode="a"`, jamais de réécriture du
fichier entier), et refuse toute ligne dont la clé
(run_id, city, model, topology, k, seed, station) existe déjà.

Aucune autre fonction publique n'est exportée. Si un script a besoin de
"corriger" une ligne existante : c'est qu'il faut relancer l'expérience et
ajouter les nouvelles lignes (nouveau run_id), jamais modifier les anciennes.
"""
import csv
import hashlib
import subprocess
import time
import uuid
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "results" / "raw_results.csv"

COLUMNS = [
    "city", "model", "variant", "topology", "k", "keep_frac", "seed",
    "checkpoint_id", "split_hash", "n_stations", "station", "split",
    "rmse", "mae", "r2", "run_id", "config_path", "git_commit", "timestamp",
    "provenance_note",
]
KEY_COLS = ["run_id", "city", "model", "variant", "topology", "k", "keep_frac", "seed", "station"]
# `variant` : sous-classification EXPLICITE du modèle (ex. "shuffled_graph",
# "no_meteorology", "1layer") — jamais pliée dans `model` (un champ structuré
# déguisé en chaîne serait imparsable par les assertions de P4). NULL pour
# les runs qui n'ont pas de variante (benchmark canonique, baselines externes).
# `keep_frac` : fraction d'arêtes conservées (édition d'arêtes UNIQUEMENT,
# NULL partout ailleurs). JAMAIS stockée dans `k`, qui reste réservé au k-NN
# et NULL pour les runs sans notion de k-NN (édition d'arêtes, baselines
# externes déterministes).

__all__ = ["append_run", "load_results", "COLUMNS", "KEY_COLS",
           "make_run_id", "git_commit_hash", "compute_split_hash"]


def _row_keys(df):
    """Clés (str) pour comparaison robuste aux dérives de dtype (cf. incident
    P2 : une colonne à valeurs mixtes se relit en dtype object depuis le CSV).
    `fillna("")` AVANT le cast str : un champ vide écrit "" (ex. variant,
    keep_frac sur les runs où ils ne s'appliquent pas) se relit en NaN depuis
    le CSV, qui se serait casté en la chaîne "nan" (≠ "") sans ce fillna —
    aurait fait manquer des collisions de clé légitimes."""
    return set(df[KEY_COLS].fillna("").astype(str).apply(tuple, axis=1))


def load_results():
    """Lecture seule. Renvoie un DataFrame vide (bonnes colonnes) si le
    fichier n'existe pas encore. `k`/`keep_frac` ne sont pas forcés en
    entier (NaN pour les lignes où ils ne s'appliquent pas)."""
    if not RAW_CSV.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(RAW_CSV, dtype={"seed": str})


def append_run(records):
    """Ajoute des lignes à results/raw_results.csv en mode APPEND STRICT.

    - `records` : liste de dicts couvrant au moins KEY_COLS ; les colonnes
      manquantes de COLUMNS sont remplies à vide.
    - Refuse (lève ValueError, n'écrit RIEN) si une clé
      (run_id, city, model, topology, k, seed, station) existe déjà dans le
      fichier, OU si `records` contient des doublons de clé en interne.
    - N'écrit JAMAIS en réécrivant le fichier existant : `mode="a"` pur sur
      un fichier déjà présent (seul le premier appel, fichier absent, écrit
      l'en-tête). Aucune ligne déjà persistée n'est physiquement touchée.
    """
    if not records:
        return
    df_new = pd.DataFrame(records)
    missing = set(KEY_COLS) - set(df_new.columns)
    if missing:
        raise ValueError(f"append_run: colonnes-clé manquantes {missing}")
    for col in COLUMNS:
        if col not in df_new.columns:
            df_new[col] = ""
    df_new = df_new[COLUMNS]

    new_keys_series = df_new[KEY_COLS].fillna("").astype(str).apply(tuple, axis=1)
    if new_keys_series.duplicated().any():
        dups = new_keys_series[new_keys_series.duplicated()].tolist()
        raise ValueError(f"append_run: doublons de clé au sein des records fournis: {dups}")

    if RAW_CSV.exists():
        existing_keys = _row_keys(load_results())
        collisions = set(new_keys_series) & existing_keys
        if collisions:
            raise ValueError(
                f"append_run: {len(collisions)} ligne(s) déjà présente(s), refuse "
                f"d'écraser (append strict uniquement) : {sorted(collisions)[:5]}"
                + (" ..." if len(collisions) > 5 else "")
            )
        df_new.to_csv(RAW_CSV, mode="a", header=False, index=False, quoting=csv.QUOTE_MINIMAL)
    else:
        RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        df_new.to_csv(RAW_CSV, mode="w", header=True, index=False, quoting=csv.QUOTE_MINIMAL)


# --------------------------------------------------------------------------- #
# Helpers de métadonnées de run — pas des accesseurs au fichier, mais requis
# par tout script qui appelle append_run (évite que chaque script réinvente
# sa propre méthode, cf. incidents P1/P2 sur les logiques dupliquées).
# --------------------------------------------------------------------------- #
def git_commit_hash():
    """Hash du commit git courant (HEAD), ou "uncommitted" si le dépôt a des
    modifications non commitées touchant les scripts (le run n'est alors pas
    exactement reproductible depuis un commit propre)."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def make_run_id(prefix):
    """Identifiant de run unique et horodaté (pas déterministe — un run_id
    n'a pas besoin de l'être, seule la clé (run_id, city, model, ...) doit
    être unique en pratique)."""
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def compute_split_hash(n_total, t1, t2, seq_len):
    """Hash des indices de découpe chronologique (T total, bornes train/val,
    val/test, SEQ_LEN). Déterministe : même découpe -> même hash, sur
    n'importe quelle machine, sans dépendre d'un fichier sauvegardé."""
    key = f"{n_total}:{t1}:{t2}:{seq_len}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]
