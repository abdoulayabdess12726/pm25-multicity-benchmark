#!/usr/bin/env python3
"""
scripts/migrate_raw_results.py — migration one-shot des résultats existants
vers results/raw_results.csv (P3, REVISION_BRIEF.md).

Un lanceur PAR SOURCE, documenté. Chaque source correspond à un fichier
existant produit avant l'instrumentation live d'append_run. Les lignes dont
la provenance est incertaine (Madrid 6-stations pré-correctif P1, ligne
UNRECOVERABLE de sensitivity_k_canonical.csv) portent un `provenance_note`
explicite — jamais silencieusement migrées comme si elles étaient fiables.

Sources dérivées volontairement NON migrées (évite la duplication) :
  - results/per_station_seed_topology.csv (Table 6) : recalculable à 100%
    depuis le JSON canonique déjà migré (source 1).
  - results/statistical_analysis/* (Table 4/5) : idem, recalculé depuis
    les mêmes données sources.
  - results/heterogeneity_index_v2.csv (Table 1) : pas un résultat de
    modèle (pas de seed/rmse/mae), hors du schéma raw_results.
  - results/sensitivity_k_canonical.csv, lignes k=5 : doublon exact des
    lignes GCN-Transformer k=5 déjà migrées depuis le JSON canonique
    (source 1) — seule la ligne UNRECOVERABLE (k=3) est unique.

Usage : python scripts/migrate_raw_results.py
"""
import importlib.util
import io
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.results_io import append_run, compute_split_hash

CITIES = ["beijing", "london", "madrid"]
N_STATIONS_CANONICAL = {"beijing": 12, "london": 8, "madrid": 7}
TS = time.strftime("%Y-%m-%dT%H:%M:%S")
GIT_COMMIT = "migration_pre_P3"  # ces données pré-existent à l'instrumentation live

stats = {"rows": 0, "suspect_or_unrecoverable": 0, "sources": {}}
_bench = None
_split_cache = {}


def load_bench():
    global _bench
    if _bench is None:
        spec = importlib.util.spec_from_file_location("bench", str(ROOT / "06_train_multistation.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["bench"] = mod
        spec.loader.exec_module(mod)
        _bench = mod
    return _bench


def get_split_hash(city, seq_len=24):
    """Charge les données une seule fois par ville (mise en cache) pour
    calculer un split_hash réel (T, t1, t2) — pas une valeur vide, la
    découpe chronologique est intégralement reconstructible."""
    if city in _split_cache:
        return _split_cache[city]
    b = load_bench()
    with redirect_stdout(io.StringIO()):
        if city == "beijing":
            ret = b.load_beijing_data(str(ROOT / "data/beijing_real/PRSA_Data_20130301-20170228"))
        elif city == "london":
            ret = b.load_london_data()
        else:
            ret = b.load_madrid_data()
    data = ret[0] if isinstance(ret, (tuple, list)) else ret
    T = len(np.asarray(data))
    t1, t2 = int(0.70 * T), int(0.85 * T)
    h = compute_split_hash(T, t1, t2, seq_len)
    _split_cache[city] = h
    return h


def _row(city, model, topology, k, seed, shash, n_stations, station, rmse, mae, r2,
         config_path, note, variant="", keep_frac="", run_id_suffix=""):
    return dict(
        city=city, model=model, variant=variant, topology=topology, k=k,
        keep_frac=keep_frac, seed=seed,
        checkpoint_id="no_checkpoint_saved", split_hash=shash, n_stations=n_stations,
        station=station, split="test", rmse=rmse, mae=mae, r2=r2,
        run_id=f"migrated_{config_path.replace('.py', '')}_{city}{run_id_suffix}",
        config_path=config_path, git_commit=GIT_COMMIT, timestamp=TS,
        provenance_note=note)


def _emit(rows, label):
    if not rows:
        print(f"[{label}] 0 ligne — rien à migrer")
        return
    append_run(rows)
    n_flagged = sum(1 for r in rows if r["provenance_note"])
    stats["rows"] += len(rows)
    stats["suspect_or_unrecoverable"] += n_flagged
    stats["sources"][label] = dict(rows=len(rows), flagged=n_flagged)
    print(f"[{label}] +{len(rows)} lignes ({n_flagged} avec provenance_note)")


# --------------------------------------------------------------------------- #
# 1. Benchmark canonique — Table 2 (results/{city}/multistation_results.json)
# --------------------------------------------------------------------------- #
def migrate_canonical():
    rows = []
    for city in CITIES:
        path = ROOT / f"results/{city}/multistation_results.json"
        js = json.loads(path.read_text())
        n_stations = js["n_stations"]
        seeds = js["seeds"]
        shash = get_split_hash(city, js["config"]["seq_len"])
        graphs = js["graphs"]

        # Linear-Transformer ET GCN-Transformer : une ligne par topologie
        # (même si Linear-Transformer est indépendant de la topologie, ses
        # valeurs sont dupliquées une fois par topologie — comportement
        # identique à ce qu'écrit une exécution live de 06_train_multistation.py,
        # cf. CHANGELOG_TABLES.md). Noms normalisés "+"->"-" (convention
        # raw_results.csv ; le JSON garde "GCN+Transformer" en interne).
        for topo, gdata in graphs.items():
            for model_key, model in [("Linear+Transformer", "Linear-Transformer"),
                                     ("GCN+Transformer", "GCN-Transformer")]:
                block = gdata[model_key]
                psa = gdata["per_station_all_seeds"][model_key]
                k = 5 if model == "GCN-Transformer" else ""
                for seed_str, per_st in psa.items():
                    seed = int(seed_str)
                    idx = seeds.index(seed)
                    for station, m in per_st.items():
                        rows.append(_row(city, model, topo, k, seed, shash, n_stations,
                                         station, m["RMSE"], m["MAE"], m["R2"],
                                         "06_train_multistation.py", ""))
                    rows.append(_row(city, model, topo, k, seed, shash, n_stations,
                                     "__aggregate__", block["RMSE"][idx], block["MAE"][idx],
                                     block["R2"][idx], "06_train_multistation.py", ""))
    _emit(rows, "canonical_benchmark (Table 2, 06_train_multistation.py)")


# --------------------------------------------------------------------------- #
# 2. Baselines externes — Table 3 (results/external_baselines.csv)
# --------------------------------------------------------------------------- #
def migrate_external_baselines():
    path = ROOT / "results/external_baselines.csv"
    if not path.exists():
        print("[external_baselines] fichier absent")
        return
    df = pd.read_csv(path, dtype={"seed": str})
    rows = []
    n_real = {"beijing": 12, "london": 8, "madrid": 6}  # Madrid réellement 6 dans ce fichier (MENDEZ absente)
    for city in CITIES:
        shash = get_split_hash(city)
        sub = df[df.city == city]
        note = ("SUSPECT_6STATION: MENDEZ ALVARO absente (ancien protocole "
                "pré-correctif P1) — non ré-agrégeable pour ARIMA/XGBoost/LSTM/"
                "Persistence sans re-run, cf. CHANGELOG_TABLES.md."
                if city == "madrid" else "")
        for _, r in sub.iterrows():
            rows.append(_row(city, r["model"], "", "", r["seed"], shash, n_real[city],
                             r["station"], r["RMSE"], r["MAE"], r["R2"],
                             "10_external_baselines.py", note))
    _emit(rows, "external_baselines (Table 3, 10_external_baselines.py)")


# --------------------------------------------------------------------------- #
# 3. Élagage des arêtes — §6.2.1 (results/edge_pruning.csv)
# --------------------------------------------------------------------------- #
def migrate_edge_pruning():
    path = ROOT / "results/edge_pruning.csv"
    if not path.exists():
        print("[edge_pruning] fichier absent")
        return
    df = pd.read_csv(path)
    rows = []
    n_real = {"beijing": 12, "london": 8, "madrid": 6}
    for city in CITIES:
        shash = get_split_hash(city)
        sub = df[df.city == city]
        note = ("SUSPECT_6STATION: MENDEZ ALVARO absente (ancien protocole "
                "pré-correctif P1) — jamais évaluée, aucune métrique par-station "
                "persistée, irrécupérable sans ré-entraînement (E10, P5)."
                if city == "madrid" else "")
        for _, r in sub.iterrows():
            rows.append(_row(city, "GCN-Transformer", "distance", "", r["seed"],
                             shash, n_real[city], r["station"], r["RMSE"], r["MAE"], r["R2"],
                             "13_edge_pruning.py", note, keep_frac=r["keep_frac"]))
    _emit(rows, "edge_pruning (§6.2.1, 13_edge_pruning.py)")


# --------------------------------------------------------------------------- #
# 4. Sensibilité k — Table 7 (results/e6_k_sensitivity.csv, 54 lignes)
# --------------------------------------------------------------------------- #
def migrate_k_sensitivity():
    path = ROOT / "results/e6_k_sensitivity.csv"
    if not path.exists():
        print("[k_sensitivity] fichier absent")
        return
    df = pd.read_csv(path, dtype={"seed": str})
    rows = []
    n_real = {"beijing": 12, "london": 8, "madrid": 7}
    for _, r in df.iterrows():
        city = r["city"]
        k = int(r["k"])
        madrid_unusable = (city == "madrid") and (k in (3, 8))
        note = ("UNRECOVERABLE_6STATION: MENDEZ ALVARO jamais évaluée pour cette "
                "cellule (exclusion au chargement, ancien code) — aucune donnée "
                "par-station persistée, ré-entraînement requis (E9, P5), cf. "
                "CHANGELOG_TABLES.md." if madrid_unusable else "")
        shash = get_split_hash(city)
        rows.append(_row(city, "GCN-Transformer", r["topology"], k, r["seed"], shash,
                         6 if madrid_unusable else n_real[city], "__aggregate__",
                         "", "", r["R2_gcn"],
                         "e6_k_sensitivity.py/e8_k_sensitivity_3seeds.py", note))
    _emit(rows, "k_sensitivity (Table 7, e6_k_sensitivity.csv)")


# --------------------------------------------------------------------------- #
# 5. Contrôles diagnostiques — Table 9 (results/diagnostics.csv)
# --------------------------------------------------------------------------- #
def migrate_diagnostics():
    path = ROOT / "results/diagnostics.csv"
    if not path.exists():
        print("[diagnostics] fichier absent")
        return
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        city = r["city"]
        shash = get_split_hash(city)
        note = ("E4/E5 diagnostics; RMSE/MAE et per-station non stockés par "
                "ce script (seul R2 agrégé).")
        rows.append(_row(city, "GCN-Transformer", r["topology"], 5,
                         r["seed"], shash, N_STATIONS_CANONICAL[city], "__aggregate__",
                         "", "", r["gcn_r2"], "11_diagnostics.py", note,
                         variant=r["experiment"]))
        if r["experiment"] == "no_meteorology":
            rows.append(_row(city, "Linear-Transformer", r["topology"], "",
                             r["seed"], shash, N_STATIONS_CANONICAL[city], "__aggregate__",
                             "", "", r["lin_r2"], "11_diagnostics.py", note,
                             variant="no_meteorology"))
    _emit(rows, "diagnostics (Table 9, 11_diagnostics.py)")


# --------------------------------------------------------------------------- #
# 6. SOTA GNN — Table 3 (results/table3_sota_baselines.csv)
# --------------------------------------------------------------------------- #
def migrate_sota():
    path = ROOT / "results/table3_sota_baselines.csv"
    if not path.exists():
        print("[sota] fichier absent")
        return
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        city = r["city"]
        shash = get_split_hash(city)
        note = "Pas de per-station stocké pour ce run historique (capture ajoutée après coup en P3)."
        rows.append(_row(city, r["model"], "correlation", 5, int(r["seed"]), shash,
                         N_STATIONS_CANONICAL[city], "__aggregate__", "", "", r["R2_aggregate"],
                         "14_sota_baselines.py", note))
    _emit(rows, "sota_baselines (Table 3, STGCN/GraphWaveNet, 14_sota_baselines.py)")


# --------------------------------------------------------------------------- #
# 7. sensitivity_k_canonical.csv — UNIQUEMENT la ligne UNRECOVERABLE
#    (les lignes k=5 sont un doublon exact de la source 1, non réinjectées)
# --------------------------------------------------------------------------- #
def migrate_sensitivity_k_canonical_unrecoverable():
    path = ROOT / "results/sensitivity_k_canonical.csv"
    if not path.exists():
        print("[sensitivity_k_canonical] fichier absent")
        return
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        if "UNRECOVERABLE" not in str(r["source"]):
            continue  # doublon des lignes k=5 déjà migrées (source 1)
        city = r["city"]
        shash = get_split_hash(city)
        rows.append(_row(
            city, "GCN-Transformer", r["topology"], int(r["k"]), "unknown_3seed_agg", shash,
            N_STATIONS_CANONICAL[city], "__aggregate__", "", "", r["gcn_r2_mean"],
            "08_sensitivity_k.py",
            "UNRECOVERABLE: agrégat de 3 seeds sans attribution individuelle possible "
            "(valeurs par-seed jamais persistées, cf. CHANGELOG_TABLES.md). "
            "seed='unknown_3seed_agg' N'EST PAS un seed réel — ne pas traiter comme 42/123/777."))
    _emit(rows, "sensitivity_k_canonical UNRECOVERABLE (08_sensitivity_k.py)")


def main():
    if (ROOT / "results" / "raw_results.csv").exists():
        raise SystemExit(
            "results/raw_results.csv existe déjà — ce script de migration one-shot "
            "refuse de s'exécuter sur un fichier non vide (append_run refusera de "
            "toute façon chaque doublon, mais on s'arrête proprement avant plutôt "
            "que de spammer des erreurs)."
        )
    migrate_canonical()
    migrate_external_baselines()
    migrate_edge_pruning()
    migrate_k_sensitivity()
    migrate_diagnostics()
    migrate_sota()
    migrate_sensitivity_k_canonical_unrecoverable()

    print(f"\n{'='*60}")
    print(f"TOTAL migré : {stats['rows']} lignes")
    print(f"  dont provenance_note (suspect/unrecoverable) : {stats['suspect_or_unrecoverable']}")
    for label, s in stats["sources"].items():
        print(f"  - {label}: {s['rows']} lignes ({s['flagged']} flaggées)")


if __name__ == "__main__":
    main()
