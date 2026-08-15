"""
tests/test_stations.py — garde-fous pour la source unique de stations (P1).

Voir REVISION_BRIEF.md et AUDIT.md §1 : trois scripts indépendants
(10_external_baselines.py, 13_edge_pruning.py, e6_k_sensitivity.py /
e8_k_sensitivity_3seeds.py) avaient chacun leur propre
`EXCLUDE = {"madrid": {"MENDEZ ALVARO"}}`. Ces tests empêchent que le même
problème (liste de stations dupliquée / logique d'exclusion ad hoc) revienne.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from src.stations import load_stations, station_metadata, VALID_PURPOSES  # noqa: E402

CITIES = ["beijing", "london", "madrid"]

# Exemptions du scan "nom de station en dur" — deux catégories légitimes,
# distinctes du bug visé (copies de la liste POST-FILTRAGE en aval) :
#
# - 06_train_multistation.py : source canonique ORIGINALE (BEIJING_COORDS).
#   C'est le loader lui-même, pas une copie en aval. On ne le modifie pas
#   (règle du projet : ne jamais toucher la logique de calcul existante sans
#   demande explicite). configs/stations/beijing.yaml en est une copie
#   vérifiée identique (cf. test_yaml_matches_canonical_loader).
# - 01a-01g_*.py : scripts d'ACQUISITION de données, en amont du filtrage
#   qualité. 01b_download_london.py par ex. liste 12 codes CANDIDATS
#   (9 confirmés + 3 backups) pour le téléchargement LAQN — une liste plus
#   large et différente en nature de la liste POST-FILTRAGE (8 stations)
#   de configs/stations/london.yaml. Les deux ne doivent PAS être unifiées :
#   la liste de candidats au téléchargement précède et détermine le
#   filtrage, elle ne le duplique pas.
_ACQUISITION_SCRIPTS = {
    "01a_download_beijing.py", "01b_download_london.py",
    "01c_preprocess_london.py", "01d_download_london_weather.py",
    "01e_download_madrid.py", "01f_download_madrid_weather.py",
    "01g_preprocess_madrid.py",
}
_EXEMPT_FROM_HARDCODE_SCAN = {ROOT / "06_train_multistation.py"} | {
    ROOT / name for name in _ACQUISITION_SCRIPTS
}

_SKIP_DIRS = {"venv", "__pycache__", ".git", "node_modules", "tests"}
_SKIP_SUFFIXES_MARKERS = (".bak", "_BACKUP_")


def _pipeline_py_files():
    """Tous les .py du dépôt destinés au pipeline de MODÉLISATION (exclut
    venv/__pycache__/backups/tests). Les exemptions ciblées
    (_EXEMPT_FROM_HARDCODE_SCAN) sont appliquées séparément par chaque test,
    pas ici, pour que test_no_local_exclude_dict reste strict sur TOUS les
    fichiers (les scripts d'acquisition n'ont de toute façon aucune raison
    d'avoir un dict EXCLUDE)."""
    files = []
    for p in ROOT.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        if any(marker in p.name for marker in _SKIP_SUFFIXES_MARKERS):
            continue
        files.append(p)
    return files


def _all_station_names():
    names = []
    for city in CITIES:
        names += [s["name"] for s in station_metadata(city)]
    return names


# --------------------------------------------------------------------------- #
# 1. Aucun nom de station en dur dans un script (sauf le loader canonique 06)
# --------------------------------------------------------------------------- #
def test_no_hardcoded_station_names_outside_canonical_loader():
    station_names = _all_station_names()
    offenders = []
    for path in _pipeline_py_files():
        if path in _EXEMPT_FROM_HARDCODE_SCAN:
            continue
        text = path.read_text(errors="ignore")
        for name in station_names:
            # cherche le nom entre guillemets (littéral de chaîne), pas une
            # sous-chaîne fortuite dans un commentaire non lié
            if f'"{name}"' in text or f"'{name}'" in text:
                offenders.append((str(path.relative_to(ROOT)), name))
    assert not offenders, (
        "Nom(s) de station en dur trouvé(s) en dehors de la source unique "
        f"(configs/stations/*.yaml) : {offenders}. Utiliser "
        "src.stations.load_stations(city, purpose) à la place."
    )


# --------------------------------------------------------------------------- #
# 2. Aucun dict EXCLUDE local (le pattern à l'origine du bug MENDEZ ALVARO)
# --------------------------------------------------------------------------- #
def test_no_local_exclude_dict():
    pattern = re.compile(r"EXCLUDE\s*=\s*\{")
    offenders = []
    for path in _pipeline_py_files():
        if path == ROOT / "src" / "stations.py":
            continue  # docstring mentionne le pattern EXCLUDE={...} à titre d'exemple, pas une définition
        text = path.read_text(errors="ignore")
        if pattern.search(text):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, (
        f"Dict EXCLUDE local trouvé dans : {offenders}. C'est exactement le "
        "pattern qui a causé l'exclusion erronée de MENDEZ ALVARO dans 3 "
        "scripts indépendants — passer par src.stations.load_stations."
    )


# --------------------------------------------------------------------------- #
# 3. load_stations est le seul point d'entrée : les scripts qui filtrent des
#    stations doivent l'importer, pas réinventer une fonction équivalente.
# --------------------------------------------------------------------------- #
CONSUMERS = [
    ROOT / "10_external_baselines.py",
    ROOT / "13_edge_pruning.py",
    ROOT / "e6_k_sensitivity.py",
    ROOT / "e8_k_sensitivity_3seeds.py",
    ROOT / "05_compute_heterogeneity_v2.py",
]


@pytest.mark.parametrize("path", CONSUMERS, ids=lambda p: p.name)
def test_consumer_scripts_import_load_stations(path):
    text = path.read_text(errors="ignore")
    assert "from src.stations import" in text, (
        f"{path.name} devrait importer depuis src.stations (load_stations "
        "ou station_metadata) plutôt que de définir sa propre logique de "
        "filtrage de stations."
    )


def test_no_competing_load_stations_definition():
    """Une seule fonction `def load_stations(` dans tout le dépôt (src/stations.py)."""
    offenders = []
    for path in _pipeline_py_files():
        if path == ROOT / "src" / "stations.py":
            continue
        if re.search(r"^def load_stations\(", path.read_text(errors="ignore"), re.M):
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"Définition concurrente de load_stations dans : {offenders}"


# --------------------------------------------------------------------------- #
# 4. Comptage Madrid : 7 en mode benchmark, 6 en mode heterophily
# --------------------------------------------------------------------------- #
def test_madrid_station_counts():
    bench = load_stations("madrid", "benchmark")
    het = load_stations("madrid", "heterophily")
    assert len(bench) == 7, f"Madrid benchmark devrait avoir 7 stations, a {len(bench)}"
    assert len(het) == 6, f"Madrid heterophily devrait avoir 6 stations, a {len(het)}"
    diff = set(bench) - set(het)
    assert len(diff) == 1, f"Une seule station devrait différer entre les deux modes, {diff}"


@pytest.mark.parametrize("city,expected", [("beijing", 12), ("london", 8), ("madrid", 7)])
def test_benchmark_counts_per_city(city, expected):
    assert len(load_stations(city, "benchmark")) == expected


@pytest.mark.parametrize("city", CITIES)
def test_heterophily_subset_of_benchmark(city):
    bench = set(load_stations(city, "benchmark"))
    het = set(load_stations(city, "heterophily"))
    assert het <= bench, f"{city}: heterophily doit être un sous-ensemble de benchmark"


def test_invalid_purpose_raises():
    with pytest.raises(ValueError):
        load_stations("madrid", "not_a_real_purpose")


# --------------------------------------------------------------------------- #
# 5. L'ordre renvoyé par load_stations(city, "benchmark") == ordre canonique
#    des loaders de 06_train_multistation.py (l'indexation des nœuds du
#    graphe en dépend directement).
# --------------------------------------------------------------------------- #
def test_yaml_matches_canonical_loader():
    import importlib.util
    import io
    import numpy as np
    from contextlib import redirect_stdout

    spec = importlib.util.spec_from_file_location("bench", str(ROOT / "06_train_multistation.py"))
    b = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = b
    spec.loader.exec_module(b)

    loaders = {
        "beijing": lambda: b.load_beijing_data(str(ROOT / "data/beijing_real/PRSA_Data_20130301-20170228")),
        "london": b.load_london_data,
        "madrid": b.load_madrid_data,
    }
    for city, loader in loaders.items():
        with redirect_stdout(io.StringIO()):
            loader()
        canonical = list(b.STATION_NAMES)
        assert load_stations(city, "benchmark") == canonical, (
            f"{city}: configs/stations/{city}.yaml (mode benchmark) != ordre "
            f"canonique du loader — l'indexation des nœuds du graphe serait "
            f"incohérente."
        )
