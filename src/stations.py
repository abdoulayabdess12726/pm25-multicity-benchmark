"""
src/stations.py — source unique pour les listes de stations par ville.

Seule voie d'accès autorisée aux listes de stations retenues après filtrage
qualité. Aucun script d'expérience ne doit contenir de nom de station en dur
ni de logique d'exclusion ad hoc (EXCLUDE = {...}) — voir REVISION_BRIEF.md
et AUDIT.md §1.

La source de vérité est configs/stations/{city}.yaml. Cette fonction ne
modifie pas le chargement des données lui-même (06_train_multistation.py
reste le loader canonique, inchangé) — elle sert à filtrer/valider quels
noms de station un script en aval doit garder pour un usage donné.
"""
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "stations"
VALID_PURPOSES = {"benchmark", "heterophily"}
_PURPOSE_KEY = {
    "benchmark": "include_in_benchmark",
    "heterophily": "include_in_heterophily_analysis",
}
_CACHE = {}


def _load_yaml(city):
    if city not in _CACHE:
        path = CONFIG_DIR / f"{city}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Pas de config stations pour '{city}': {path}")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        if cfg.get("city") != city:
            raise ValueError(f"{path}: champ 'city'={cfg.get('city')!r} != '{city}'")
        _CACHE[city] = cfg
    return _CACHE[city]


def load_stations(city, purpose):
    """Renvoie la liste ordonnée (alphabétique) des noms de station de `city`
    retenues pour `purpose` ∈ {"benchmark", "heterophily"}.

    - "benchmark"   : stations à inclure dans le benchmark de prévision
                       (graphe, entraînement GNN, agrégat R²).
    - "heterophily" : stations à inclure dans l'analyse station-level
                       d'hétérophilie (indice h_i, Table 6 ΔR² par station).

    L'ordre alphabétique renvoyé est celui utilisé par les loaders canoniques
    de 06_train_multistation.py (load_beijing_data / load_london_data /
    load_madrid_data, qui trient déjà alphabétiquement ou utilisent un dict
    déjà en ordre alphabétique) — NE PAS changer cet ordre sans vérifier
    l'indexation des nœuds du graphe partout en aval.
    """
    if purpose not in VALID_PURPOSES:
        raise ValueError(f"purpose doit être dans {VALID_PURPOSES}, reçu {purpose!r}")
    cfg = _load_yaml(city)
    key = _PURPOSE_KEY[purpose]
    names = [s["name"] for s in cfg["stations"] if s.get(key, False)]
    return sorted(names)


def station_metadata(city):
    """Liste complète des dicts station (name, lat, lon, les deux booléens,
    exclusion_reason le cas échéant) — pour les scripts qui ont besoin des
    coordonnées plutôt que de la seule liste filtrée."""
    return list(_load_yaml(city)["stations"])
