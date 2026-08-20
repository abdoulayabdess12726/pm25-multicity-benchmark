"""
Régression : build_correlation_graph ne doit jamais laisser un candidat à
corrélation indéfinie (NaN) consommer un rang de voisin sans le réattribuer.

Incident (P5, cf. CHANGELOG_TABLES.md) : `np.argsort(corr[i])[::-1]` plaçait
les NaN en tête du tri décroissant ; le candidat était ensuite éliminé par
le seuil `corr>0` sans que le rang libéré soit réattribué au candidat réel
suivant. Madrid perdait 11/35 arêtes à k=5 (MENDEZ ALVARO totalement isolée,
les 6 autres stations à 4 voisins réels au lieu de 5) ; ce test aurait
attrapé le bug avant qu'il n'affecte Table 2/3/4/6/7.

Pour chaque ville × k testé, le nombre d'arêtes réelles doit égaler la
somme, sur toutes les stations, de min(k_eff, nb de candidats à
corrélation FINIE pour cette station) — pas une formule uniforme n×k_eff,
qui serait fausse dès qu'une station a des corrélations indéfinies.
"""
import importlib.util
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_bench():
    spec = importlib.util.spec_from_file_location("bench", str(ROOT / "06_train_multistation.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench"] = mod
    with redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def _load_city_data(b, city):
    with redirect_stdout(io.StringIO()):
        if city == "beijing":
            ret = b.load_beijing_data(str(ROOT / "data/beijing_real/PRSA_Data_20130301-20170228"))
        elif city == "london":
            ret = b.load_london_data()
        else:
            ret = b.load_madrid_data()
    data = ret[0] if isinstance(ret, (tuple, list)) else ret
    return np.asarray(data, dtype=np.float32)


def _expected_edge_count(data, train_len, k):
    """Compte attendu station-par-station, en excluant les candidats NaN
    (pas une formule uniforme n×k_eff)."""
    pm25 = data[:train_len, :, 0]  # PM2.5 toujours en colonne 0 (cf. 06)
    corr = np.corrcoef(pm25.T)
    np.fill_diagonal(corr, -np.inf)
    n = corr.shape[0]
    k_eff = min(k, n - 1)
    total = 0
    for i in range(n):
        valid = np.isfinite(corr[i])
        n_take = min(k_eff, int(valid.sum()))
        # threshold=0.0 par défaut : ne compte que les candidats retenus à corrélation > 0
        masked = np.where(valid, corr[i], -np.inf)
        neighbors = np.argsort(masked)[::-1][:n_take]
        total += int((corr[i, neighbors] > 0.0).sum())
    return total


@pytest.mark.parametrize("city", ["beijing", "london", "madrid"])
@pytest.mark.parametrize("k", [3, 5, 8])
def test_correlation_graph_edge_count_matches_valid_neighbors(city, k):
    b = _load_bench()
    data = _load_city_data(b, city)
    train_len = int(0.70 * len(data))

    with redirect_stdout(io.StringIO()):
        ei, _ = b.build_correlation_graph(data[:train_len], k=k)

    actual = ei.shape[1]
    expected = _expected_edge_count(data, train_len, k)
    assert actual == expected, (
        f"{city}/k={k} : {actual} arêtes réelles, {expected} attendues "
        "(un candidat NaN a peut-être consommé un rang de voisin sans être "
        "réattribué — cf. incident build_correlation_graph, CHANGELOG_TABLES.md)"
    )


def test_madrid_mendez_alvaro_isolated_but_others_get_full_valid_neighbors():
    """Cas nominal documenté : MENDEZ ALVARO (toutes corrélations train
    indéfinies) doit être isolée (0 arête, dans les deux sens) ; les 6
    autres stations doivent obtenir leurs 5 voisins réels (pas 4)."""
    b = _load_bench()
    data = _load_city_data(b, "madrid")
    train_len = int(0.70 * len(data))
    names = list(b.STATION_NAMES)
    mendez_idx = names.index("MENDEZ ALVARO")

    with redirect_stdout(io.StringIO()):
        ei, _ = b.build_correlation_graph(data[:train_len], k=5)
    ei_np = ei.numpy()

    assert (ei_np[0] == mendez_idx).sum() == 0, "MENDEZ ALVARO ne doit avoir aucune arête sortante"
    assert (ei_np[1] == mendez_idx).sum() == 0, "MENDEZ ALVARO ne doit avoir aucune arête entrante"

    for i, name in enumerate(names):
        if i == mendez_idx:
            continue
        out_deg = int((ei_np[0] == i).sum())
        assert out_deg == 5, (
            f"{name} : degré sortant {out_deg}, attendu 5 (5 voisins réels "
            "disponibles hors MENDEZ ALVARO, aucun rang ne doit être perdu)"
        )
