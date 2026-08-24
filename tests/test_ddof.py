"""
tests/test_ddof.py — garde-fous pour la convention ddof=1 unique (P2).

Voir REVISION_BRIEF.md : Table 4 en ddof=0, Table 7 en ddof=1 — pas un choix
explicite incohérent, un défaut IMPLICITE différent entre numpy (`.std()` →
ddof=0) et pandas (`.std()` → ddof=1). `agg_mean_std` (src/stats.py)
neutralise ce défaut : ddof=1 toujours explicite. Ces tests empêchent qu'un
appel `.std()`/`np.std()` nu ne réapparaisse dans un chemin d'agrégation
INTER-SEEDS (3 seeds : 42, 123, 777).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.stats import agg_mean_std  # noqa: E402

# --------------------------------------------------------------------------- #
# 1. Correction de agg_mean_std elle-même
# --------------------------------------------------------------------------- #
def test_agg_mean_std_matches_ddof1_formula():
    values = [0.4853, 0.4962, 0.4966]  # Madrid GCN k=5, 3 seeds (valeurs réelles)
    mean, std = agg_mean_std(values)
    assert mean == pytest.approx(np.mean(values), abs=1e-12)
    assert std == pytest.approx(np.std(values, ddof=1), abs=1e-12)
    assert std != pytest.approx(np.std(values, ddof=0), abs=1e-6)  # les deux ddof divergent bien


def test_agg_mean_std_single_value_returns_zero_std():
    mean, std = agg_mean_std([0.813])
    assert mean == pytest.approx(0.813)
    assert std == 0.0


def test_agg_mean_std_input_type_does_not_change_ddof():
    """Le ddof ne doit JAMAIS dépendre du type d'entrée (liste, array numpy,
    Series pandas) — c'est exactement le défaut implicite qui a causé le bug."""
    import pandas as pd
    values = [0.30, 0.32, 0.28]
    m_list, s_list = agg_mean_std(values)
    m_arr, s_arr = agg_mean_std(np.array(values))
    m_ser, s_ser = agg_mean_std(pd.Series(values))
    assert s_list == pytest.approx(s_arr) == pytest.approx(s_ser)


def test_agg_mean_std_rejects_empty():
    with pytest.raises(ValueError):
        agg_mean_std([])


# --------------------------------------------------------------------------- #
# 2. Vérification numérique du cas de référence (London, Table 4 vs Table 7)
# --------------------------------------------------------------------------- #
def test_london_table4_table7_reconciliation():
    """Valeurs citées dans la tâche P2 : Table 4 (ancien, ddof=0) donnait
    0.021 (distance) / 0.005 (corrélation) pour Londres ; en ddof=1 elles
    doivent tomber sur les valeurs de la Table 7 (k=5), 0.026 / 0.006."""
    import json
    js = json.loads((ROOT / "results/london/multistation_results.json").read_text())
    for topo, old_ddof0_std, expected_ddof1_std in [
        ("distance", 0.021, 0.026),
        ("correlation", 0.005, 0.006),
    ]:
        gcn = js["graphs"][topo]["GCN+Transformer"]["R2"]
        lin = js["graphs"][topo]["Linear+Transformer"]["R2"]
        delta = np.array(gcn) - np.array(lin)
        assert np.std(delta, ddof=0) == pytest.approx(old_ddof0_std, abs=5e-4), (
            f"{topo}: le ddof=0 stocké dans le JSON ne correspond plus à la "
            "valeur citée dans la tâche — vérifier la source avant de conclure."
        )
        _, new_std = agg_mean_std(delta)
        assert round(new_std, 3) == pytest.approx(expected_ddof1_std, abs=1e-3)
        # relation exacte ddof=0 -> ddof=1 pour n=3 : facteur sqrt(3/2)
        assert new_std == pytest.approx(np.std(delta, ddof=0) * np.sqrt(3 / 2), abs=1e-9)


def test_table7_k5_matches_table4_exactly():
    """Table 7 (k=5) et Table 4 doivent être calculées sur EXACTEMENT les
    mêmes 3 valeurs GCN-Linear (k=5 = benchmark canonique) — même mean,
    même std, ddof=1 partout.

    Vérifié sur le PIPELINE RÉEL (raw_results.csv -> regenerate_tables.py),
    pas sur results/{city}/multistation_results.json ni sur
    results/table7_k_sensitivity.csv : ces deux fichiers sont des artefacts
    intermédiaires écrits directement par des scripts historiques
    (06_train_multistation.py, e8_k_sensitivity_3seeds.py) et déconnectés de
    la régénération — trouvé le 2026-08-24 en corrigeant Madrid/correlation
    (cause racine « tri NaN de build_correlation_graph », P11.6) : les deux
    étaient gelés au MÊME état intermédiaire (-0.3795, une valeur antérieure
    à la version finale -0.4144 actuellement dans le manuscrit), donc
    cohérents entre eux sans être corrects — ce test passait par coïncidence,
    pas par protection réelle. Table 7(k=5)==Table 4 sur le pipeline réel est
    déjà vérifié par regenerate_tables.assert_t7_k5_equals_t4 ; ce test
    délègue à la même fonction plutôt que dupliquer une comparaison sur des
    fichiers qui peuvent diverger silencieusement du pipeline."""
    import scripts.regenerate_tables as rt
    rt.ASSERTIONS.clear()
    df = rt.load()
    rt.assert_t7_k5_equals_t4(df)
    for a in rt.ASSERTIONS:
        assert a["status"] == "PASS", f"{a['name']}: {a['detail']}"
    rt.ASSERTIONS.clear()


# --------------------------------------------------------------------------- #
# 3. Aucun .std()/np.std() nu en dehors de agg_mean_std, hors périmètre exempté
# --------------------------------------------------------------------------- #
_ACQUISITION_SCRIPTS = {
    "01a_download_beijing.py", "01b_download_london.py",
    "01c_preprocess_london.py", "01d_download_london_weather.py",
    "01e_download_madrid.py", "01f_download_madrid_weather.py",
    "01g_preprocess_madrid.py", "01h_download_czt.py",
    "01i_preprocess_czt.py",
}
# Fichiers entiers légitimement hors périmètre (dispersion qui n'est PAS un
# agrégat inter-seeds) :
#   - scripts d'acquisition 01*.py : filtres qualité sur la variance d'une
#     série temporelle, sans rapport avec les seeds.
#   - 05_compute_heterogeneity_v2.py : dispersion intra-série (n grand),
#     commentée sur place, cf. REVISION_BRIEF.md.
#   - src/stats.py : l'implémentation elle-même.
_EXEMPT_FILES = {ROOT / name for name in _ACQUISITION_SCRIPTS} | {
    ROOT / "05_compute_heterogeneity_v2.py",
    ROOT / "src" / "stats.py",
}
# Lignes spécifiques hors périmètre au sein d'un fichier par ailleurs corrigé
# (dispersion inter-STATION, pas inter-seed) :
_EXEMPT_LINE_SUBSTRINGS = [
    (ROOT / "07_statistical_analysis.py", "diff.std()"),
    # Figure 1 (fig1()) : z-score d'une série synthétique fabriquée pour
    # l'illustration conceptuelle (pas de données réelles, pas de seeds,
    # aucun agrégat inter-seed) — hors périmètre de la convention ddof.
    (ROOT / "regenerate_figures.py", "y = (y - y.mean()) / y.std()"),
]

_SKIP_DIRS = {"venv", "__pycache__", ".git", "node_modules", "tests", "external"}
_SKIP_SUFFIXES_MARKERS = (".bak", "_BACKUP_")  # gitignorés, hors pipeline actif (cf. test_stations.py)
_STD_CALL = re.compile(r"\.std\(|np\.std\(|statistics\.stdev|statistics\.pstdev")


def _pipeline_py_files():
    files = []
    for p in ROOT.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        if any(marker in p.name for marker in _SKIP_SUFFIXES_MARKERS):
            continue
        files.append(p)
    return files


def test_no_raw_std_outside_agg_mean_std():
    offenders = []
    for path in _pipeline_py_files():
        if path in _EXEMPT_FILES:
            continue
        for lineno, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if not _STD_CALL.search(line):
                continue
            if any(path == p and sub in line for p, sub in _EXEMPT_LINE_SUBSTRINGS):
                continue
            offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "Appel .std()/np.std() nu trouvé hors agg_mean_std et hors périmètre "
        f"exempté : {offenders}. Router vers src.stats.agg_mean_std, ou "
        "documenter l'exemption (commentaire sur place + entrée dans "
        "_EXEMPT_FILES/_EXEMPT_LINE_SUBSTRINGS ci-dessus) si ce n'est "
        "vraiment pas un agrégat inter-seeds."
    )
