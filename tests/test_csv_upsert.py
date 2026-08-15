"""
tests/test_csv_upsert.py — garde-fou pour la fusion CSV par clé exacte (P2).

Incident : `08_sensitivity_k.py` fusionnait en supprimant TOUTES les lignes
de chaque ville listée dans `--cities` avant de concaténer les nouvelles
lignes — un ré-run ciblé sur k=5 seul a effacé silencieusement la ligne
`beijing k=3 distance` déjà calculée. Voir CHANGELOG_TABLES.md.
"""
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.csv_upsert import upsert_rows  # noqa: E402


def test_upsert_does_not_drop_untargeted_rows_same_city(tmp_path):
    """Le cas exact de l'incident : ré-écrire k=5 pour beijing ne doit pas
    effacer la ligne k=3 déjà présente pour beijing."""
    csv = tmp_path / "sensitivity.csv"
    initial = pd.DataFrame([
        dict(city="beijing", k=3, topology="distance", delta_r2_mean=-0.0134),
        dict(city="beijing", k=5, topology="distance", delta_r2_mean=-0.0172),
        dict(city="london", k=5, topology="distance", delta_r2_mean=-0.3754),
    ])
    upsert_rows(csv, initial, key_cols=["city", "k", "topology"])

    # ré-écriture ciblée : seulement (beijing, k=5, distance)
    refresh = pd.DataFrame([
        dict(city="beijing", k=5, topology="distance", delta_r2_mean=-0.0173),
    ])
    result = upsert_rows(csv, refresh, key_cols=["city", "k", "topology"])

    keys = set(map(tuple, result[["city", "k", "topology"]].values.tolist()))
    assert ("beijing", 3, "distance") in keys, (
        "La ligne beijing/k=3/distance a été effacée par une écriture qui ne "
        "la ciblait pas — régression de l'incident sensitivity_k_canonical.csv"
    )
    assert ("london", 5, "distance") in keys, "Ville non ciblée effacée"
    # la ligne ciblée doit bien être mise à jour
    row = result[(result.city == "beijing") & (result.k == 5) & (result.topology == "distance")]
    assert row.delta_r2_mean.iloc[0] == pytest.approx(-0.0173)
    assert len(result) == 3  # aucune ligne perdue, aucun doublon


def test_upsert_replaces_exact_key_match_only():
    """Une ligne dont la clé correspond exactement est remplacée, pas dupliquée."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        csv = Path(d) / "x.csv"
        initial = pd.DataFrame([dict(city="madrid", k=8, topology="correlation", v=1.0)])
        upsert_rows(csv, initial, key_cols=["city", "k", "topology"])
        updated = pd.DataFrame([dict(city="madrid", k=8, topology="correlation", v=2.0)])
        result = upsert_rows(csv, updated, key_cols=["city", "k", "topology"])
        assert len(result) == 1
        assert result.v.iloc[0] == 2.0


def test_upsert_new_rows_append_without_key_collision():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        csv = Path(d) / "x.csv"
        a = pd.DataFrame([dict(city="madrid", k=3, topology="distance", v=1.0)])
        upsert_rows(csv, a, key_cols=["city", "k", "topology"])
        b = pd.DataFrame([dict(city="madrid", k=8, topology="distance", v=2.0)])
        result = upsert_rows(csv, b, key_cols=["city", "k", "topology"])
        assert len(result) == 2


# --------------------------------------------------------------------------- #
# Sites réels migrés vers upsert_rows — un par script trouvé par la recherche
# élargie (pas seulement les 2 signalés initialement : 11_diagnostics.py a été
# trouvé en élargissant la recherche au motif, pas au nom de fichier).
# --------------------------------------------------------------------------- #
_MIGRATED_SITES = {
    "08_sensitivity_k.py": ["city", "k", "topology"],
    "10_external_baselines.py": ["city", "model", "station", "seed"],
    "13_edge_pruning.py": ["city", "keep_frac", "seed", "station"],
    "11_diagnostics.py": ["city", "topology", "experiment", "seed"],
}


@pytest.mark.parametrize("filename,key_cols", _MIGRATED_SITES.items(), ids=list(_MIGRATED_SITES))
def test_site_uses_upsert_rows(filename, key_cols):
    text = (ROOT / filename).read_text(errors="ignore")
    assert "from src.csv_upsert import upsert_rows" in text, (
        f"{filename} devrait importer upsert_rows plutôt que fusionner par ville à la main"
    )
    assert "upsert_rows(" in text, f"{filename} importe upsert_rows mais ne l'utilise pas"
    for col in key_cols:
        assert f'"{col}"' in text, (
            f"{filename}: colonne-clé attendue {col!r} introuvable près de l'appel upsert_rows "
            "— vérifier que key_cols correspond bien au schéma du CSV de ce script"
        )


def test_no_bulk_city_filter_before_csv_rewrite_anywhere():
    """Garde-fou global : le motif `old[~old.city.isin(...)]` (ou variantes
    équivalentes de suppression en bloc par ville avant réécriture) ne doit
    réapparaître dans AUCUN script du dépôt, pas seulement les 3 corrigés ici.
    C'est le même bug que MENDEZ ALVARO (P1) sous une autre forme : une même
    faute de fusion réinventée indépendamment dans plusieurs scripts."""
    pattern = re.compile(r"\.isin\(args\.cities\)|old\[~old\.city")
    skip_dirs = {"venv", "__pycache__", ".git", "node_modules", "tests"}
    offenders = []
    for p in ROOT.rglob("*.py"):
        if any(part in skip_dirs for part in p.parts):
            continue
        if p == ROOT / "src" / "csv_upsert.py":
            continue  # mentionné dans le docstring à titre d'exemple historique
        if pattern.search(p.read_text(errors="ignore")):
            offenders.append(str(p.relative_to(ROOT)))
    assert not offenders, (
        f"Motif de suppression en bloc par ville retrouvé dans : {offenders}. "
        "Utiliser src.csv_upsert.upsert_rows à la place."
    )


def test_sensitivity_k_canonical_beijing_k3_marked_unrecoverable():
    """Régression du cas concret : la ligne beijing/k=3/distance doit être
    marquée UNRECOVERABLE (pas une valeur reconstruite par un facteur ddof),
    tant qu'aucune valeur par-seed brute n'est retrouvée."""
    path = ROOT / "results" / "sensitivity_k_canonical.csv"
    df = pd.read_csv(path)
    row = df[(df.city == "beijing") & (df.k == 3) & (df.topology == "distance")]
    assert len(row) == 1
    assert "UNRECOVERABLE" in row.source.iloc[0]
    assert pd.isna(row.delta_r2_std.iloc[0]), (
        "delta_r2_std ne doit pas contenir de valeur reconstruite sans "
        "données brutes derrière — cf. CHANGELOG_TABLES.md"
    )
