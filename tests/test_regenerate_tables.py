"""
tests/test_regenerate_tables.py — garde-fou pour l'unicité structurelle des
conditions dans raw_results.csv (P4).

Régression exacte de l'incident : le doublon k=5 (JSON canonique vs
e6_k_sensitivity.csv) est passé inaperçu parce qu'`append_run` ne détecte
les collisions que sur (run_id, ...) — deux run_id différents pour la même
condition logique ne sont jamais comparés. Cette assertion vérifie
l'identité logique (IDENTITY_COLS), indépendamment de run_id.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.regenerate_tables as rt  # noqa: E402


def _base_row(**overrides):
    row = dict(city="madrid", model="GCN-Transformer", topology="distance", k=5,
              keep_frac=None, variant="", seed=42, station="__aggregate__", split="test",
              r2=0.49, run_id="run_a", provenance_note="")
    row.update(overrides)
    return row


@pytest.fixture(autouse=True)
def _reset_assertions():
    rt.ASSERTIONS.clear()
    yield
    rt.ASSERTIONS.clear()


def test_flags_same_condition_under_two_run_ids():
    """Reproduction exacte de l'incident : même condition, run_id différent."""
    df = pd.DataFrame([_base_row(run_id="migrated_06_train_multistation_madrid", r2=0.485319),
                       _base_row(run_id="migrated_e6_k_sensitivity_madrid", r2=0.4853)])
    rt.assert_no_duplicate_conditions_across_run_ids(df)
    assert len(rt.ASSERTIONS) == 1
    assert rt.ASSERTIONS[0]["status"] == "FAIL"
    assert "run_a" not in rt.ASSERTIONS[0]["detail"]  # pas de faux positif du fixture par défaut


def test_passes_when_run_ids_match_distinct_conditions():
    """Seeds différents = conditions différentes légitimement sous des run_id
    différents — ne doit PAS être signalé."""
    df = pd.DataFrame([
        _base_row(seed=42, run_id="run_a"),
        _base_row(seed=123, run_id="run_b"),
        _base_row(seed=777, run_id="run_c"),
    ])
    rt.assert_no_duplicate_conditions_across_run_ids(df)
    assert rt.ASSERTIONS[0]["status"] == "PASS"


def test_passes_when_same_condition_same_run_id_repeated():
    """Une seule condition, un seul run_id (cas normal, plusieurs stations
    du même run) ne doit pas être signalée."""
    df = pd.DataFrame([
        _base_row(station="A", run_id="run_a"),
        _base_row(station="B", run_id="run_a"),
    ])
    rt.assert_no_duplicate_conditions_across_run_ids(df)
    assert rt.ASSERTIONS[0]["status"] == "PASS"


def test_handles_nan_identity_columns():
    """k/keep_frac/variant NULL (ex. Linear-Transformer, Persistence) ne
    doivent pas planter le groupby ni être confondus entre eux."""
    df = pd.DataFrame([
        _base_row(model="Linear-Transformer", k=None, keep_frac=None, run_id="run_a"),
        _base_row(model="Persistence", k=None, keep_frac=None, run_id="run_b"),
    ])
    rt.assert_no_duplicate_conditions_across_run_ids(df)
    assert rt.ASSERTIONS[0]["status"] == "PASS"


# --------------------------------------------------------------------------- #
# resolve_superseded_suspect_rows (P11.2) — supersession volontaire d'une
# ligne SUSPECT_6STATION par un re-run propre, sans affaiblir la détection
# des vrais doublons ci-dessus.
# --------------------------------------------------------------------------- #
def test_resolve_drops_suspect_row_when_clean_rerun_exists():
    """Même condition, une ligne SUSPECT + une ligne propre sous un run_id
    différent : la ligne SUSPECT doit être exclue, la propre conservée."""
    df = pd.DataFrame([
        _base_row(model="ARIMA", run_id="migrated_old", provenance_note="SUSPECT_6STATION: ..."),
        _base_row(model="ARIMA", run_id="10_external_baselines_new", provenance_note=""),
    ])
    out = rt.resolve_superseded_suspect_rows(df)
    assert len(out) == 1
    assert out.iloc[0]["run_id"] == "10_external_baselines_new"


def test_resolve_leaves_pure_duplicate_bug_untouched():
    """Deux lignes PROPRES (ni l'une ni l'autre SUSPECT), même condition,
    run_id différents : ce n'est PAS une supersession — doit rester intact
    pour qu'assert_no_duplicate_conditions_across_run_ids le détecte
    toujours comme avant (régression sur l'incident k=5)."""
    df = pd.DataFrame([
        _base_row(run_id="run_a", provenance_note=""),
        _base_row(run_id="run_b", provenance_note=""),
    ])
    out = rt.resolve_superseded_suspect_rows(df)
    assert len(out) == 2
    rt.assert_no_duplicate_conditions_across_run_ids(out)
    assert rt.ASSERTIONS[0]["status"] == "FAIL"


def test_resolve_leaves_all_suspect_group_untouched():
    """Deux lignes SUSPECT, même condition, run_id différents : aucun re-run
    propre n'existe encore pour cette condition — rien à superséder, le
    groupe reste tel quel (signalé par l'assertion principale)."""
    df = pd.DataFrame([
        _base_row(run_id="run_a", provenance_note="SUSPECT_6STATION: ..."),
        _base_row(run_id="run_b", provenance_note="SUSPECT_6STATION: ..."),
    ])
    out = rt.resolve_superseded_suspect_rows(df)
    assert len(out) == 2


def test_current_raw_results_has_zero_duplicates():
    """Régression sur le fichier réel, post-correctif P4 : doit rester à 0
    SUR LA VUE UTILISÉE PAR LES TABLES (rt.load(), qui applique
    resolve_superseded_suspect_rows) — pas sur le CSV brut. Depuis P11.2,
    raw_results.csv contient légitimement 42 conditions à deux run_id
    (Madrid ARIMA/XGBoost/LSTM/Persistence : ligne SUSPECT_6STATION
    pré-correctif + re-run propre 7 stations) ; c'est l'objet même de
    resolve_superseded_suspect_rows de les résoudre avant génération des
    tables. Utiliser load_results() nu ici redeviendrait un faux positif à
    chaque supersession légitime future — rt.load() est la vue qui compte."""
    from src.results_io import load_results
    if len(load_results()) == 0:
        pytest.skip("raw_results.csv absent")
    df = rt.load()
    rt.assert_no_duplicate_conditions_across_run_ids(df)
    assert rt.ASSERTIONS[0]["status"] == "PASS", rt.ASSERTIONS[0]["detail"]
