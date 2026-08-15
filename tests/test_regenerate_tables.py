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
              r2=0.49, run_id="run_a")
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


def test_current_raw_results_has_zero_duplicates():
    """Régression sur le fichier réel, post-correctif P4 : doit rester à 0."""
    from src.results_io import load_results
    df = load_results()
    if len(df) == 0:
        pytest.skip("raw_results.csv absent")
    rt.assert_no_duplicate_conditions_across_run_ids(df)
    assert rt.ASSERTIONS[0]["status"] == "PASS", rt.ASSERTIONS[0]["detail"]
