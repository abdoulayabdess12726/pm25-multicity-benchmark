"""
tests/test_results_io.py — garde-fous pour l'immuabilité de raw_results.csv (P3).

REVISION_BRIEF.md : raw_results.csv est écrit une fois par les runs, jamais
édité à la main. `append_run` doit rendre ça structurellement impossible
(mode append pur, refus des doublons de clé), pas juste le documenter.
"""
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import src.results_io as rio  # noqa: E402


@pytest.fixture
def raw_csv(tmp_path, monkeypatch):
    path = tmp_path / "raw_results.csv"
    monkeypatch.setattr(rio, "RAW_CSV", path)
    return path


def _row(run_id="run1", city="madrid", model="gcn", variant="", topology="distance",
         k=5, keep_frac="", seed=42, station="CASTELLANA", r2=0.5):
    return dict(city=city, model=model, variant=variant, topology=topology, k=k,
                keep_frac=keep_frac, seed=seed,
                checkpoint_id="", split_hash="abc123", n_stations=7,
                station=station, split="test", rmse=4.0, mae=2.0, r2=r2,
                run_id=run_id, config_path="06_train_multistation.py",
                git_commit="deadbeef", timestamp="2026-08-16T00:00:00")


# --------------------------------------------------------------------------- #
# 1. Comportement de base
# --------------------------------------------------------------------------- #
def test_append_run_creates_file_with_header(raw_csv):
    rio.append_run([_row()])
    assert raw_csv.exists()
    df = rio.load_results()
    assert len(df) == 1
    assert list(df.columns) == rio.COLUMNS


def test_load_results_empty_when_no_file(raw_csv):
    df = rio.load_results()
    assert len(df) == 0
    assert list(df.columns) == rio.COLUMNS


def test_append_run_noop_on_empty_records(raw_csv):
    rio.append_run([])
    assert not raw_csv.exists()


# --------------------------------------------------------------------------- #
# 2. Immuabilité : append pur, jamais de réécriture des lignes existantes
# --------------------------------------------------------------------------- #
def test_append_run_never_rewrites_existing_bytes(raw_csv):
    """Les octets déjà écrits doivent rester un préfixe exact du fichier
    après un append supplémentaire — preuve qu'aucune réécriture du fichier
    entier n'a eu lieu (pas seulement que les données sont correctes)."""
    rio.append_run([_row(run_id="run1", station="CASTELLANA")])
    bytes_before = raw_csv.read_bytes()

    rio.append_run([_row(run_id="run1", station="ESCUELAS AGUIRRE")])
    bytes_after = raw_csv.read_bytes()

    assert bytes_after.startswith(bytes_before), (
        "append_run a modifié des octets déjà écrits — ce n'est plus un "
        "append pur, l'immuabilité n'est plus garantie structurellement"
    )
    assert len(bytes_after) > len(bytes_before)


def test_append_run_multiple_stations_and_aggregate(raw_csv):
    rows = [_row(run_id="run1", station=s) for s in ["A", "B", "__aggregate__"]]
    rio.append_run(rows)
    df = rio.load_results()
    assert len(df) == 3
    assert "__aggregate__" in df.station.values


# --------------------------------------------------------------------------- #
# 3. Refus strict des doublons — rien n'est écrit en cas de refus
# --------------------------------------------------------------------------- #
def test_append_run_refuses_existing_key(raw_csv):
    rio.append_run([_row(run_id="run1", station="CASTELLANA", r2=0.5)])
    bytes_before = raw_csv.read_bytes()

    with pytest.raises(ValueError, match="refuse d'écraser"):
        rio.append_run([_row(run_id="run1", station="CASTELLANA", r2=0.999)])

    # rien n'a été écrit suite au refus — pas de corruption partielle
    assert raw_csv.read_bytes() == bytes_before
    df = rio.load_results()
    assert len(df) == 1
    assert df.r2.iloc[0] == pytest.approx(0.5)  # valeur originale intacte, pas écrasée


def test_append_run_refuses_internal_duplicate_keys(raw_csv):
    with pytest.raises(ValueError, match="doublons de clé"):
        rio.append_run([_row(run_id="run1", station="A"), _row(run_id="run1", station="A")])
    assert not raw_csv.exists()


def test_append_run_different_run_id_same_condition_both_kept(raw_csv):
    """Deux exécutions différentes (run_id différent) de la même condition
    (ex. re-run) doivent toutes les deux être conservées — ce n'est PAS un
    upsert, c'est un append-only : l'historique complet est préservé."""
    rio.append_run([_row(run_id="run1", station="A", r2=0.5)])
    rio.append_run([_row(run_id="run2", station="A", r2=0.55)])
    df = rio.load_results()
    assert len(df) == 2
    assert sorted(df.r2.tolist()) == [0.5, 0.55]


# --------------------------------------------------------------------------- #
# 4. Aucune fonction de mise à jour / suppression exposée par le module
# --------------------------------------------------------------------------- #
def test_no_update_or_delete_function_exists():
    forbidden = re.compile(r"update|delete|remove|overwrite|edit|drop|truncate", re.I)
    public_names = [n for n in dir(rio) if not n.startswith("_")]
    offenders = [n for n in public_names if callable(getattr(rio, n)) and forbidden.search(n)]
    assert not offenders, (
        f"src.results_io expose une fonction dont le nom suggère une "
        f"mise à jour/suppression : {offenders} — l'API doit rester "
        "append-only, rien à part append_run/load_results/helpers de métadonnées."
    )


def test_module_public_api_is_minimal():
    assert set(rio.__all__) == {
        "append_run", "load_results", "COLUMNS", "KEY_COLS",
        "make_run_id", "git_commit_hash", "compute_split_hash",
    }


# --------------------------------------------------------------------------- #
# 5. Helpers de métadonnées
# --------------------------------------------------------------------------- #
def test_compute_split_hash_deterministic():
    h1 = rio.compute_split_hash(1000, 700, 850, 24)
    h2 = rio.compute_split_hash(1000, 700, 850, 24)
    h3 = rio.compute_split_hash(1000, 700, 851, 24)
    assert h1 == h2
    assert h1 != h3


def test_make_run_id_unique():
    assert rio.make_run_id("test") != rio.make_run_id("test")


def test_git_commit_hash_returns_string():
    h = rio.git_commit_hash()
    assert isinstance(h, str) and len(h) > 0
