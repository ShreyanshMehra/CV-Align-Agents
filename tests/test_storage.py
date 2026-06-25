"""Tests for the SQLite run store."""

from __future__ import annotations

import pytest

from cv_align_agents.state import CandidateResult, ScreeningResult
from cv_align_agents.storage.runs import RunStore


def _result(mode="recruiter") -> ScreeningResult:
    return ScreeningResult(
        mode=mode,
        job_title="Backend",
        candidates=[
            CandidateResult(filename="a.pdf", score=0.8, verdict="strong_fit"),
            CandidateResult(filename="b.pdf", score=0.4, verdict="weak_fit"),
        ],
    )


def test_save_assigns_run_id_and_created_at(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    result = _result()
    assert result.run_id is None

    run_id = store.save(result)
    assert run_id
    assert result.run_id == run_id
    assert result.created_at is not None


def test_get_round_trips(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    result = _result()
    run_id = store.save(result)

    fetched = store.get(run_id)
    assert fetched is not None
    assert fetched.run_id == run_id
    assert fetched.job_title == "Backend"
    assert len(fetched.candidates) == 2
    assert fetched.candidates[0].filename == "a.pdf"


def test_get_unknown_returns_none(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    assert store.get("does-not-exist") is None


def test_list_runs_newest_first(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    id1 = store.save(ScreeningResult(mode="recruiter", created_at="2026-01-01T00:00:00",
                                     job_title="A"))
    id2 = store.save(ScreeningResult(mode="candidate", created_at="2026-02-01T00:00:00",
                                     job_title="B"))

    runs = store.list_runs()
    assert [r["id"] for r in runs] == [id2, id1]
    assert runs[0]["job_title"] == "B"


def test_persists_across_store_instances(tmp_path):
    db = tmp_path / "runs.db"
    run_id = RunStore(db).save(_result())
    # A new store pointing at the same file can read the run back.
    assert RunStore(db).get(run_id) is not None


def test_memory_path_rejected():
    with pytest.raises(ValueError):
        RunStore(":memory:")
