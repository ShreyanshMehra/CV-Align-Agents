"""Tests for the FastAPI app (offline, fake LLM via dependency override)."""

from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from cv_align_agents.agents.critic import _CriticLLMOutput
from cv_align_agents.api.main import app, get_llm, get_store
from cv_align_agents.state import (
    MatchResult,
    StructuredJD,
    StructuredResume,
    SubScore,
)
from cv_align_agents.storage.runs import RunStore


class _RoutingRunnable:
    def __init__(self, schema, results):
        self._schema = schema
        self._results = results

    def invoke(self, messages):
        return self._results[self._schema.__name__]


class _RoutingFakeLLM:
    def __init__(self, results: dict):
        self.results = results

    def with_structured_output(self, schema, **kwargs):
        return _RoutingRunnable(schema, self.results)


def _fake_llm() -> _RoutingFakeLLM:
    return _RoutingFakeLLM(
        {
            "StructuredResume": StructuredResume(name="Cand", skills=["python"]),
            "StructuredJD": StructuredJD(title="Backend", required_skills=["python"]),
            "MatchResult": MatchResult(
                sub_scores=[SubScore(section="skills", score=0.8)],
                overall_evidence_quality=0.9,
            ),
            "_CriticLLMOutput": _CriticLLMOutput(
                gaps=[], suggestions=["Add a link"], verdict="moderate_fit",
                confidence_in_scoring=0.95,
            ),
        }
    )


def _blank_pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    store = RunStore(tmp_path / "test_runs.db")
    app.dependency_overrides[get_llm] = _fake_llm
    app.dependency_overrides[get_store] = lambda: store
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health():
    c = TestClient(app)
    resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_root_redirects_to_docs():
    c = TestClient(app)
    resp = c.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/docs"


def test_screen_recruiter_ranks_candidates(client):
    pdf = _blank_pdf_bytes()
    files = [
        ("resumes", ("a.pdf", pdf, "application/pdf")),
        ("resumes", ("b.pdf", pdf, "application/pdf")),
    ]
    resp = client.post(
        "/screen",
        data={"jd": "Backend engineer, Python required", "mode": "recruiter"},
        files=files,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "recruiter"
    assert body["job_title"] == "Backend"
    assert len(body["candidates"]) == 2
    scores = [c["score"] for c in body["candidates"]]
    assert scores == sorted(scores, reverse=True)


def test_screen_candidate_mode(client):
    files = [("resumes", ("a.pdf", _blank_pdf_bytes(), "application/pdf"))]
    resp = client.post(
        "/screen",
        data={"jd": "Backend engineer", "mode": "candidate"},
        files=files,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "candidate"
    assert len(body["candidates"]) == 1
    assert "suggestions" in body["candidates"][0]


def test_screen_empty_jd_rejected(client):
    files = [("resumes", ("a.pdf", _blank_pdf_bytes(), "application/pdf"))]
    resp = client.post("/screen", data={"jd": "   "}, files=files)
    assert resp.status_code == 422


def test_screen_invalid_mode_rejected(client):
    files = [("resumes", ("a.pdf", _blank_pdf_bytes(), "application/pdf"))]
    resp = client.post(
        "/screen",
        data={"jd": "Backend", "mode": "not_a_mode"},
        files=files,
    )
    assert resp.status_code == 422


def test_screen_requires_resume_file(client):
    resp = client.post("/screen", data={"jd": "Backend"})
    # FastAPI returns 422 when the required file field is missing.
    assert resp.status_code == 422


def test_screen_persists_run_and_is_retrievable(client):
    files = [("resumes", ("a.pdf", _blank_pdf_bytes(), "application/pdf"))]
    resp = client.post(
        "/screen", data={"jd": "Backend engineer", "mode": "recruiter"}, files=files
    )
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]
    assert run_id

    # The run is retrievable via GET /runs/{id}.
    fetched = client.get(f"/runs/{run_id}")
    assert fetched.status_code == 200
    assert fetched.json()["run_id"] == run_id

    # And it shows up in the recent-runs listing.
    listing = client.get("/runs")
    assert listing.status_code == 200
    assert any(r["id"] == run_id for r in listing.json()["runs"])


def test_get_unknown_run_returns_404(client):
    resp = client.get("/runs/nonexistent-id")
    assert resp.status_code == 404
