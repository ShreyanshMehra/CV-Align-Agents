"""Tests for the FastAPI app (offline, fake LLM via dependency override)."""

from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from cv_align_agents.agents.critic import _CriticLLMOutput
from cv_align_agents.api.main import app, get_llm
from cv_align_agents.state import (
    MatchResult,
    StructuredJD,
    StructuredResume,
    SubScore,
)


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
def client():
    app.dependency_overrides[get_llm] = _fake_llm
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health():
    c = TestClient(app)
    resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


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
