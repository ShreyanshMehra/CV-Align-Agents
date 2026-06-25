"""Tests for the matcher agent (offline, using a fake LLM)."""

from __future__ import annotations

from cv_align_agents.agents.matcher import match
from cv_align_agents.state import (
    SECTIONS,
    MatchResult,
    StructuredJD,
    StructuredResume,
    SubScore,
)


class _FakeStructuredRunnable:
    def __init__(self, result, calls):
        self._result = result
        self._calls = calls

    def invoke(self, messages):
        self._calls.append(messages)
        return self._result


class _FakeLLM:
    def __init__(self, result):
        self.result = result
        self.calls: list = []

    def with_structured_output(self, schema, **kwargs):
        return _FakeStructuredRunnable(self.result, self.calls)


def _resume() -> StructuredResume:
    return StructuredResume(name="Test", skills=["python", "go"])


def _jd() -> StructuredJD:
    return StructuredJD(title="Backend", required_skills=["python"])


def test_match_normalises_to_all_sections():
    # LLM returns only one section; matcher must fill the rest with zeros.
    partial = MatchResult(
        sub_scores=[SubScore(section="skills", score=0.9, evidence=["python"])],
        overall_evidence_quality=0.8,
    )
    out = match(_resume(), _jd(), llm=_FakeLLM(partial))

    sections = {s.section for s in out.sub_scores}
    assert sections == set(SECTIONS)
    assert len(out.sub_scores) == len(SECTIONS)

    by_section = {s.section: s for s in out.sub_scores}
    assert by_section["skills"].score == 0.9
    assert by_section["experience"].score == 0.0  # filled default
    assert out.overall_evidence_quality == 0.8


def test_match_deduplicates_sections():
    dup = MatchResult(
        sub_scores=[
            SubScore(section="skills", score=0.9),
            SubScore(section="skills", score=0.1),  # duplicate, must be dropped
        ],
        overall_evidence_quality=0.5,
    )
    out = match(_resume(), _jd(), llm=_FakeLLM(dup))
    skills = [s for s in out.sub_scores if s.section == "skills"]
    assert len(skills) == 1
    assert skills[0].score == 0.9  # first one kept


def test_match_coerces_dict_result():
    payload = {
        "sub_scores": [{"section": "projects", "score": 0.5}],
        "overall_evidence_quality": 0.4,
    }
    out = match(_resume(), _jd(), llm=_FakeLLM(payload))
    assert isinstance(out, MatchResult)
    assert len(out.sub_scores) == len(SECTIONS)


def test_match_injects_feedback_into_prompt():
    result = MatchResult(sub_scores=[], overall_evidence_quality=0.3)
    llm = _FakeLLM(result)
    match(_resume(), _jd(), llm=llm, feedback=["Re-check the skills evidence"])

    human_msg = llm.calls[0][1].content
    assert "Re-check the skills evidence" in human_msg
