"""Tests for the critic agent (offline, using a fake LLM)."""

from __future__ import annotations

from cv_align_agents.agents.critic import _CriticLLMOutput, critique
from cv_align_agents.state import (
    FinalScore,
    HygieneIssue,
    HygieneReport,
    MatchResult,
    PipelineConfig,
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


def _match(evidence_quality: float = 1.0) -> MatchResult:
    return MatchResult(
        sub_scores=[SubScore(section="skills", score=0.8)],
        overall_evidence_quality=evidence_quality,
    )


def _final() -> FinalScore:
    return FinalScore(score=0.7, weights_used={}, breakdown={})


def _llm_out(confidence: float) -> _CriticLLMOutput:
    return _CriticLLMOutput(
        gaps=["Missing Kafka experience"],
        suggestions=["Add a project using Kafka"],
        verdict="moderate_fit",
        confidence_in_scoring=confidence,
    )


def test_high_confidence_does_not_trigger_rescore():
    out = critique(
        StructuredResume(), StructuredJD(),
        _match(evidence_quality=0.9), _final(),
        PipelineConfig(confidence_threshold=0.6),
        llm=_FakeLLM(_llm_out(0.9)),
    )
    assert out.needs_rescore is False
    assert out.verdict == "moderate_fit"
    assert out.gaps == ["Missing Kafka experience"]


def test_low_confidence_triggers_rescore():
    out = critique(
        StructuredResume(), StructuredJD(),
        _match(evidence_quality=0.9), _final(),
        PipelineConfig(confidence_threshold=0.6),
        llm=_FakeLLM(_llm_out(0.3)),  # below threshold
    )
    assert out.needs_rescore is True


def test_low_evidence_quality_triggers_rescore():
    out = critique(
        StructuredResume(), StructuredJD(),
        _match(evidence_quality=0.2), _final(),  # low evidence quality
        PipelineConfig(confidence_threshold=0.6),
        llm=_FakeLLM(_llm_out(0.95)),  # high confidence, but evidence weak
    )
    assert out.needs_rescore is True


def test_hygiene_issues_included_in_prompt():
    hygiene = HygieneReport(
        score=0.7,
        issues=[HygieneIssue(check="missing_github", severity="warning",
                             message="No GitHub link found")],
    )
    llm = _FakeLLM(_llm_out(0.9))
    critique(
        StructuredResume(), StructuredJD(), _match(), _final(),
        PipelineConfig(), hygiene=hygiene, llm=llm,
    )
    human_msg = llm.calls[0][1].content
    assert "No GitHub link found" in human_msg
    assert "RESUME HYGIENE" in human_msg


def test_critic_coerces_dict_result():
    payload = {
        "gaps": [],
        "suggestions": ["x"],
        "verdict": "strong_fit",
        "confidence_in_scoring": 0.8,
    }
    out = critique(
        StructuredResume(), StructuredJD(), _match(), _final(),
        PipelineConfig(), llm=_FakeLLM(payload),
    )
    assert out.verdict == "strong_fit"
    assert out.needs_rescore is False
