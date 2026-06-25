"""Tests for the LangGraph pipeline orchestration (offline, routing fake LLM)."""

from __future__ import annotations

from cv_align_agents.agents.critic import _CriticLLMOutput
from cv_align_agents.pipeline.graph import build_pipeline, run_pipeline
from cv_align_agents.state import (
    CandidateResult,
    JDRaw,
    MatchResult,
    PipelineConfig,
    ResumeRaw,
    StructuredJD,
    StructuredResume,
    SubScore,
)


class _RoutingRunnable:
    def __init__(self, schema, results, counts):
        self._schema = schema
        self._results = results
        self._counts = counts

    def invoke(self, messages):
        self._counts[self._schema.__name__] = (
            self._counts.get(self._schema.__name__, 0) + 1
        )
        return self._results[self._schema.__name__]


class _RoutingFakeLLM:
    """Dispatches with_structured_output() by the requested schema name."""

    def __init__(self, results: dict):
        self.results = results
        self.counts: dict[str, int] = {}

    def with_structured_output(self, schema, **kwargs):
        return _RoutingRunnable(schema, self.results, self.counts)


def _results(confidence: float) -> dict:
    return {
        "StructuredResume": StructuredResume(name="Jane", skills=["python", "go"]),
        "StructuredJD": StructuredJD(title="Backend", required_skills=["python"]),
        "MatchResult": MatchResult(
            sub_scores=[SubScore(section="skills", score=0.8)],
            overall_evidence_quality=0.9,
        ),
        "_CriticLLMOutput": _CriticLLMOutput(
            gaps=["Missing Kafka"],
            suggestions=["Add a Kafka project"],
            verdict="moderate_fit",
            confidence_in_scoring=confidence,
        ),
    }


def _inputs():
    return (
        ResumeRaw(filename="jane.pdf", text="Jane\nSkills: python, go"),
        JDRaw(text="Backend engineer, Python required"),
    )


def test_happy_path_no_retry():
    resume_raw, jd_raw = _inputs()
    llm = _RoutingFakeLLM(_results(confidence=0.95))  # high confidence -> no loop

    state = run_pipeline(resume_raw, jd_raw, PipelineConfig(), llm=llm)

    assert state.resume_structured.name == "Jane"
    assert state.jd_structured.title == "Backend"
    assert state.match_result is not None
    assert state.final_score is not None
    assert state.hygiene is not None
    assert state.critique.verdict == "moderate_fit"
    assert state.retry_count == 0
    # Matcher and critic each ran exactly once.
    assert llm.counts["MatchResult"] == 1
    assert llm.counts["_CriticLLMOutput"] == 1
    # Trace recorded every stage in order.
    agents = [t.agent for t in state.trace]
    assert agents == ["parser", "jd_parser", "matcher", "scorer", "hygiene", "critic"]


def test_self_correction_loop_runs_once():
    resume_raw, jd_raw = _inputs()
    # Low confidence -> needs_rescore -> loop back to matcher exactly once.
    llm = _RoutingFakeLLM(_results(confidence=0.2))

    state = run_pipeline(
        resume_raw, jd_raw, PipelineConfig(max_retries=1, confidence_threshold=0.6),
        llm=llm,
    )

    assert state.retry_count == 1
    # Matcher ran twice (initial + one retry); critic ran twice as well.
    assert llm.counts["MatchResult"] == 2
    assert llm.counts["_CriticLLMOutput"] == 2
    # Parsers ran only once (idempotent skip on the second pass).
    assert llm.counts["StructuredResume"] == 1
    assert llm.counts["StructuredJD"] == 1
    assert any(t.agent == "self_correction" for t in state.trace)


def test_retry_disabled_when_max_retries_zero():
    resume_raw, jd_raw = _inputs()
    llm = _RoutingFakeLLM(_results(confidence=0.1))  # would want to rescore

    state = run_pipeline(
        resume_raw, jd_raw, PipelineConfig(max_retries=0), llm=llm,
    )

    assert state.retry_count == 0
    assert llm.counts["MatchResult"] == 1  # no loop


def test_pipeline_result_converts_to_candidate_result():
    resume_raw, jd_raw = _inputs()
    llm = _RoutingFakeLLM(_results(confidence=0.95))
    state = run_pipeline(resume_raw, jd_raw, PipelineConfig(), llm=llm)

    result = CandidateResult.from_state(state)
    assert result.filename == "jane.pdf"
    assert result.candidate_name == "Jane"
    assert result.verdict == "moderate_fit"
    assert 0.0 <= result.score <= 1.0
    assert result.hygiene_score is not None


def test_prebuilt_pipeline_is_reusable():
    resume_raw, jd_raw = _inputs()
    llm = _RoutingFakeLLM(_results(confidence=0.95))
    pipeline = build_pipeline(llm=llm)

    from cv_align_agents.state import PipelineState

    out = pipeline.invoke(
        PipelineState(resume_raw=resume_raw, jd_raw=jd_raw, config=PipelineConfig())
    )
    state = out if isinstance(out, PipelineState) else PipelineState.model_validate(out)
    assert state.critique is not None
