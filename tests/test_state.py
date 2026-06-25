"""Tests for the pipeline state schema and its contracts."""

from __future__ import annotations

import math

from cv_align_agents.state import (
    DEFAULT_WEIGHTS,
    SECTIONS,
    CandidateResult,
    Critique,
    FinalScore,
    JDRaw,
    PipelineConfig,
    PipelineState,
    ResumeRaw,
    StructuredResume,
)


def _make_state(**config_kwargs) -> PipelineState:
    return PipelineState(
        resume_raw=ResumeRaw(filename="alice.pdf", text="resume text"),
        jd_raw=JDRaw(text="job description text"),
        config=PipelineConfig(**config_kwargs),
    )


def test_defaults_are_sane():
    state = _make_state()
    assert state.config.mode == "recruiter"
    assert state.config.critic_mode == "full"
    assert state.retry_count == 0
    assert state.resume_structured is None
    assert state.trace == []
    # All four sections are present in the default weights.
    assert set(DEFAULT_WEIGHTS) == set(SECTIONS)


def test_weights_default_sum_to_one():
    state = _make_state()
    assert math.isclose(sum(state.config.weights.values()), 1.0, rel_tol=1e-9)


def test_weight_override_is_normalised():
    # Arbitrary, non-normalised override should be rescaled to sum to 1.0.
    state = _make_state(weights={"skills": 3, "experience": 1})
    total = sum(state.config.weights.values())
    assert math.isclose(total, 1.0, rel_tol=1e-9)
    # Relative ordering preserved: skills dominates.
    assert state.config.weights["skills"] > state.config.weights["experience"]
    # Unspecified sections fall back to their defaults (then normalised), so are
    # still present and non-zero.
    assert state.config.weights["projects"] > 0
    assert set(state.config.weights) == set(SECTIONS)


def test_unknown_weight_keys_are_ignored():
    state = _make_state(weights={"skills": 1, "bogus": 99})
    assert "bogus" not in state.config.weights
    assert math.isclose(sum(state.config.weights.values()), 1.0, rel_tol=1e-9)


def test_zero_weights_fall_back_to_defaults():
    state = _make_state(weights={"skills": 0, "experience": 0,
                                 "projects": 0, "education": 0})
    assert math.isclose(sum(state.config.weights.values()), 1.0, rel_tol=1e-9)


def test_can_retry_respects_cap():
    state = _make_state(max_retries=1)
    assert state.can_retry() is True
    state.retry_count = 1
    assert state.can_retry() is False


def test_add_trace_appends_entry():
    state = _make_state()
    state.add_trace("parser", note="extracted 5 skills", tokens_used=123)
    assert len(state.trace) == 1
    entry = state.trace[0]
    assert entry.agent == "parser"
    assert entry.tokens_used == 123
    assert entry.timestamp  # auto-populated ISO timestamp


def test_candidate_result_from_state():
    state = _make_state()
    state.resume_structured = StructuredResume(name="Alice Doe")
    state.final_score = FinalScore(
        score=0.82, weights_used=dict(DEFAULT_WEIGHTS),
        breakdown={"skills": 0.9, "experience": 0.7},
    )
    state.critique = Critique(
        gaps=["No Kubernetes experience"],
        suggestions=["Add a cloud deployment project"],
        verdict="strong_fit",
        confidence_in_scoring=0.9,
    )

    result = CandidateResult.from_state(state)
    assert result.filename == "alice.pdf"
    assert result.candidate_name == "Alice Doe"
    assert result.score == 0.82
    assert result.verdict == "strong_fit"
    assert result.gaps == ["No Kubernetes experience"]


def test_candidate_result_handles_empty_state():
    # A pipeline that failed before scoring still yields a safe result.
    state = _make_state()
    result = CandidateResult.from_state(state)
    assert result.score == 0.0
    assert result.verdict == "weak_fit"
    assert result.gaps == []


def test_state_round_trips_through_json():
    state = _make_state(mode="candidate")
    dumped = state.model_dump_json()
    restored = PipelineState.model_validate_json(dumped)
    assert restored.config.mode == "candidate"
    assert restored.resume_raw.filename == "alice.pdf"
