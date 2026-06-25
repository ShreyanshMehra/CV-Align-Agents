"""Tests for the deterministic scorer."""

from __future__ import annotations

import math

from cv_align_agents.pipeline.scorer import score
from cv_align_agents.state import (
    DEFAULT_WEIGHTS,
    MatchResult,
    PipelineConfig,
    SubScore,
)


def _match(skills=0.0, experience=0.0, projects=0.0, education=0.0) -> MatchResult:
    return MatchResult(
        sub_scores=[
            SubScore(section="skills", score=skills),
            SubScore(section="experience", score=experience),
            SubScore(section="projects", score=projects),
            SubScore(section="education", score=education),
        ],
        overall_evidence_quality=1.0,
    )


def test_perfect_match_scores_one():
    result = score(_match(1, 1, 1, 1), PipelineConfig())
    assert math.isclose(result.score, 1.0, rel_tol=1e-9)


def test_zero_match_scores_zero():
    result = score(_match(0, 0, 0, 0), PipelineConfig())
    assert result.score == 0.0


def test_weighted_sum_matches_manual_calculation():
    m = _match(skills=1.0, experience=0.5, projects=0.0, education=1.0)
    result = score(m, PipelineConfig())  # default weights
    expected = (
        DEFAULT_WEIGHTS["skills"] * 1.0
        + DEFAULT_WEIGHTS["experience"] * 0.5
        + DEFAULT_WEIGHTS["projects"] * 0.0
        + DEFAULT_WEIGHTS["education"] * 1.0
    )
    assert math.isclose(result.score, expected, rel_tol=1e-9)


def test_breakdown_records_all_sections():
    result = score(_match(0.2, 0.4, 0.6, 0.8), PipelineConfig())
    assert result.breakdown == {
        "skills": 0.2,
        "experience": 0.4,
        "projects": 0.6,
        "education": 0.8,
    }


def test_missing_section_treated_as_zero():
    partial = MatchResult(
        sub_scores=[SubScore(section="skills", score=1.0)],
        overall_evidence_quality=1.0,
    )
    result = score(partial, PipelineConfig())
    assert result.breakdown["experience"] == 0.0
    # Only the skills weight contributes.
    assert math.isclose(result.score, DEFAULT_WEIGHTS["skills"], rel_tol=1e-9)


def test_custom_weights_change_result():
    m = _match(skills=1.0, experience=0.0, projects=0.0, education=0.0)
    # All weight on skills -> score should be 1.0.
    cfg = PipelineConfig(weights={"skills": 1, "experience": 0,
                                  "projects": 0, "education": 0})
    result = score(m, cfg)
    assert math.isclose(result.score, 1.0, rel_tol=1e-9)


def test_score_is_deterministic():
    m = _match(0.3, 0.7, 0.5, 0.9)
    cfg = PipelineConfig()
    assert score(m, cfg).score == score(m, cfg).score
