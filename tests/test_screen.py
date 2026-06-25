"""Tests for the screening orchestration (offline, routing fake LLM)."""

from __future__ import annotations

import asyncio

from cv_align_agents.agents.critic import _CriticLLMOutput
from cv_align_agents.pipeline.screen import screen
from cv_align_agents.state import (
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
        name = self._schema.__name__
        self._counts[name] = self._counts.get(name, 0) + 1
        result = self._results[name]
        return result(self._counts) if callable(result) else result


class _RoutingFakeLLM:
    def __init__(self, results: dict):
        self.results = results
        self.counts: dict[str, int] = {}

    def with_structured_output(self, schema, **kwargs):
        return _RoutingRunnable(schema, self.results, self.counts)


def _make_llm(score_for_resume) -> _RoutingFakeLLM:
    # MatchResult score varies per call so candidates get distinct scores.
    def match_result(counts):
        n = counts["MatchResult"]
        return MatchResult(
            sub_scores=[SubScore(section="skills", score=score_for_resume(n))],
            overall_evidence_quality=0.9,
        )

    return _RoutingFakeLLM(
        {
            "StructuredResume": StructuredResume(name="Cand", skills=["python"]),
            "StructuredJD": StructuredJD(title="Backend", required_skills=["python"]),
            "MatchResult": match_result,
            "_CriticLLMOutput": _CriticLLMOutput(
                gaps=[], suggestions=["s"], verdict="moderate_fit",
                confidence_in_scoring=0.95,
            ),
        }
    )


def _resumes(n: int) -> list[ResumeRaw]:
    return [ResumeRaw(filename=f"r{i}.pdf", text=f"resume {i}") for i in range(n)]


def test_candidate_mode_returns_single_ranked_result():
    llm = _make_llm(lambda n: 0.8)
    result = asyncio.run(
        screen(_resumes(1), JDRaw(text="jd"), PipelineConfig(mode="candidate"), llm=llm)
    )
    assert result.mode == "candidate"
    assert result.job_title == "Backend"
    assert len(result.candidates) == 1
    assert result.candidates[0].suggestions  # critic ran


def test_recruiter_full_ranks_all_and_critiques_all():
    # Distinct scores by call order so ranking is observable.
    scores = {1: 0.2, 2: 0.9, 3: 0.5}
    llm = _make_llm(lambda n: scores[n])
    result = asyncio.run(
        screen(
            _resumes(3), JDRaw(text="jd"),
            PipelineConfig(mode="recruiter", critic_mode="full"), llm=llm,
        )
    )
    assert len(result.candidates) == 3
    # Sorted descending by score.
    out_scores = [c.score for c in result.candidates]
    assert out_scores == sorted(out_scores, reverse=True)
    # Every candidate critiqued in full mode.
    assert all(c.verdict == "moderate_fit" for c in result.candidates)


def test_recruiter_fast_critiques_only_top_k():
    scores = {1: 0.1, 2: 0.9, 3: 0.5, 4: 0.7}
    llm = _make_llm(lambda n: scores[n])
    result = asyncio.run(
        screen(
            _resumes(4), JDRaw(text="jd"),
            PipelineConfig(mode="recruiter", critic_mode="fast", critic_top_k=2),
            llm=llm,
        )
    )
    assert len(result.candidates) == 4
    # Critic ran for exactly the top 2 (those have suggestions).
    critiqued = [c for c in result.candidates if c.suggestions]
    assert len(critiqued) == 2
    # The two highest scores are the critiqued ones.
    assert result.candidates[0].suggestions
    assert result.candidates[1].suggestions
    assert not result.candidates[2].suggestions
    assert llm.counts["_CriticLLMOutput"] == 2


def test_jd_parsed_once_across_resumes():
    llm = _make_llm(lambda n: 0.5)
    asyncio.run(
        screen(
            _resumes(3), JDRaw(text="jd"),
            PipelineConfig(mode="recruiter", critic_mode="full"), llm=llm,
        )
    )
    # JD parsed exactly once even though 3 resumes were screened.
    assert llm.counts["StructuredJD"] == 1
