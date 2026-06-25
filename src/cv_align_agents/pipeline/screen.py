"""High-level screening orchestration over the per-resume pipeline.

This layer sits above the LangGraph pipeline and handles the two personas:

- ``candidate`` mode: one resume, full critic feedback.
- ``recruiter`` mode: many resumes ranked against one JD. In ``full`` critic
  mode every candidate is critiqued; in ``fast`` mode only the top-K ranked
  candidates are critiqued (saving LLM calls at scale).

The JD is parsed once and reused across all resumes (the pipeline's parser nodes
are idempotent), and resumes are processed concurrently.
"""

from __future__ import annotations

import asyncio

from langchain_core.language_models import BaseChatModel

from cv_align_agents.agents.critic import critique
from cv_align_agents.agents.jd_parser import parse_jd
from cv_align_agents.pipeline.graph import build_pipeline
from cv_align_agents.state import (
    CandidateResult,
    JDRaw,
    PipelineConfig,
    PipelineState,
    ResumeRaw,
    ScreeningResult,
)


async def _run_one(
    pipeline,
    resume_raw: ResumeRaw,
    jd_raw: JDRaw,
    jd_structured,
    config: PipelineConfig,
) -> PipelineState:
    state = PipelineState(
        resume_raw=resume_raw,
        jd_raw=jd_raw,
        jd_structured=jd_structured,  # parsed once, reused (parser node skips)
        config=config,
    )
    # pipeline.invoke is blocking; run it off the event loop for concurrency.
    result = await asyncio.to_thread(pipeline.invoke, state)
    if isinstance(result, PipelineState):
        return result
    return PipelineState.model_validate(result)


async def screen(
    resumes: list[ResumeRaw],
    jd: JDRaw,
    config: PipelineConfig | None = None,
    llm: BaseChatModel | None = None,
) -> ScreeningResult:
    """Screen one or more resumes against a job description.

    Args:
        resumes: Resumes to screen (one for candidate mode; many for recruiter).
        jd: The job description.
        config: Pipeline config (mode, critic_mode, weights, ...).
        llm: Optional chat model (dependency injection for tests).

    Returns:
        A :class:`ScreeningResult` with candidates ranked by score (descending).
    """
    config = config or PipelineConfig()
    jd_structured = parse_jd(jd, llm=llm)
    pipeline = build_pipeline(llm=llm)

    # In recruiter "fast" mode, score everyone without the critic first, then
    # critique only the top-K. Otherwise run the critic inline for all.
    fast = config.mode == "recruiter" and config.critic_mode == "fast"
    run_config = config.model_copy(update={"enable_critic": not fast})

    states = await asyncio.gather(
        *(_run_one(pipeline, r, jd, jd_structured, run_config) for r in resumes)
    )

    if fast:
        ranked_states = sorted(
            states,
            key=lambda s: s.final_score.score if s.final_score else 0.0,
            reverse=True,
        )
        for state in ranked_states[: config.critic_top_k]:
            state.critique = critique(
                state.resume_structured,
                state.jd_structured,
                state.match_result,
                state.final_score,
                config,
                hygiene=state.hygiene,
                llm=llm,
            )

    candidates = sorted(
        (CandidateResult.from_state(s) for s in states),
        key=lambda c: c.score,
        reverse=True,
    )
    return ScreeningResult(
        mode=config.mode,
        job_title=jd_structured.title,
        candidates=candidates,
    )
