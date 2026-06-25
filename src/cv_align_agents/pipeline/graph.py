"""LangGraph orchestration wiring the agents into one screening pipeline.

Flow (per resume)::

    START
      → parse_resume        (agent, skipped if already structured)
      → parse_jd            (agent, skipped if already structured)
      → match               (agent; uses critic feedback on retry)
      → score               (deterministic)
      → hygiene             (deterministic)
      → critic              (agent; sets needs_rescore deterministically)
      → [conditional]
            ├─ prepare_retry → match     (one self-correction loop)
            └─ END

The single conditional edge (loop back to the matcher when the critic is
low-confidence, capped by ``config.max_retries``) is what makes this a genuine
LangGraph state machine rather than a straight-line function call.

Each node appends to ``state.trace`` by returning an extended list, so the audit
log accumulates without needing a custom channel reducer.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from cv_align_agents.agents.critic import critique
from cv_align_agents.agents.jd_parser import parse_jd
from cv_align_agents.agents.matcher import match
from cv_align_agents.agents.parser import parse_resume
from cv_align_agents.pipeline.hygiene import check_hygiene
from cv_align_agents.pipeline.scorer import score
from cv_align_agents.state import (
    JDRaw,
    PipelineConfig,
    PipelineState,
    ResumeRaw,
    TraceEntry,
)


def build_pipeline(llm: BaseChatModel | None = None):
    """Build and compile the screening pipeline graph.

    Args:
        llm: Optional chat model injected into every agent (used by tests to run
            the graph offline). When ``None``, each agent resolves the configured
            provider itself.

    Returns:
        A compiled LangGraph runnable. Invoke it with a :class:`PipelineState`
        (or an equivalent dict) and it returns the final state.
    """

    def _trace(state: PipelineState, agent: str, note: str = "") -> list[TraceEntry]:
        return state.trace + [TraceEntry(agent=agent, note=note)]

    def parse_resume_node(state: PipelineState) -> dict:
        if state.resume_structured is not None:
            return {}
        structured = parse_resume(state.resume_raw, llm=llm)
        return {
            "resume_structured": structured,
            "trace": _trace(state, "parser", f"{len(structured.skills)} skills"),
        }

    def parse_jd_node(state: PipelineState) -> dict:
        if state.jd_structured is not None:
            return {}
        structured = parse_jd(state.jd_raw, llm=llm)
        return {
            "jd_structured": structured,
            "trace": _trace(state, "jd_parser", structured.title or ""),
        }

    def match_node(state: PipelineState) -> dict:
        # On a retry pass, feed the critic's suggestions back to the matcher.
        feedback = None
        if state.retry_count > 0 and state.critique is not None:
            feedback = state.critique.suggestions
        result = match(
            state.resume_structured,
            state.jd_structured,
            llm=llm,
            feedback=feedback,
        )
        note = f"retry={state.retry_count}" if feedback else "initial"
        return {"match_result": result, "trace": _trace(state, "matcher", note)}

    def score_node(state: PipelineState) -> dict:
        final = score(state.match_result, state.config)
        return {
            "final_score": final,
            "trace": _trace(state, "scorer", f"score={final.score:.2f}"),
        }

    def hygiene_node(state: PipelineState) -> dict:
        report = check_hygiene(state.resume_structured, state.resume_raw.text)
        return {
            "hygiene": report,
            "trace": _trace(state, "hygiene", f"{len(report.issues)} issues"),
        }

    def critic_node(state: PipelineState) -> dict:
        result = critique(
            state.resume_structured,
            state.jd_structured,
            state.match_result,
            state.final_score,
            state.config,
            hygiene=state.hygiene,
            llm=llm,
        )
        note = f"{result.verdict}, rescore={result.needs_rescore}"
        return {"critique": result, "trace": _trace(state, "critic", note)}

    def prepare_retry_node(state: PipelineState) -> dict:
        return {
            "retry_count": state.retry_count + 1,
            "trace": _trace(state, "self_correction", "looping back to matcher"),
        }

    def route_after_critic(state: PipelineState) -> str:
        if (
            state.critique is not None
            and state.critique.needs_rescore
            and state.can_retry()
        ):
            return "retry"
        return "end"

    graph = StateGraph(PipelineState)
    graph.add_node("parse_resume", parse_resume_node)
    graph.add_node("parse_jd", parse_jd_node)
    graph.add_node("match", match_node)
    graph.add_node("score", score_node)
    graph.add_node("check_hygiene", hygiene_node)
    graph.add_node("critic", critic_node)
    graph.add_node("prepare_retry", prepare_retry_node)

    graph.add_edge(START, "parse_resume")
    graph.add_edge("parse_resume", "parse_jd")
    graph.add_edge("parse_jd", "match")
    graph.add_edge("match", "score")
    graph.add_edge("score", "check_hygiene")
    graph.add_edge("check_hygiene", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"retry": "prepare_retry", "end": END},
    )
    graph.add_edge("prepare_retry", "match")

    return graph.compile()


def run_pipeline(
    resume_raw: ResumeRaw,
    jd_raw: JDRaw,
    config: PipelineConfig | None = None,
    llm: BaseChatModel | None = None,
) -> PipelineState:
    """Run one resume through the full pipeline and return the final state.

    Args:
        resume_raw: Resume filename + extracted text.
        jd_raw: Raw job-description text.
        config: Optional pipeline config (mode, weights, retries, ...).
        llm: Optional chat model (dependency injection for tests).

    Returns:
        The final :class:`PipelineState` with structured data, scores, hygiene,
        critique, and the accumulated trace.
    """
    pipeline = build_pipeline(llm=llm)
    initial = PipelineState(
        resume_raw=resume_raw,
        jd_raw=jd_raw,
        config=config or PipelineConfig(),
    )
    result = pipeline.invoke(initial)
    # LangGraph may return a dict-like; normalise back to a PipelineState.
    if isinstance(result, PipelineState):
        return result
    return PipelineState.model_validate(result)
