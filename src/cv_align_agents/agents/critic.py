"""Critic agent: review the match, produce feedback, and gate self-correction.

The critic is the last agent in the pipeline. It looks at the structured resume,
the job description, the matcher's per-section scores and evidence, the final
score, and the deterministic hygiene report, and produces:

- ``gaps``: specific JD requirements the candidate does not clearly meet
- ``suggestions``: actionable improvements (including hygiene fixes)
- ``verdict``: an overall strong/moderate/weak fit label
- ``confidence_in_scoring``: how well-justified the matcher's scores look

Whether to loop back for a single re-evaluation is decided *deterministically*
from the confidence and the matcher's evidence quality (not left to the LLM), so
the self-correction behaviour is predictable and tunable.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from cv_align_agents.llm.client import get_chat_model
from cv_align_agents.state import (
    Critique,
    FinalScore,
    HygieneReport,
    MatchResult,
    PipelineConfig,
    StructuredJD,
    StructuredResume,
    Verdict,
)


class _CriticLLMOutput(BaseModel):
    """What we ask the LLM for. ``needs_rescore`` is decided deterministically
    afterwards, so it is intentionally not part of this schema."""

    gaps: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    verdict: Verdict = "moderate_fit"
    confidence_in_scoring: float = Field(default=1.0, ge=0.0, le=1.0)


SYSTEM_PROMPT = """\
You are a constructive hiring reviewer. You are given a job description, a \
candidate's structured resume, the per-section match scores with their evidence, \
the final weighted score, and a deterministic resume-hygiene report.

Produce:
- `gaps`: specific requirements from the JOB DESCRIPTION that the candidate does \
not clearly satisfy. Be concrete (name the missing skill/experience). Empty list \
if there are none.
- `suggestions`: actionable, candidate-facing advice to improve fit for THIS \
role. Fold in any relevant hygiene issues (e.g. missing links, unquantified \
experience). Keep each suggestion short and specific.
- `verdict`: one of strong_fit, moderate_fit, weak_fit, reflecting overall fit.
- `confidence_in_scoring`: 0.0-1.0 — how confident you are that the per-section \
scores are well-justified by the cited evidence. Lower this when evidence is \
thin, ambiguous, or seems inconsistent with the scores.

Be honest and specific; do not pad the lists.\
"""


def _build_human_message(
    resume: StructuredResume,
    jd: StructuredJD,
    match_result: MatchResult,
    final_score: FinalScore,
    hygiene: HygieneReport | None,
) -> str:
    lines = [
        "## JOB DESCRIPTION",
        jd.model_dump_json(indent=2),
        "",
        "## CANDIDATE RESUME",
        resume.model_dump_json(indent=2),
        "",
        "## PER-SECTION MATCH SCORES",
    ]
    for sub in match_result.sub_scores:
        lines.append(
            f"- {sub.section}: {sub.score:.2f} | evidence={sub.evidence} | "
            f"reasoning={sub.reasoning}"
        )
    lines.append(f"\nOverall evidence quality: {match_result.overall_evidence_quality:.2f}")
    lines.append(f"Final weighted score: {final_score.score:.2f}")

    if hygiene is not None:
        lines.append("\n## RESUME HYGIENE (deterministic)")
        lines.append(f"Hygiene score: {hygiene.score:.2f}")
        for issue in hygiene.issues:
            lines.append(f"- [{issue.severity}] {issue.message}")
    return "\n".join(lines)


def critique(
    resume: StructuredResume,
    jd: StructuredJD,
    match_result: MatchResult,
    final_score: FinalScore,
    config: PipelineConfig,
    hygiene: HygieneReport | None = None,
    llm: BaseChatModel | None = None,
) -> Critique:
    """Generate critic feedback and decide whether to self-correct.

    Args:
        resume: Structured resume.
        jd: Structured job description.
        match_result: Matcher output (sub-scores + evidence quality).
        final_score: Deterministic final score.
        config: Pipeline config (supplies the confidence threshold).
        hygiene: Optional deterministic hygiene report to fold into suggestions.
        llm: Optional chat model (dependency injection for tests).

    Returns:
        A :class:`Critique`. ``needs_rescore`` is set deterministically: true when
        the critic's confidence or the matcher's evidence quality falls below
        ``config.confidence_threshold``.
    """
    llm = llm or get_chat_model()
    reviewer = llm.with_structured_output(_CriticLLMOutput)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=_build_human_message(
                resume, jd, match_result, final_score, hygiene
            )
        ),
    ]
    out = reviewer.invoke(messages)
    if isinstance(out, dict):
        out = _CriticLLMOutput.model_validate(out)

    # Deterministic self-correction decision (not left to the LLM).
    threshold = config.confidence_threshold
    needs_rescore = (
        out.confidence_in_scoring < threshold
        or match_result.overall_evidence_quality < threshold
    )

    return Critique(
        gaps=out.gaps,
        suggestions=out.suggestions,
        verdict=out.verdict,
        confidence_in_scoring=out.confidence_in_scoring,
        needs_rescore=needs_rescore,
    )
