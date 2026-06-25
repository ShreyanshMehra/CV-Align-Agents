"""Matcher agent: score a structured resume against a structured JD.

This is the core of the pipeline and our key differentiator: rather than grading
a resume against a fixed rubric, the matcher evaluates how well a *specific*
candidate fits a *specific* job description. It produces one
:class:`SubScore` per section (skills, experience, projects, education) with
quoted evidence and short reasoning, plus an ``overall_evidence_quality`` signal
that later feeds the self-correction loop.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from cv_align_agents.llm.client import get_chat_model
from cv_align_agents.state import (
    SECTIONS,
    MatchResult,
    StructuredJD,
    StructuredResume,
    SubScore,
)

SYSTEM_PROMPT = """\
You are a rigorous, fair technical hiring evaluator. You are given a candidate's \
structured resume and a structured job description. Score how well the candidate \
matches the job, section by section.

Produce exactly one entry per section: skills, experience, projects, education.

For each section:
- `score`: a float from 0.0 to 1.0 measuring fit for THIS job (not generic \
resume quality). 1.0 = excellent match; 0.0 = no relevant evidence.
- `evidence`: short snippets quoted/paraphrased from the resume that justify the \
score. If there is no relevant evidence, use an empty list.
- `reasoning`: one or two sentences explaining the score with respect to the \
job's requirements.

Also produce `overall_evidence_quality` (0.0-1.0): how strong and unambiguous \
the resume evidence was for making these judgments. Low values mean the resume \
was vague, sparse, or hard to map to the job.

Be evidence-driven and avoid inflating scores. Do not reward skills the job did \
not ask for.\
"""

FEEDBACK_PREFIX = """\

A previous evaluation pass was flagged as low-confidence. Address this feedback \
and re-evaluate more carefully, grounding every score in concrete resume \
evidence:
"""


def _build_human_message(
    resume: StructuredResume,
    jd: StructuredJD,
    feedback: list[str] | None,
) -> str:
    parts = [
        "## JOB DESCRIPTION (structured)",
        jd.model_dump_json(indent=2),
        "",
        "## CANDIDATE RESUME (structured)",
        resume.model_dump_json(indent=2),
    ]
    if feedback:
        parts.append(FEEDBACK_PREFIX)
        parts.extend(f"- {item}" for item in feedback)
    return "\n".join(parts)


def _normalise_sections(result: MatchResult) -> MatchResult:
    """Ensure exactly one sub-score per known section.

    The LLM is asked for one entry per section, but we defensively de-duplicate
    (keeping the first per section) and fill any missing section with a 0 score
    so downstream scoring always has a complete, predictable set.
    """
    by_section: dict[str, SubScore] = {}
    for sub in result.sub_scores:
        if sub.section in SECTIONS and sub.section not in by_section:
            by_section[sub.section] = sub

    normalised = [
        by_section.get(
            section,
            SubScore(
                section=section,
                score=0.0,
                evidence=[],
                reasoning="No relevant evidence found.",
            ),
        )
        for section in SECTIONS
    ]
    return MatchResult(
        sub_scores=normalised,
        overall_evidence_quality=result.overall_evidence_quality,
    )


def match(
    resume: StructuredResume,
    jd: StructuredJD,
    llm: BaseChatModel | None = None,
    feedback: list[str] | None = None,
) -> MatchResult:
    """Evaluate a resume against a job description.

    Args:
        resume: The structured resume.
        jd: The structured job description.
        llm: Optional chat model (dependency injection for tests). Falls back to
            the configured provider via :func:`get_chat_model`.
        feedback: Optional suggestions from a previous critic pass, injected into
            the prompt to drive the single self-correction re-evaluation.

    Returns:
        A :class:`MatchResult` with exactly one :class:`SubScore` per section and
        an ``overall_evidence_quality`` signal.
    """
    llm = llm or get_chat_model()
    matcher = llm.with_structured_output(MatchResult)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_build_human_message(resume, jd, feedback)),
    ]
    result = matcher.invoke(messages)

    if isinstance(result, dict):
        result = MatchResult.model_validate(result)
    return _normalise_sections(result)
