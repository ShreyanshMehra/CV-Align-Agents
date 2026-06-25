"""Shared pipeline state and the typed contracts passed between agents.

Every LangGraph node receives a :class:`PipelineState`, fills in the part it is
responsible for, and passes it along. Using explicit Pydantic models (instead of
loose dicts) means LangGraph validates the data flowing between agents and the
whole pipeline is self-documenting.

A single :class:`PipelineState` represents one resume's journey through the
pipeline. In recruiter mode the API runs one state per resume (in parallel) and
ranks the resulting :class:`CandidateResult` objects afterwards.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Sections we score a resume on, and their default contribution to the final
# score. Weights are a starting point; they can be overridden per request and
# are always normalised to sum to 1.0 (see PipelineConfig).
Section = Literal["skills", "experience", "projects", "education"]
SECTIONS: tuple[Section, ...] = ("skills", "experience", "projects", "education")
DEFAULT_WEIGHTS: dict[str, float] = {
    "skills": 0.35,
    "experience": 0.30,
    "projects": 0.25,
    "education": 0.10,
}

Mode = Literal["candidate", "recruiter"]
CriticMode = Literal["fast", "full"]
Verdict = Literal["strong_fit", "moderate_fit", "weak_fit"]


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (for trace entries)."""
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
class ResumeRaw(BaseModel):
    """A single resume after PDF -> text extraction (a deterministic function)."""

    filename: str
    text: str


class JDRaw(BaseModel):
    """The raw job description text."""

    text: str


# --------------------------------------------------------------------------- #
# Parser outputs (structured resume / JD)
# --------------------------------------------------------------------------- #
class EducationItem(BaseModel):
    degree: str | None = None
    institute: str | None = None
    year: str | None = None
    gpa: str | None = None


class ExperienceItem(BaseModel):
    company: str | None = None
    role: str | None = None
    dates: str | None = None
    bullets: list[str] = Field(default_factory=list)


class ProjectItem(BaseModel):
    name: str | None = None
    description: str | None = None
    tech: list[str] = Field(default_factory=list)


class StructuredResume(BaseModel):
    """Structured fields extracted from a resume by the parser agent."""

    name: str | None = None
    email: str | None = None
    education: list[EducationItem] = Field(default_factory=list)
    experience: list[ExperienceItem] = Field(default_factory=list)
    projects: list[ProjectItem] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)


class StructuredJD(BaseModel):
    """Structured requirements extracted from a job description."""

    title: str | None = None
    required_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    min_experience_years: float | None = None
    responsibilities: list[str] = Field(default_factory=list)
    qualifications: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Matcher output
# --------------------------------------------------------------------------- #
class SubScore(BaseModel):
    """Per-section match score with the evidence that justifies it."""

    section: Section
    score: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    reasoning: str = ""


class MatchResult(BaseModel):
    """All per-section sub-scores plus an overall evidence-quality signal.

    ``overall_evidence_quality`` (0-1) feeds the self-correction decision: if the
    matcher's own evidence is weak, the critic is more likely to request a
    re-evaluation.
    """

    sub_scores: list[SubScore] = Field(default_factory=list)
    overall_evidence_quality: float = Field(default=0.0, ge=0.0, le=1.0)


# --------------------------------------------------------------------------- #
# Scorer output (deterministic function, not an agent)
# --------------------------------------------------------------------------- #
class FinalScore(BaseModel):
    """Deterministic weighted score derived from the matcher's sub-scores."""

    score: float = Field(ge=0.0, le=1.0)
    weights_used: dict[str, float] = Field(default_factory=dict)
    breakdown: dict[str, float] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Critic output
# --------------------------------------------------------------------------- #
class Critique(BaseModel):
    """The critic's feedback and its confidence in the scoring.

    ``needs_rescore`` together with the retry cap drives the single
    self-correction loop back to the matcher.
    """

    gaps: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    verdict: Verdict = "moderate_fit"
    confidence_in_scoring: float = Field(default=1.0, ge=0.0, le=1.0)
    needs_rescore: bool = False


# --------------------------------------------------------------------------- #
# Hygiene report (deterministic, objective resume checks)
# --------------------------------------------------------------------------- #
class HygieneIssue(BaseModel):
    """A single objective issue found by the deterministic hygiene checker."""

    check: str
    severity: Literal["info", "warning"]
    message: str


class HygieneReport(BaseModel):
    """Objective resume-quality signal, independent of JD match.

    This is advisory: it informs the critic's suggestions and is surfaced to
    candidates, but it does NOT change the JD-match ``FinalScore``.
    """

    score: float = Field(default=1.0, ge=0.0, le=1.0)
    issues: list[HygieneIssue] = Field(default_factory=list)
    positives: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class PipelineConfig(BaseModel):
    """Per-request configuration.

    - ``mode`` selects the persona: ``candidate`` (CV feedback) or ``recruiter``
      (ranked screening).
    - ``critic_mode`` controls cost: ``full`` critiques every resume; ``fast``
      critiques only the top ``critic_top_k`` after ranking (recruiter mode).
    - ``weights`` are normalised to sum to 1.0 so the final score stays in [0, 1].
    """

    mode: Mode = "recruiter"
    critic_mode: CriticMode = "full"
    critic_top_k: int = Field(default=5, ge=1)
    enable_critic: bool = True
    weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    max_retries: int = Field(default=1, ge=0)
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _normalise_weights(self) -> PipelineConfig:
        # Start from defaults, apply only valid section overrides, drop unknowns.
        merged = dict(DEFAULT_WEIGHTS)
        for key, value in self.weights.items():
            if key in SECTIONS and value >= 0:
                merged[key] = float(value)

        total = sum(merged.values())
        if total <= 0:
            merged = dict(DEFAULT_WEIGHTS)
            total = sum(merged.values())

        # Normalise so the weights always sum to exactly 1.0.
        self.weights = {k: v / total for k, v in merged.items()}
        return self


# --------------------------------------------------------------------------- #
# Audit trail
# --------------------------------------------------------------------------- #
class TraceEntry(BaseModel):
    """One step in the pipeline's audit log (the explainability backbone)."""

    agent: str
    timestamp: str = Field(default_factory=_now_iso)
    note: str = ""
    tokens_used: int | None = None


# --------------------------------------------------------------------------- #
# The shared state (one per resume)
# --------------------------------------------------------------------------- #
class PipelineState(BaseModel):
    """Mutable state threaded through the LangGraph pipeline for one resume."""

    # Inputs
    resume_raw: ResumeRaw
    jd_raw: JDRaw
    config: PipelineConfig = Field(default_factory=PipelineConfig)

    # Accumulated as agents run
    resume_structured: StructuredResume | None = None
    jd_structured: StructuredJD | None = None
    match_result: MatchResult | None = None
    final_score: FinalScore | None = None
    hygiene: HygieneReport | None = None
    critique: Critique | None = None

    # Self-correction bookkeeping (cap lives in config.max_retries)
    retry_count: int = 0

    # Explainability / debugging
    trace: list[TraceEntry] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def add_trace(
        self, agent: str, note: str = "", tokens_used: int | None = None
    ) -> None:
        """Append an audit-log entry for an agent invocation."""
        self.trace.append(
            TraceEntry(agent=agent, note=note, tokens_used=tokens_used)
        )

    def can_retry(self) -> bool:
        """True if another self-correction pass is allowed."""
        return self.retry_count < self.config.max_retries


# --------------------------------------------------------------------------- #
# Public output (decoupled from internal state)
# --------------------------------------------------------------------------- #
class CandidateResult(BaseModel):
    """The clean, API-facing result for one resume.

    Kept separate from :class:`PipelineState` so the external contract does not
    leak internal bookkeeping (retry counts, raw text, etc.).
    """

    filename: str
    candidate_name: str | None = None
    score: float = Field(ge=0.0, le=1.0)
    verdict: Verdict
    breakdown: dict[str, float] = Field(default_factory=dict)
    gaps: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    hygiene_score: float | None = None
    hygiene_issues: list[str] = Field(default_factory=list)

    @classmethod
    def from_state(cls, state: PipelineState) -> CandidateResult:
        """Build the public result from a fully-processed pipeline state."""
        score = state.final_score.score if state.final_score else 0.0
        breakdown = state.final_score.breakdown if state.final_score else {}
        verdict: Verdict = state.critique.verdict if state.critique else "weak_fit"
        gaps = state.critique.gaps if state.critique else []
        suggestions = state.critique.suggestions if state.critique else []
        name = state.resume_structured.name if state.resume_structured else None
        hygiene_score = state.hygiene.score if state.hygiene else None
        hygiene_issues = (
            [issue.message for issue in state.hygiene.issues]
            if state.hygiene
            else []
        )
        return cls(
            filename=state.resume_raw.filename,
            candidate_name=name,
            score=score,
            verdict=verdict,
            breakdown=breakdown,
            gaps=gaps,
            suggestions=suggestions,
            hygiene_score=hygiene_score,
            hygiene_issues=hygiene_issues,
        )


class ScreeningResult(BaseModel):
    """The full response for a screening request (one or more candidates)."""

    run_id: str | None = None
    created_at: str | None = None
    mode: Mode
    job_title: str | None = None
    candidates: list[CandidateResult] = Field(default_factory=list)

    @property
    def top(self) -> CandidateResult | None:
        """The highest-scoring candidate, if any."""
        return self.candidates[0] if self.candidates else None
