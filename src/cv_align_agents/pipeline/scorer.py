"""Deterministic scorer: combine matcher sub-scores into a final score.

This is intentionally a plain function, not an agent. Keeping the final score
deterministic means it is reproducible, unit-testable, and tunable (via weights)
without touching any LLM prompt. Given the same ``MatchResult`` and weights, the
output never changes.

    final_score = Σ weightᵢ · sub_scoreᵢ      (sections i)

Because ``PipelineConfig`` normalises weights to sum to 1.0 and every sub-score
is in [0, 1], the final score is always in [0, 1].
"""

from __future__ import annotations

from cv_align_agents.state import (
    SECTIONS,
    FinalScore,
    MatchResult,
    PipelineConfig,
)


def score(match_result: MatchResult, config: PipelineConfig) -> FinalScore:
    """Compute the deterministic weighted final score.

    Args:
        match_result: Per-section sub-scores from the matcher.
        config: Pipeline config supplying the (already-normalised) section
            weights.

    Returns:
        A :class:`FinalScore` with the overall score, the per-section breakdown,
        and the exact weights used (useful for explainability/auditing).
    """
    breakdown: dict[str, float] = {
        sub.section: sub.score
        for sub in match_result.sub_scores
        if sub.section in SECTIONS
    }
    # Any section the matcher omitted contributes 0.
    for section in SECTIONS:
        breakdown.setdefault(section, 0.0)

    weights = config.weights
    total = sum(weights.get(section, 0.0) * breakdown[section] for section in SECTIONS)
    # Clamp against tiny floating-point drift so the result stays within [0, 1].
    total = max(0.0, min(1.0, total))

    return FinalScore(
        score=total,
        weights_used=dict(weights),
        breakdown=breakdown,
    )
