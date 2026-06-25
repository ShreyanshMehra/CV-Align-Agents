"""Deterministic resume hygiene checks.

Objective, zero-LLM checks that flag common resume problems independent of any
job description: missing contact links, unquantified experience, generic project
names, thin skill lists, and so on. The result is advisory — it informs the
critic's suggestions and is surfaced to candidates, but it does not change the
JD-match score.

Being deterministic, these checks are cheap, reproducible, and easy to defend
("we combine objective rule-based checks with LLM judgment, not one big prompt").
The concept is inspired by HackerRank's open-source hiring-agent bonus/deduction
idea; the checks here are an independent, clean-room implementation.
"""

from __future__ import annotations

import re

from cv_align_agents.state import (
    HygieneIssue,
    HygieneReport,
    StructuredResume,
)

# Penalty applied to the hygiene score per issue, by severity.
_PENALTY = {"warning": 0.15, "info": 0.05}

# Minimum number of skills before we consider the list "thin".
_MIN_SKILLS = 5

_URL_RE = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)
_GITHUB_RE = re.compile(r"github\.com/", re.IGNORECASE)
_LINKEDIN_RE = re.compile(r"linkedin\.com/", re.IGNORECASE)
_DIGIT_RE = re.compile(r"\d")
_PLACEHOLDER_RE = re.compile(
    r"example\.com|localhost|your-?website|todo|lorem ipsum|xxx", re.IGNORECASE
)
_GENERIC_NAME_RE = re.compile(
    r"^\s*(project\s*\d*|final\s*(year)?\s*project|minor\s*project|"
    r"major\s*project|untitled|test|demo|sample)\s*$",
    re.IGNORECASE,
)


def check_hygiene(
    resume: StructuredResume,
    raw_text: str = "",
) -> HygieneReport:
    """Run deterministic hygiene checks over a resume.

    Args:
        resume: The structured resume.
        raw_text: The original resume text (used for URL/link detection, which is
            more reliable on the raw text than on the structured fields).

    Returns:
        A :class:`HygieneReport` with a 0-1 hygiene score, a list of objective
        issues, and a list of positives.
    """
    issues: list[HygieneIssue] = []
    positives: list[str] = []

    # Combined text gives link checks the best chance of finding a URL.
    combined = f"{raw_text}\n{resume.model_dump_json()}"

    _check_contact(resume, combined, issues, positives)
    _check_links(combined, issues, positives)
    _check_placeholder_urls(combined, issues)
    _check_skills(resume, issues, positives)
    _check_projects(resume, issues, positives)
    _check_quantified_experience(resume, issues, positives)
    _check_education(resume, issues)

    penalty = sum(_PENALTY[i.severity] for i in issues)
    hygiene_score = max(0.0, min(1.0, 1.0 - penalty))

    return HygieneReport(score=hygiene_score, issues=issues, positives=positives)


def _add(issues: list[HygieneIssue], check: str, severity: str, message: str) -> None:
    issues.append(HygieneIssue(check=check, severity=severity, message=message))


def _check_contact(resume, combined, issues, positives) -> None:
    if not resume.email and "@" not in combined:
        _add(
            issues,
            "missing_email",
            "warning",
            "No email address found; recruiters need a way to contact you.",
        )
    elif resume.email:
        positives.append("Includes a contact email.")


def _check_links(combined, issues, positives) -> None:
    has_github = bool(_GITHUB_RE.search(combined))
    has_linkedin = bool(_LINKEDIN_RE.search(combined))
    has_any_url = bool(_URL_RE.search(combined))

    if has_github:
        positives.append("Links to a GitHub profile/projects.")
    if has_linkedin:
        positives.append("Links to a LinkedIn profile.")

    if not has_github:
        _add(
            issues,
            "missing_github",
            "warning",
            "No GitHub link found; add one so reviewers can see your code.",
        )
    if not has_linkedin:
        _add(
            issues,
            "missing_linkedin",
            "info",
            "No LinkedIn link found; consider adding your profile URL.",
        )
    if not has_any_url and not has_github and not has_linkedin:
        _add(
            issues,
            "no_links",
            "warning",
            "No links at all; add GitHub, a portfolio, or live project URLs.",
        )


def _check_placeholder_urls(combined, issues) -> None:
    if _PLACEHOLDER_RE.search(combined):
        _add(
            issues,
            "placeholder_url",
            "warning",
            "Found a placeholder/example link (e.g. example.com or TODO); "
            "replace it with a real URL.",
        )


def _check_skills(resume, issues, positives) -> None:
    n = len(resume.skills)
    if n == 0:
        _add(
            issues,
            "no_skills",
            "warning",
            "No skills listed; add a skills section with concrete technologies.",
        )
    elif n < _MIN_SKILLS:
        _add(
            issues,
            "thin_skills",
            "info",
            f"Only {n} skill(s) listed; consider expanding to show breadth.",
        )
    else:
        positives.append(f"Lists {n} skills.")


def _check_projects(resume, issues, positives) -> None:
    if not resume.projects:
        _add(
            issues,
            "no_projects",
            "warning",
            "No projects listed; projects are strong signal for most roles.",
        )
        return

    generic = [
        p.name
        for p in resume.projects
        if p.name and _GENERIC_NAME_RE.match(p.name)
    ]
    if generic:
        _add(
            issues,
            "generic_project_name",
            "info",
            "Generic project name(s) found "
            f"({', '.join(generic)}); use specific, descriptive titles.",
        )
    else:
        positives.append("Project names are specific.")


def _check_quantified_experience(resume, issues, positives) -> None:
    if not resume.experience:
        return  # Absence of experience is judged by the matcher, not hygiene.

    has_quantified = any(
        _DIGIT_RE.search(bullet)
        for exp in resume.experience
        for bullet in exp.bullets
    )
    if has_quantified:
        positives.append("Experience bullets include quantified impact.")
    else:
        _add(
            issues,
            "unquantified_experience",
            "info",
            "Experience lacks numbers; quantify impact (e.g. 'cut latency 40%').",
        )


def _check_education(resume, issues) -> None:
    if not resume.education:
        _add(
            issues,
            "missing_education",
            "info",
            "No education section detected.",
        )
