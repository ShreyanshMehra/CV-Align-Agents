"""Tests for the deterministic hygiene checker."""

from __future__ import annotations

from cv_align_agents.pipeline.hygiene import check_hygiene
from cv_align_agents.state import (
    EducationItem,
    ExperienceItem,
    ProjectItem,
    StructuredResume,
)


def _strong_resume() -> StructuredResume:
    return StructuredResume(
        name="Jane Smith",
        email="jane@example-mail.org",
        skills=["Python", "Go", "Docker", "Kubernetes", "PostgreSQL"],
        experience=[
            ExperienceItem(
                company="Acme",
                role="SWE Intern",
                bullets=["Reduced latency by 40% serving 10k requests/day"],
            )
        ],
        projects=[ProjectItem(name="RAG Knowledge Assistant", tech=["Python"])],
        education=[EducationItem(degree="B.Tech CS", institute="IIT")],
    )


def _strong_raw_text() -> str:
    return (
        "Jane Smith\njane@example-mail.org\n"
        "https://github.com/jane  https://linkedin.com/in/jane\n"
    )


def test_strong_resume_scores_high_with_no_warnings():
    report = check_hygiene(_strong_resume(), _strong_raw_text())
    warnings = [i for i in report.issues if i.severity == "warning"]
    assert warnings == []
    assert report.score >= 0.9
    assert report.positives  # collected some positives


def test_missing_links_flagged():
    report = check_hygiene(_strong_resume(), raw_text="no links here")
    checks = {i.check for i in report.issues}
    assert "missing_github" in checks
    assert "no_links" in checks


def test_thin_skills_flagged():
    resume = _strong_resume()
    resume.skills = ["Python"]
    report = check_hygiene(resume, _strong_raw_text())
    assert any(i.check == "thin_skills" for i in report.issues)


def test_no_projects_flagged_as_warning():
    resume = _strong_resume()
    resume.projects = []
    report = check_hygiene(resume, _strong_raw_text())
    assert any(
        i.check == "no_projects" and i.severity == "warning"
        for i in report.issues
    )


def test_generic_project_name_flagged():
    resume = _strong_resume()
    resume.projects = [ProjectItem(name="Project 1")]
    report = check_hygiene(resume, _strong_raw_text())
    assert any(i.check == "generic_project_name" for i in report.issues)


def test_unquantified_experience_flagged():
    resume = _strong_resume()
    resume.experience = [
        ExperienceItem(company="Acme", role="Intern",
                       bullets=["Worked on backend services"])
    ]
    report = check_hygiene(resume, _strong_raw_text())
    assert any(i.check == "unquantified_experience" for i in report.issues)


def test_placeholder_url_flagged():
    resume = _strong_resume()
    report = check_hygiene(resume, raw_text="see https://github.com/x and example.com")
    assert any(i.check == "placeholder_url" for i in report.issues)


def test_score_never_negative():
    # An almost-empty resume accrues many penalties but score stays >= 0.
    report = check_hygiene(StructuredResume(), raw_text="")
    assert 0.0 <= report.score <= 1.0


def test_deterministic():
    resume = _strong_resume()
    raw = _strong_raw_text()
    assert check_hygiene(resume, raw).score == check_hygiene(resume, raw).score
