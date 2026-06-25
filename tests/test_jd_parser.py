"""Tests for the JD parser agent (offline, using a fake LLM)."""

from __future__ import annotations

from cv_align_agents.agents.jd_parser import parse_jd
from cv_align_agents.state import JDRaw, StructuredJD


class _FakeStructuredRunnable:
    def __init__(self, result, calls):
        self._result = result
        self._calls = calls

    def invoke(self, messages):
        self._calls.append(messages)
        return self._result


class _FakeLLM:
    def __init__(self, result):
        self.result = result
        self.calls: list = []

    def with_structured_output(self, schema, **kwargs):
        return _FakeStructuredRunnable(self.result, self.calls)


def test_parse_jd_returns_structured():
    expected = StructuredJD(
        title="Backend Engineer",
        required_skills=["python", "postgresql"],
        nice_to_have_skills=["kubernetes"],
        min_experience_years=3,
    )
    llm = _FakeLLM(expected)

    out = parse_jd(JDRaw(text="We need a backend engineer with Python..."), llm=llm)

    assert out.title == "Backend Engineer"
    assert out.required_skills == ["python", "postgresql"]
    assert out.min_experience_years == 3
    assert llm.calls, "expected the LLM to be invoked"


def test_parse_jd_coerces_dict_result():
    llm = _FakeLLM({"title": "ML Engineer", "required_skills": ["pytorch"]})
    out = parse_jd(JDRaw(text="ML role"), llm=llm)
    assert isinstance(out, StructuredJD)
    assert out.title == "ML Engineer"
    assert out.required_skills == ["pytorch"]


def test_parse_jd_empty_text_skips_llm():
    class _BoomLLM:
        def with_structured_output(self, *args, **kwargs):
            raise AssertionError("LLM must not be called for empty input")

    out = parse_jd(JDRaw(text="   "), llm=_BoomLLM())
    assert out == StructuredJD()
