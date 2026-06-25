"""Tests for PDF extraction and the resume parser agent.

These are offline tests: the parser is exercised with a fake LLM so no API key
or network is required. Real end-to-end parsing is verified via
``scripts/parse_resume.py``.
"""

from __future__ import annotations

from io import BytesIO

import pytest
from pypdf import PdfWriter

from cv_align_agents.agents.parser import parse_resume
from cv_align_agents.pdf import extract_text_from_pdf
from cv_align_agents.state import ResumeRaw, StructuredResume


# --------------------------------------------------------------------------- #
# Fake LLM that mimics the .with_structured_output(...).invoke(...) interface
# --------------------------------------------------------------------------- #
class _FakeStructuredRunnable:
    def __init__(self, result, calls):
        self._result = result
        self._calls = calls

    def invoke(self, messages):
        self._calls.append(messages)
        return self._result


class _FakeLLM:
    """Returns a preset object from with_structured_output(...).invoke(...)."""

    def __init__(self, result):
        self.result = result
        self.calls: list = []

    def with_structured_output(self, schema, **kwargs):
        return _FakeStructuredRunnable(self.result, self.calls)


# --------------------------------------------------------------------------- #
# PDF extraction
# --------------------------------------------------------------------------- #
def test_extract_text_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        extract_text_from_pdf("definitely_not_a_real_file_12345.pdf")


def test_extract_text_rejects_bad_type():
    with pytest.raises(TypeError):
        extract_text_from_pdf(12345)  # type: ignore[arg-type]


def test_extract_text_from_blank_pdf_bytes():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = BytesIO()
    writer.write(buf)

    text = extract_text_from_pdf(buf.getvalue())
    assert isinstance(text, str)  # blank page -> empty-ish string, no crash


# --------------------------------------------------------------------------- #
# Parser agent
# --------------------------------------------------------------------------- #
def test_parse_resume_passes_text_and_returns_structured():
    expected = StructuredResume(name="Bob Builder", skills=["python", "fastapi"])
    llm = _FakeLLM(expected)

    out = parse_resume(
        ResumeRaw(filename="bob.pdf", text="Bob Builder\nSkills: python, fastapi"),
        llm=llm,
    )

    assert out.name == "Bob Builder"
    assert out.skills == ["python", "fastapi"]
    # The resume text was forwarded to the model (human message present).
    assert llm.calls, "expected the LLM to be invoked"


def test_parse_resume_coerces_dict_result():
    llm = _FakeLLM({"name": "Dict Person", "skills": ["sql"]})
    out = parse_resume(ResumeRaw(filename="d.pdf", text="some text"), llm=llm)
    assert isinstance(out, StructuredResume)
    assert out.name == "Dict Person"
    assert out.skills == ["sql"]


def test_parse_resume_empty_text_skips_llm():
    class _BoomLLM:
        def with_structured_output(self, *args, **kwargs):
            raise AssertionError("LLM must not be called for empty input")

    out = parse_resume(ResumeRaw(filename="empty.pdf", text="   \n  "), llm=_BoomLLM())
    assert out == StructuredResume()
