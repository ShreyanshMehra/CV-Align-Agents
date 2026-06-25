"""JD parser agent: job-description text -> :class:`StructuredJD`.

Mirrors the resume parser: extracting structured requirements from a free-form
job description needs judgment (what is *required* vs *nice-to-have*, how many
years of experience are implied), so we use an LLM with structured output.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from cv_align_agents.llm.client import get_chat_model
from cv_align_agents.state import JDRaw, StructuredJD

SYSTEM_PROMPT = """\
You are an expert technical recruiter. Extract the hiring requirements from the \
job description into the required structured format.

Rules:
- Extract only what the job description states; do not invent requirements.
- `required_skills`: skills/technologies the candidate MUST have (listed as \
"required", "must have", or clearly essential). Use a flat list of individual \
items.
- `nice_to_have_skills`: skills described as "preferred", "bonus", "a plus", or \
"nice to have". Do not duplicate items already in `required_skills`.
- `min_experience_years`: the minimum years of experience as a number if stated \
(e.g. "3+ years" -> 3). Leave null if not specified.
- `responsibilities`: what the person will do in the role.
- `qualifications`: education, certifications, or background requirements.
- Keep each list item concise and specific.\
"""


def parse_jd(
    jd_raw: JDRaw,
    llm: BaseChatModel | None = None,
) -> StructuredJD:
    """Parse a raw job description into a :class:`StructuredJD`.

    Args:
        jd_raw: The raw job-description text.
        llm: Optional chat model (dependency injection for tests). Falls back to
            the configured provider via :func:`get_chat_model`.

    Returns:
        A populated :class:`StructuredJD`. For empty input, an empty
        ``StructuredJD`` is returned without calling the LLM.
    """
    if not jd_raw.text.strip():
        return StructuredJD()

    llm = llm or get_chat_model()
    extractor = llm.with_structured_output(StructuredJD)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=jd_raw.text),
    ]
    result = extractor.invoke(messages)

    if isinstance(result, dict):
        result = StructuredJD.model_validate(result)
    return result
