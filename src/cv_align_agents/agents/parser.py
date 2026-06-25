"""Parser agent: resume text -> :class:`StructuredResume`.

This is the first real *agent*. Extracting structured fields from free-form
resume text needs judgment (section boundaries, what counts as a "skill", how to
split bullets), so we use an LLM. We rely on LangChain's
``with_structured_output`` so the model returns a validated Pydantic object
directly instead of free text we would have to parse and repair.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from cv_align_agents.llm.client import get_chat_model
from cv_align_agents.state import ResumeRaw, StructuredResume

SYSTEM_PROMPT = """\
You are a meticulous resume parser. Extract the candidate's information from the \
resume text into the required structured format.

Rules:
- Extract only what is present. Never invent or infer missing information; leave \
a field empty or null if the resume does not state it.
- `skills` must be a flat list of individual skills/technologies (split comma- \
or bullet-separated lists into separate items; do not include whole sentences).
- For `experience`, capture each role's company, title, dates, and its bullet \
points verbatim (lightly cleaned of formatting artefacts).
- For `projects`, capture the name, a short description, and the technologies \
used.
- For `education`, capture degree, institute, year, and GPA/percentage if shown.
- Preserve the candidate's wording; do not summarise or embellish.\
"""


def parse_resume(
    resume_raw: ResumeRaw,
    llm: BaseChatModel | None = None,
) -> StructuredResume:
    """Parse raw resume text into a :class:`StructuredResume`.

    Args:
        resume_raw: The resume filename + extracted text.
        llm: Optional chat model to use (dependency injection for tests). Falls
            back to the configured provider via :func:`get_chat_model`.

    Returns:
        A populated :class:`StructuredResume`. For empty input, an empty
        ``StructuredResume`` is returned without calling the LLM.
    """
    if not resume_raw.text.strip():
        return StructuredResume()

    llm = llm or get_chat_model()
    extractor = llm.with_structured_output(StructuredResume)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=resume_raw.text),
    ]
    result = extractor.invoke(messages)

    # Some providers may return a dict rather than the model instance.
    if isinstance(result, dict):
        result = StructuredResume.model_validate(result)
    return result
