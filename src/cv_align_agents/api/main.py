"""FastAPI application exposing the screening pipeline.

Endpoints:

- ``GET  /health`` — liveness check.
- ``POST /screen`` — upload one or more resume PDFs plus a job description and
  receive ranked, explainable results.

The persona is selected with the ``mode`` field: ``candidate`` (CV feedback) or
``recruiter`` (ranked screening).
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from cv_align_agents import __version__
from cv_align_agents.llm.client import LLMConfigError
from cv_align_agents.pdf import PDFExtractionError, extract_text_from_pdf
from cv_align_agents.pipeline.screen import screen
from cv_align_agents.state import (
    JDRaw,
    PipelineConfig,
    ResumeRaw,
    ScreeningResult,
)

load_dotenv()

app = FastAPI(
    title="CV-Align-Agents",
    version=__version__,
    summary="Multi-agent resume screening (LangGraph + Gemini).",
)


def get_llm() -> BaseChatModel | None:
    """LLM dependency. Returns ``None`` so agents use the configured provider.

    Tests override this via ``app.dependency_overrides`` to inject a fake model.
    """
    return None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/screen", response_model=ScreeningResult)
async def screen_endpoint(
    jd: str = Form(..., description="Job description text."),
    mode: str = Form("recruiter", description="'candidate' or 'recruiter'."),
    critic_mode: str = Form("full", description="'fast' or 'full'."),
    critic_top_k: int = Form(5, ge=1),
    resumes: list[UploadFile] = File(..., description="Resume PDF file(s)."),
    llm: BaseChatModel | None = Depends(get_llm),
) -> ScreeningResult:
    if not jd.strip():
        raise HTTPException(status_code=422, detail="Job description is empty.")
    if not resumes:
        raise HTTPException(status_code=422, detail="At least one resume is required.")

    # Build config first so invalid mode/critic_mode is reported clearly.
    try:
        config = PipelineConfig(
            mode=mode,  # type: ignore[arg-type]
            critic_mode=critic_mode,  # type: ignore[arg-type]
            critic_top_k=critic_top_k,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    resume_inputs: list[ResumeRaw] = []
    for upload in resumes:
        raw = await upload.read()
        try:
            text = extract_text_from_pdf(raw)
        except PDFExtractionError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Could not read '{upload.filename}': {exc}",
            ) from exc
        resume_inputs.append(
            ResumeRaw(filename=upload.filename or "resume.pdf", text=text)
        )

    try:
        return await screen(resume_inputs, JDRaw(text=jd), config, llm=llm)
    except LLMConfigError as exc:
        # Misconfiguration (e.g. missing API key) -> 503 with a clear message.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
