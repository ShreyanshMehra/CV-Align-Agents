"""FastAPI application exposing the screening pipeline.

Endpoints:

- ``GET  /health`` — liveness check.
- ``POST /screen`` — upload one or more resume PDFs plus a job description and
  receive ranked, explainable results.

The persona is selected with the ``mode`` field: ``candidate`` (CV feedback) or
``recruiter`` (ranked screening).
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

from cv_align_agents import __version__
from cv_align_agents.llm.client import LLMConfigError
from cv_align_agents.pdf import PDFExtractionError, extract_text_from_pdf
from cv_align_agents.pipeline.screen import screen
from cv_align_agents.settings import get_settings
from cv_align_agents.state import (
    JDRaw,
    PipelineConfig,
    ResumeRaw,
    ScreeningResult,
)
from cv_align_agents.storage.runs import RunStore

load_dotenv()

app = FastAPI(
    title="CV-Align-Agents",
    version=__version__,
    summary="Multi-agent resume screening (LangGraph + Gemini).",
)

# Serve the static frontend (HTML/CSS/JS) bundled alongside the package.
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def get_llm() -> BaseChatModel | None:
    """LLM dependency. Returns ``None`` so agents use the configured provider.

    Tests override this via ``app.dependency_overrides`` to inject a fake model.
    """
    return None


@lru_cache(maxsize=1)
def _default_store() -> RunStore:
    return RunStore(get_settings().db_path)


def get_store() -> RunStore:
    """Run-store dependency. Tests override this with a temp-file store."""
    return _default_store()


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    # Serve the single-page frontend. The interactive API docs stay at /docs.
    return FileResponse(WEB_DIR / "index.html")


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
    store: RunStore = Depends(get_store),
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
        result = await screen(resume_inputs, JDRaw(text=jd), config, llm=llm)
    except LLMConfigError as exc:
        # Misconfiguration (e.g. missing API key) -> 503 with a clear message.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    await asyncio.to_thread(store.save, result)  # sets run_id + created_at
    return result


@app.get("/runs/{run_id}", response_model=ScreeningResult)
async def get_run(
    run_id: str,
    store: RunStore = Depends(get_store),
) -> ScreeningResult:
    result = await asyncio.to_thread(store.get, run_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return result


@app.get("/runs")
async def list_runs(
    limit: int = 50,
    store: RunStore = Depends(get_store),
) -> dict:
    runs = await asyncio.to_thread(store.list_runs, limit)
    return {"runs": runs}
