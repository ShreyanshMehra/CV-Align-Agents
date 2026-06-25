"""Run the full screening pipeline end to end on a resume PDF + a job description.

Usage (from the project root, with the virtualenv active):

    python scripts/screen.py path/to/resume.pdf path/to/job.txt
    python scripts/screen.py path/to/resume.pdf --jd "inline job description text"

Extracts text from the PDF, runs the LangGraph pipeline against the configured
LLM, and prints the candidate result plus the agent trace. Never prints your
API key.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

from cv_align_agents.pdf import PDFExtractionError, extract_text_from_pdf
from cv_align_agents.pipeline.graph import run_pipeline
from cv_align_agents.state import CandidateResult, JDRaw, PipelineConfig, ResumeRaw


def _read_jd(args: list[str]) -> str | None:
    if "--jd" in args:
        idx = args.index("--jd")
        return args[idx + 1] if idx + 1 < len(args) else None
    if len(args) >= 3:
        jd_path = Path(args[2])
        if jd_path.is_file():
            return jd_path.read_text(encoding="utf-8")
    return None


def main(argv: list[str]) -> int:
    load_dotenv()

    if len(argv) < 3:
        print("Usage: python scripts/screen.py <resume.pdf> <job.txt | --jd 'text'>")
        return 2

    resume_path = Path(argv[1])
    try:
        resume_text = extract_text_from_pdf(resume_path)
    except (FileNotFoundError, PDFExtractionError) as exc:
        print(f"[PDF ERROR] {exc}")
        return 1

    jd_text = _read_jd(argv)
    if not jd_text:
        print("No job description provided (give a .txt path or --jd 'text').")
        return 2

    print(f"Screening {resume_path.name} ...\n")
    state = run_pipeline(
        ResumeRaw(filename=resume_path.name, text=resume_text),
        JDRaw(text=jd_text),
        PipelineConfig(mode="candidate"),
    )

    result = CandidateResult.from_state(state)
    print("=" * 60)
    print(result.model_dump_json(indent=2))
    print("=" * 60)
    print("\nAgent trace:")
    for entry in state.trace:
        note = f" — {entry.note}" if entry.note else ""
        print(f"  [{entry.agent}]{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
