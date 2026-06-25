"""Parse a real resume PDF end to end and print the structured result.

Usage (from the project root, with the virtualenv active):

    python scripts/parse_resume.py path/to/resume.pdf

Extracts text from the PDF (deterministic) and runs the parser agent against the
configured LLM provider. Prints the structured resume as JSON. Never prints your
API key.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

from cv_align_agents.agents.parser import parse_resume
from cv_align_agents.pdf import PDFExtractionError, extract_text_from_pdf
from cv_align_agents.state import ResumeRaw


def main(argv: list[str]) -> int:
    load_dotenv()

    if len(argv) < 2:
        print("Usage: python scripts/parse_resume.py <resume.pdf>")
        return 2

    pdf_path = Path(argv[1])
    try:
        text = extract_text_from_pdf(pdf_path)
    except (FileNotFoundError, PDFExtractionError) as exc:
        print(f"[PDF ERROR] {exc}")
        return 1

    print(f"Extracted {len(text)} characters from {pdf_path.name}\n")
    if not text.strip():
        print("No extractable text found (is this a scanned/image-only PDF?).")
        return 1

    print("Parsing with the configured LLM ...\n")
    structured = parse_resume(ResumeRaw(filename=pdf_path.name, text=text))
    print(structured.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
