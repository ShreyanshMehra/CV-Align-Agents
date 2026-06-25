"""Deterministic PDF -> text extraction.

This is a plain function, not an agent: turning a PDF into text needs no
judgment, so we keep it cheap and reproducible with ``pypdf``. The resulting
text is what the parser *agent* later turns into structured fields.
"""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


class PDFExtractionError(RuntimeError):
    """Raised when a PDF cannot be read or its text cannot be extracted."""


def extract_text_from_pdf(source: str | Path | bytes) -> str:
    """Extract plain text from a PDF given a file path or raw bytes.

    Args:
        source: A path (``str``/``Path``) to a PDF file, or the PDF's raw bytes
            (as received from a file upload).

    Returns:
        The concatenated text of all pages, with excessive blank lines collapsed.
        May be an empty string for scanned/image-only PDFs.

    Raises:
        FileNotFoundError: If a path is given but no file exists there.
        TypeError: If ``source`` is neither a path nor bytes.
        PDFExtractionError: If the PDF is malformed or unreadable.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
        reader_source: str | BytesIO = str(path)
    elif isinstance(source, (bytes, bytearray)):
        reader_source = BytesIO(bytes(source))
    else:
        raise TypeError(
            f"Expected a path or bytes, got {type(source).__name__}."
        )

    try:
        reader = PdfReader(reader_source)
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 - surface any pypdf failure clearly
        raise PDFExtractionError(f"Failed to read PDF: {exc}") from exc

    text = "\n".join(pages)
    # Collapse 3+ consecutive newlines into a blank-line separator.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
