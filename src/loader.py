
from pathlib import Path

import pymupdf


def load_pdf(pdf_path):
    """Extract text page by page from a PDF file."""

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if not path.is_file():
        raise ValueError(f"PDF path is not a file: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.name}")

    try:
        with pymupdf.open(path) as document:
            if document.is_encrypted:
                raise ValueError(f"PDF is encrypted: {path.name}")

            pages = []

            for page_number, page in enumerate(document, start=1):
                pages.append(
                    {
                        "text": page.get_text(),
                        "page": page_number,
                        "source": path.name,
                    }
                )

            return pages

    except (FileNotFoundError, ValueError):
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read PDF '{path.name}': {exc}"
        ) from exc
