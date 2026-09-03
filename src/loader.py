import fitz
from pathlib import Path


def load_pdf(pdf_path):
    """
    Reads a PDF and extracts text page by page.
    """

    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text()

        pages.append({
            "text": text,
            "page": page_number + 1,
            "source": Path(pdf_path).name
        })

    document.close()

    return pages