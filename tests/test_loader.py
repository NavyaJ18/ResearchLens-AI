
import pymupdf
import pytest

from src.loader import load_pdf


def test_load_pdf_extracts_text_and_metadata(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "PERCLOS measures eye closure.")
        document.save(pdf_path)

    pages = load_pdf(pdf_path)

    assert len(pages) == 1
    assert "PERCLOS measures eye closure." in pages[0]["text"]
    assert pages[0]["page"] == 1
    assert pages[0]["source"] == "sample.pdf"


def test_load_pdf_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_pdf(tmp_path / "missing.pdf")


def test_load_pdf_rejects_non_pdf_file(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("Not a PDF", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a PDF"):
        load_pdf(file_path)


def test_load_pdf_rejects_directory(tmp_path):
    with pytest.raises(ValueError, match="not a file"):
        load_pdf(tmp_path)


def test_load_pdf_rejects_corrupted_pdf(tmp_path):
    pdf_path = tmp_path / "corrupted.pdf"
    pdf_path.write_text("This is not a valid PDF", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Failed to read PDF"):
        load_pdf(pdf_path)
