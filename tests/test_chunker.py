
import pytest

from src.chunker import chunk_pages


@pytest.fixture
def sample_pages():
    return [
        {
            "text": "A" * 12,
            "page": 1,
            "source": "paper.pdf",
        }
    ]


def test_chunk_pages_preserves_metadata(sample_pages):
    chunks = chunk_pages(
        sample_pages,
        chunk_size=5,
        chunk_overlap=1,
    )

    assert [chunk["text"] for chunk in chunks] == [
        "AAAAA",
        "AAAAA",
        "AAAA",
    ]
    assert all(chunk["page"] == 1 for chunk in chunks)
    assert all(chunk["source"] == "paper.pdf" for chunk in chunks)
    assert [chunk["chunk_id"] for chunk in chunks] == [0, 1, 2]


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True, "10"])
def test_rejects_invalid_chunk_size(sample_pages, chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_pages(sample_pages, chunk_size=chunk_size)


@pytest.mark.parametrize("overlap", [-1, 1.5, True, "2"])
def test_rejects_invalid_overlap(sample_pages, overlap):
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_pages(
            sample_pages,
            chunk_size=5,
            chunk_overlap=overlap,
        )


def test_rejects_overlap_equal_to_chunk_size(sample_pages):
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        chunk_pages(
            sample_pages,
            chunk_size=5,
            chunk_overlap=5,
        )


@pytest.mark.parametrize("pages", [None, "invalid", 123])
def test_rejects_invalid_pages_container(pages):
    with pytest.raises(ValueError, match="pages"):
        chunk_pages(pages)


def test_rejects_page_missing_required_fields():
    with pytest.raises(ValueError, match="missing fields"):
        chunk_pages([{"text": "Some text"}])


def test_rejects_non_string_page_text():
    page = {"text": 123, "page": 1, "source": "paper.pdf"}

    with pytest.raises(ValueError, match="must be a string"):
        chunk_pages([page])


def test_rejects_invalid_page_number():
    page = {"text": "Some text", "page": True, "source": "paper.pdf"}

    with pytest.raises(ValueError, match="positive integer"):
        chunk_pages([page])


def test_skips_empty_pages():
    pages = [
        {"text": "  ", "page": 1, "source": "paper.pdf"},
        {"text": "Valid text", "page": 2, "source": "paper.pdf"},
    ]

    chunks = chunk_pages(pages)

    assert len(chunks) == 1
    assert chunks[0]["page"] == 2
