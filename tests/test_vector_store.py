
import numpy as np
import pytest
import chromadb

from src.vector_store import (
    create_chunk_id,
    get_or_create_collection,
    get_missing_chunks,
    store_chunks,
    search_vector_store,
)


@pytest.fixture
def collection(tmp_path):
    """Create an isolated temporary ChromaDB collection for each test."""
    return get_or_create_collection(
        persist_directory=tmp_path / "test_chroma_db",
        collection_name="test_collection",
    )


@pytest.fixture
def sample_chunk():
    """Provide a valid document chunk."""
    return {
        "text": "PERCLOS measures the percentage of time the eyes are closed.",
        "source": "test_paper.pdf",
        "page": 1,
        "chunk_id": 0,
    }


@pytest.fixture
def sample_embedding():
    """Provide a normalized embedding for testing."""
    return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def test_create_chunk_id_is_stable(sample_chunk):
    """The same chunk should always produce the same ID."""
    assert create_chunk_id(sample_chunk) == create_chunk_id(sample_chunk)


def test_create_chunk_id_changes_with_text(sample_chunk):
    """Changing chunk text should change its ID."""
    original_id = create_chunk_id(sample_chunk)

    modified_chunk = sample_chunk.copy()
    modified_chunk["text"] = "A different research passage."

    assert create_chunk_id(modified_chunk) != original_id


def test_create_chunk_id_rejects_missing_fields():
    """Malformed chunks should raise a clear validation error."""
    with pytest.raises(ValueError, match="missing required fields"):
        create_chunk_id({"text": "Some text"})


def test_create_chunk_id_rejects_empty_text(sample_chunk):
    """Empty chunk text should not be accepted."""
    invalid_chunk = sample_chunk.copy()
    invalid_chunk["text"] = "   "

    with pytest.raises(ValueError, match="non-empty string"):
        create_chunk_id(invalid_chunk)


def test_store_chunks_and_count(
    collection, sample_chunk, sample_embedding
):
    """Storing a chunk should increase the collection count."""
    stored = store_chunks(
        collection,
        [sample_chunk],
        [sample_embedding],
    )

    assert stored == 1
    assert collection.count() == 1


def test_store_chunks_rejects_mismatched_embeddings(
    collection, sample_chunk
):
    """Each chunk must have a matching embedding."""
    with pytest.raises(ValueError, match="one embedding"):
        store_chunks(collection, [sample_chunk], [])


def test_get_missing_chunks_detects_stored_chunks(
    collection, sample_chunk, sample_embedding
):
    """Stored chunks should not be reported as missing."""
    store_chunks(collection, [sample_chunk], [sample_embedding])

    missing = get_missing_chunks(collection, [sample_chunk])

    assert missing == []


def test_get_missing_chunks_finds_new_chunks(
    collection, sample_chunk, sample_embedding
):
    """Chunks not yet stored should be reported as missing."""
    store_chunks(collection, [sample_chunk], [sample_embedding])

    new_chunk = sample_chunk.copy()
    new_chunk["chunk_id"] = 1
    new_chunk["text"] = "A new passage from the paper."

    missing = get_missing_chunks(
        collection,
        [sample_chunk, new_chunk],
    )

    assert len(missing) == 1
    assert missing[0]["chunk_id"] == 1


def test_search_returns_relevant_chunk(
    collection, sample_chunk, sample_embedding
):
    """Search should return the stored chunk and its metadata."""
    store_chunks(collection, [sample_chunk], [sample_embedding])

    results = search_vector_store(
        collection,
        query_embedding=sample_embedding,
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["chunk"]["text"] == sample_chunk["text"]
    assert results[0]["chunk"]["source"] == "test_paper.pdf"
    assert results[0]["chunk"]["page"] == 1
    assert results[0]["score"] == pytest.approx(1.0)


def test_search_returns_empty_for_empty_collection(collection):
    """Searching an empty collection should return no results."""
    results = search_vector_store(
        collection,
        query_embedding=np.array([1.0, 0.0, 0.0]),
        top_k=3,
    )

    assert results == []


@pytest.mark.parametrize("invalid_top_k", [0, -1, 1.5, True])
def test_search_rejects_invalid_top_k(collection, invalid_top_k):
    """Search result count must be a positive integer."""
    with pytest.raises(ValueError, match="positive integer"):
        search_vector_store(
            collection,
            query_embedding=np.array([1.0, 0.0, 0.0]),
            top_k=invalid_top_k,
        )


def test_store_chunks_rejects_non_finite_embeddings(
    collection, sample_chunk
):
    """Embeddings containing NaN should be rejected."""
    invalid_embedding = np.array([1.0, np.nan, 0.0])

    with pytest.raises(ValueError, match="finite numeric values"):
        store_chunks(
            collection,
            [sample_chunk],
            [invalid_embedding],
        )
