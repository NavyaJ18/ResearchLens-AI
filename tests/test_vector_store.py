
import numpy as np
import pytest

from src.vector_store import (
    create_chunk_id,
    get_or_create_collection,
    get_missing_chunks,
    store_chunks,
    search_vector_store,
)


@pytest.fixture
def collection(tmp_path):
    """Create an isolated temporary ChromaDB collection."""
    return get_or_create_collection(
        persist_directory=tmp_path / "test_chroma_db",
        collection_name="test_collection",
    )


@pytest.fixture
def sample_chunk():
    """Return a valid document chunk."""
    return {
        "text": "PERCLOS measures the percentage of time the eyes are closed.",
        "source": "test_paper.pdf",
        "page": 1,
        "chunk_id": 0,
    }


@pytest.fixture
def sample_embedding():
    """Return a normalized three-dimensional embedding."""
    return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def test_create_chunk_id_is_stable(sample_chunk):
    assert create_chunk_id(sample_chunk) == create_chunk_id(sample_chunk)


def test_create_chunk_id_changes_with_text(sample_chunk):
    original_id = create_chunk_id(sample_chunk)

    modified_chunk = sample_chunk.copy()
    modified_chunk["text"] = "A different research passage."

    assert create_chunk_id(modified_chunk) != original_id


def test_create_chunk_id_rejects_missing_fields():
    with pytest.raises(ValueError, match="missing required fields"):
        create_chunk_id({"text": "Some text"})


def test_create_chunk_id_rejects_empty_text(sample_chunk):
    invalid_chunk = sample_chunk.copy()
    invalid_chunk["text"] = "   "

    with pytest.raises(ValueError, match="non-empty string"):
        create_chunk_id(invalid_chunk)


def test_store_chunks_and_count(
    collection, sample_chunk, sample_embedding
):
    stored = store_chunks(
        collection, [sample_chunk], [sample_embedding]
    )

    assert stored == 1
    assert collection.count() == 1


def test_store_chunks_rejects_mismatched_embeddings(
    collection, sample_chunk
):
    with pytest.raises(ValueError, match="one embedding"):
        store_chunks(collection, [sample_chunk], [])


def test_store_chunks_rejects_non_finite_embeddings(
    collection, sample_chunk
):
    invalid_embedding = np.array([1.0, np.nan, 0.0])

    with pytest.raises(ValueError, match="finite numeric values"):
        store_chunks(
            collection, [sample_chunk], [invalid_embedding]
        )


def test_get_missing_chunks_detects_stored_chunks(
    collection, sample_chunk, sample_embedding
):
    store_chunks(collection, [sample_chunk], [sample_embedding])

    assert get_missing_chunks(collection, [sample_chunk]) == []


def test_get_missing_chunks_finds_new_chunks(
    collection, sample_chunk, sample_embedding
):
    store_chunks(collection, [sample_chunk], [sample_embedding])

    new_chunk = sample_chunk.copy()
    new_chunk["chunk_id"] = 1
    new_chunk["text"] = "A new passage from the paper."

    missing = get_missing_chunks(
        collection, [sample_chunk, new_chunk]
    )

    assert len(missing) == 1
    assert missing[0]["chunk_id"] == 1


def test_search_returns_relevant_chunk(
    collection, sample_chunk, sample_embedding
):
    store_chunks(collection, [sample_chunk], [sample_embedding])

    results = search_vector_store(
        collection, query_embedding=sample_embedding, top_k=1
    )

    assert len(results) == 1
    assert results[0]["chunk"]["text"] == sample_chunk["text"]
    assert results[0]["chunk"]["source"] == "test_paper.pdf"
    assert results[0]["chunk"]["page"] == 1
    assert results[0]["score"] == pytest.approx(1.0)


def test_search_returns_empty_for_empty_collection(collection):
    results = search_vector_store(
        collection,
        query_embedding=np.array([1.0, 0.0, 0.0]),
        top_k=3,
    )

    assert results == []


@pytest.mark.parametrize("invalid_top_k", [0, -1, 1.5, True])
def test_search_rejects_invalid_top_k(collection, invalid_top_k):
    with pytest.raises(ValueError, match="positive integer"):
        search_vector_store(
            collection,
            query_embedding=np.array([1.0, 0.0, 0.0]),
            top_k=invalid_top_k,
        )


def test_search_filters_results_below_similarity_threshold(
    collection, sample_chunk
):
    """A lower-similarity chunk should be excluded by the threshold."""
    embeddings = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    ]

    second_chunk = sample_chunk.copy()
    second_chunk["chunk_id"] = 1
    second_chunk["text"] = "A different passage about another topic."

    store_chunks(
        collection,
        [sample_chunk, second_chunk],
        embeddings,
    )

    results = search_vector_store(
        collection,
        query_embedding=embeddings[0],
        top_k=2,
        min_similarity=0.8,
    )

    assert len(results) == 1
    assert results[0]["chunk"]["chunk_id"] == 0
    assert results[0]["score"] >= 0.8


def test_search_without_threshold_preserves_results(
    collection, sample_chunk, sample_embedding
):
    store_chunks(collection, [sample_chunk], [sample_embedding])

    results = search_vector_store(
        collection,
        query_embedding=sample_embedding,
        top_k=3,
    )

    assert len(results) == 1


@pytest.mark.parametrize(
    "invalid_threshold",
    [True, float("nan"), 1.1, -1.1, "0.5"],
)
def test_search_rejects_invalid_similarity_threshold(
    collection, invalid_threshold
):
    with pytest.raises(ValueError, match="min_similarity"):
        search_vector_store(
            collection,
            query_embedding=np.array([1.0, 0.0, 0.0]),
            min_similarity=invalid_threshold,
        )
