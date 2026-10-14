
import numpy as np

from src.vector_store import (
    get_or_create_collection,
    search_vector_store,
    store_chunks,
)


def test_end_to_end_storage_and_retrieval(tmp_path):
    """Verify that a research chunk can be stored and retrieved."""

    collection = get_or_create_collection(
        persist_directory=tmp_path / "workflow_chroma_db",
        collection_name="workflow_test",
    )

    chunks = [
        {
            "text": (
                "PERCLOS measures the percentage of time "
                "that a person's eyes are closed."
            ),
            "source": "research_paper.pdf",
            "page": 3,
            "chunk_id": 0,
        },
        {
            "text": (
                "Convolutional neural networks are used "
                "for image classification tasks."
            ),
            "source": "computer_vision.pdf",
            "page": 7,
            "chunk_id": 0,
        },
    ]

    embeddings = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    ]

    stored_count = store_chunks(
        collection=collection,
        chunks=chunks,
        embeddings=embeddings,
    )

    assert stored_count == 2
    assert collection.count() == 2

    results = search_vector_store(
        collection=collection,
        query_embedding=embeddings[0],
        top_k=2,
    )

    assert len(results) == 2

    best_result = results[0]

    assert (
        best_result["chunk"]["text"]
        == chunks[0]["text"]
    )
    assert best_result["chunk"]["source"] == "research_paper.pdf"
    assert best_result["chunk"]["page"] == 3
    assert best_result["chunk"]["chunk_id"] == 0
    assert best_result["score"] > 0.99


def test_end_to_end_retrieval_preserves_metadata_for_multiple_queries(
    tmp_path,
):
    """Verify retrieval relevance and metadata across different queries."""

    collection = get_or_create_collection(
        persist_directory=tmp_path / "multi_query_chroma_db",
        collection_name="multi_query_test",
    )

    chunks = [
        {
            "text": "PERCLOS measures prolonged eye closure.",
            "source": "drowsiness.pdf",
            "page": 4,
            "chunk_id": 0,
        },
        {
            "text": "Convolutional neural networks classify images.",
            "source": "vision.pdf",
            "page": 8,
            "chunk_id": 1,
        },
    ]

    embeddings = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    ]

    store_chunks(collection, chunks, embeddings)

    test_queries = [
        (embeddings[0], "drowsiness.pdf", 4),
        (embeddings[1], "vision.pdf", 8),
    ]

    for query_embedding, expected_source, expected_page in test_queries:
        results = search_vector_store(
            collection=collection,
            query_embedding=query_embedding,
            top_k=1,
        )

        assert len(results) == 1
        assert results[0]["chunk"]["source"] == expected_source
        assert results[0]["chunk"]["page"] == expected_page
        assert results[0]["score"] > 0.99
