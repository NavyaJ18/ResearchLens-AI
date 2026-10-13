
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
