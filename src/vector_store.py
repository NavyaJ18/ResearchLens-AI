
import hashlib
import math

import chromadb


def get_or_create_collection(persist_directory, collection_name):
    """Open or create a persistent ChromaDB collection."""
    if (
        not isinstance(collection_name, str)
        or not collection_name.strip()
    ):
        raise ValueError("Collection name must be a non-empty string.")

    client = chromadb.PersistentClient(path=str(persist_directory))

    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def create_chunk_id(chunk):
    """Generate a stable ID for a document chunk."""
    required_fields = ("text", "source", "page", "chunk_id")

    if not isinstance(chunk, dict):
        raise ValueError("Chunk must be a dictionary.")

    missing_fields = [
        field for field in required_fields if field not in chunk
    ]

    if missing_fields:
        raise ValueError(
            f"Chunk is missing required fields: {', '.join(missing_fields)}"
        )

    text = chunk["text"]
    source = chunk["source"]

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Chunk text must be a non-empty string.")

    if not isinstance(source, str) or not source.strip():
        raise ValueError("Chunk source must be a non-empty string.")

    try:
        page = int(chunk["page"])
        chunk_number = int(chunk["chunk_id"])
    except (TypeError, ValueError):
        raise ValueError("Page and chunk ID must be integers.") from None

    if page < 1 or chunk_number < 0:
        raise ValueError(
            "Page must be positive and chunk ID cannot be negative."
        )

    content_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:16]

    return f"{source}:{page}:{chunk_number}:{content_hash}"


def _validate_embedding(embedding, label):
    """Convert an embedding to a list and validate its values."""
    if not hasattr(embedding, "tolist"):
        raise ValueError(f"{label} must support tolist().")

    values = embedding.tolist()

    if not isinstance(values, list) or not values:
        raise ValueError(
            f"{label} must be a non-empty one-dimensional vector."
        )

    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values
    ):
        raise ValueError(
            f"{label} must contain only finite numeric values."
        )

    return values


def store_chunks(collection, chunks, embeddings):
    """Store chunk text, metadata, and embeddings."""
    if len(chunks) != len(embeddings):
        raise ValueError("Each chunk must have one embedding.")

    if not chunks:
        return 0

    chunk_ids = [create_chunk_id(chunk) for chunk in chunks]

    embedding_values = [
        _validate_embedding(embedding, f"Embedding at index {index}")
        for index, embedding in enumerate(embeddings)
    ]

    expected_dimension = len(embedding_values[0])

    if any(
        len(embedding) != expected_dimension
        for embedding in embedding_values
    ):
        raise ValueError("All embeddings must have the same dimension.")

    collection.upsert(
        ids=chunk_ids,
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {
                "source": chunk["source"],
                "page": int(chunk["page"]),
                "chunk_id": int(chunk["chunk_id"]),
            }
            for chunk in chunks
        ],
        embeddings=embedding_values,
    )

    return len(chunks)


def get_missing_chunks(collection, chunks):
    """Return chunks that are not already stored."""
    if not chunks:
        return []

    ids = [create_chunk_id(chunk) for chunk in chunks]
    existing = collection.get(ids=ids, include=[])

    existing_ids = set(existing["ids"])

    return [
        chunk
        for chunk, chunk_id in zip(chunks, ids)
        if chunk_id not in existing_ids
    ]


def search_vector_store(
    collection,
    query_embedding,
    top_k=3,
    min_similarity=None,
):
    """Retrieve similar chunks and optionally filter weak matches."""
    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k <= 0
    ):
        raise ValueError("top_k must be a positive integer.")

    if min_similarity is not None:
        if (
            isinstance(min_similarity, bool)
            or not isinstance(min_similarity, (int, float))
            or not math.isfinite(min_similarity)
            or not -1.0 <= min_similarity <= 1.0
        ):
            raise ValueError(
                "min_similarity must be a finite number between -1 and 1."
            )

    query_values = _validate_embedding(
        query_embedding,
        "Query embedding",
    )

    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    results = collection.query(
        query_embeddings=[query_values],
        n_results=min(top_k, total_chunks),
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []

    if not results["ids"] or not results["ids"][0]:
        return retrieved

    for i, chunk_id in enumerate(results["ids"][0]):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        similarity_score = 1.0 - distance

        if (
            min_similarity is not None
            and similarity_score < min_similarity
        ):
            continue

        retrieved.append(
            {
                "chunk": {
                    "text": results["documents"][0][i],
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "chunk_id": metadata["chunk_id"],
                },
                "score": similarity_score,
                "id": chunk_id,
            }
        )

    return retrieved
