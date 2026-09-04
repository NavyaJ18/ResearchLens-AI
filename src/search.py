import numpy as np


def search_similar_chunks(
    model,
    query,
    chunks,
    embeddings,
    top_k=3
):
    """
    Finds the most semantically similar chunks
    for a given query.
    """

    # Validate query
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    # Validate top_k
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    # Validate available data
    if len(chunks) == 0:
        raise ValueError("No document chunks available.")

    # Prevent requesting more results than available
    top_k = min(top_k, len(chunks))

    # Create normalized query embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    # Similarity scores
    scores = np.dot(
        embeddings,
        query_embedding
    )

    # Get best matching chunk indices
    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []

    for index in top_indices:
        results.append({
            "chunk": chunks[index],
            "score": float(scores[index])
        })

    return results