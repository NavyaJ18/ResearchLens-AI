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

    # Convert the user's query into an embedding
    query_embedding = model.encode(query)

    # Calculate similarity scores
    scores = np.dot(
        embeddings,
        query_embedding
    )

    # Get indices of highest scoring chunks
    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []

    for index in top_indices:

        results.append({
            "chunk": chunks[index],
            "score": float(scores[index])
        })

    return results