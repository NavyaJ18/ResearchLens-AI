def precision_at_k(retrieved_ids, relevant_ids, k):
    """
    Calculates Precision@K for retrieved document chunks.

    retrieved_ids: IDs returned by the retriever, ordered by relevance.
    relevant_ids: IDs considered relevant to the query.
    k: Number of retrieved results to evaluate.
    """

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    if not retrieved_ids:
        return 0.0

    retrieved_at_k = retrieved_ids[:k]

    relevant_retrieved = sum(
        1 for chunk_id in retrieved_at_k
        if chunk_id in relevant_ids
    )

    return relevant_retrieved / len(retrieved_at_k)