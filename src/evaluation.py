
import math


def _validate_k(k):
    """Ensure k is a positive integer."""
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer.")


def precision_at_k(retrieved_ids, relevant_ids, k):
    """Calculate Precision@K for retrieved document chunks."""
    _validate_k(k)

    if not retrieved_ids:
        return 0.0

    retrieved_at_k = retrieved_ids[:k]

    relevant_retrieved = sum(
        1 for chunk_id in retrieved_at_k
        if chunk_id in relevant_ids
    )

    return relevant_retrieved / len(retrieved_at_k)


def recall_at_k(retrieved_ids, relevant_ids, k):
    """Calculate Recall@K for retrieved document chunks."""
    _validate_k(k)

    if not relevant_ids:
        return 0.0

    retrieved_at_k = retrieved_ids[:k]

    relevant_retrieved = sum(
        1 for chunk_id in retrieved_at_k
        if chunk_id in relevant_ids
    )

    return relevant_retrieved / len(relevant_ids)


def f1_at_k(retrieved_ids, relevant_ids, k):
    """Calculate F1@K using Precision@K and Recall@K."""
    precision = precision_at_k(retrieved_ids, relevant_ids, k)
    recall = recall_at_k(retrieved_ids, relevant_ids, k)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)
