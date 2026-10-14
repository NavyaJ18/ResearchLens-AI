# Retrieval Evaluation Guide

## Purpose

ResearchLens-AI evaluates semantic retrieval by comparing retrieved chunk IDs against a set of known relevant chunk IDs. The evaluation module currently provides Precision@K, Recall@K, and F1@K.

## Metrics

### Precision@K

Measures the proportion of retrieved results within the first K positions that are relevant.

\[
Precision@K = \frac{\text{Relevant retrieved chunks}}{\text{Retrieved chunks within K}}
\]

### Recall@K

Measures the proportion of all known relevant chunks that were retrieved within the first K positions.

\[
Recall@K = \frac{\text{Relevant retrieved chunks}}{\text{Total relevant chunks}}
\]

### F1@K

Combines Precision@K and Recall@K using their harmonic mean.

\[
F1@K = \frac{2 \times Precision@K \times Recall@K}{Precision@K + Recall@K}
\]

When both precision and recall are zero, the implementation returns zero.

## Running the Tests

Activate the project's virtual environment and run:

```powershell
python -m pytest -q
```

To run only the evaluation tests:

```powershell
python -m pytest tests/test_evaluation.py -v
```

To check patch formatting:

```powershell
git diff --check
```

## Interpreting Results

- Higher Precision@K indicates fewer irrelevant chunks among the retrieved results.
- Higher Recall@K indicates that more of the known relevant chunks were retrieved.
- F1@K balances precision and recall into one score.
- Results depend on the relevance labels, query set, and chosen K.

## Limitations

Evaluation requires known relevant chunk IDs for each query. Metrics alone do not establish that generated answers are factually correct. Reliable comparisons require a consistent, representative query set and relevance labels.

The current test suite validates metric calculations and retrieval behavior. It does not, by itself, establish benchmark performance on a large labelled research-question dataset.