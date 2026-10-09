
import pytest

from src.evaluation import precision_at_k, recall_at_k


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], ["a", "c"], 3) == pytest.approx(2 / 3)


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["a", "c", "d"], 3) == pytest.approx(2 / 3)


def test_precision_with_no_results():
    assert precision_at_k([], ["a"], 3) == 0.0


def test_recall_with_no_relevant_chunks():
    assert recall_at_k(["a", "b"], [], 2) == 0.0


def test_precision_rejects_nonpositive_k():
    with pytest.raises(ValueError):
        precision_at_k(["a"], ["a"], 0)


def test_recall_rejects_nonpositive_k():
    with pytest.raises(ValueError):
        recall_at_k(["a"], ["a"], 0)


def test_recall_when_no_chunks_are_retrieved():
    assert recall_at_k([], ["a", "b"], 3) == 0.0
