
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


from src.evaluation import f1_at_k


def test_f1_at_k():
    result = f1_at_k(
        ["a", "b", "c"],
        ["a", "c", "d"],
        3
    )
    assert result == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "invalid_k",
    [0, -1, 1.5, True, "3", None],
)
@pytest.mark.parametrize(
    "metric",
    [precision_at_k, recall_at_k, f1_at_k],
)
def test_evaluation_metrics_reject_invalid_k(metric, invalid_k):
    with pytest.raises(ValueError, match="positive integer"):
        metric(["a", "b"], ["a"], invalid_k)


def test_precision_when_k_exceeds_retrieved_results():
    assert precision_at_k(["a", "b"], ["a"], 5) == pytest.approx(0.5)


def test_recall_when_k_exceeds_retrieved_results():
    assert recall_at_k(["a", "b"], ["a", "c"], 5) == pytest.approx(0.5)


def test_f1_is_zero_when_no_retrieved_chunks_are_relevant():
    assert f1_at_k(["a", "b"], ["c", "d"], 2) == 0.0


def test_f1_is_zero_when_retrieved_results_are_empty():
    assert f1_at_k([], ["a", "b"], 3) == 0.0
