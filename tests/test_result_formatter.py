
import pytest

from src.result_formatter import format_search_results


@pytest.fixture
def sample_results():
    return [
        {
            "chunk": {
                "text": "PERCLOS measures prolonged eye closure.",
                "source": "perclos_detection.pdf",
                "page": 4,
                "chunk_id": 12,
            },
            "score": 0.875,
        }
    ]


def test_formatter_includes_source_traceability(sample_results):
    output = format_search_results(sample_results)

    assert "Source: perclos_detection.pdf" in output
    assert "Page: 4" in output
    assert "Chunk ID: 12" in output


def test_formatter_displays_similarity_score(sample_results):
    output = format_search_results(sample_results)

    assert "Similarity Score: 0.8750" in output
    assert "Similarity Percentage: 87.50%" in output


def test_formatter_includes_chunk_text(sample_results):
    output = format_search_results(sample_results)

    assert "PERCLOS measures prolonged eye closure." in output


def test_formatter_handles_empty_results():
    assert (
        format_search_results([])
        == "No relevant results found."
    )


@pytest.mark.parametrize(
    "invalid_result",
    [
        None,
        {"chunk": "invalid", "score": 0.8},
        {"chunk": {}, "score": None},
        {"chunk": {}, "score": "invalid"},
        {"chunk": {}, "score": float("nan")},
        {"chunk": {}, "score": float("inf")},
        {"chunk": {}, "score": -0.1},
        {"chunk": {}, "score": 1.1},
    ],
)
def test_formatter_skips_invalid_results(invalid_result):
    output = format_search_results([invalid_result])

    assert output == "No valid search results to display."


def test_formatter_keeps_valid_results_when_another_is_invalid(
    sample_results,
):
    results = [
        {"chunk": {}, "score": 1.5},
        *sample_results,
    ]

    output = format_search_results(results)

    assert "Source: perclos_detection.pdf" in output
    assert "Page: 4" in output
    assert "Chunk ID: 12" in output
    assert "Similarity Score: 0.8750" in output
