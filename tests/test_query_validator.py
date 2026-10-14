
import pytest

from src.query_validator import validate_query


def test_validate_query_preserves_valid_question():
    query = "What does PERCLOS measure?"

    assert validate_query(query) == query


def test_validate_query_strips_surrounding_whitespace():
    assert validate_query("  What is PERCLOS?  ") == "What is PERCLOS?"


@pytest.mark.parametrize("query", ["", " ", "\t", "\n"])
def test_validate_query_rejects_empty_input(query):
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_query(query)


@pytest.mark.parametrize("query", [None, 123, ["PERCLOS"]])
def test_validate_query_rejects_non_string_input(query):
    with pytest.raises(ValueError, match="must be a string"):
        validate_query(query)
