
def validate_query(query):
    """Validate and normalize a research question."""

    if not isinstance(query, str):
        raise ValueError("Research question must be a string.")

    query = query.strip()

    if not query:
        raise ValueError("Research question cannot be empty.")

    return query
