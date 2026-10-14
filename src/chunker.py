
def chunk_pages(pages, chunk_size=1000, chunk_overlap=200):
    """Split extracted PDF pages into smaller overlapping chunks."""

    if (
        not isinstance(chunk_size, int)
        or isinstance(chunk_size, bool)
        or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer.")

    if (
        not isinstance(chunk_overlap, int)
        or isinstance(chunk_overlap, bool)
        or chunk_overlap < 0
    ):
        raise ValueError("chunk_overlap must be a non-negative integer.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    if not isinstance(pages, (list, tuple)):
        raise ValueError("pages must be a list or tuple.")

    chunks = []

    for page_index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ValueError(
                f"Page at index {page_index} must be a dictionary."
            )

        required_fields = ("text", "page", "source")
        missing_fields = [
            field for field in required_fields if field not in page
        ]

        if missing_fields:
            raise ValueError(
                f"Page at index {page_index} is missing fields: "
                f"{', '.join(missing_fields)}"
            )

        text = page["text"]

        if not isinstance(text, str):
            raise ValueError(
                f"Page text at index {page_index} must be a string."
            )

        source = page["source"]
        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                f"Page source at index {page_index} must be non-empty."
            )

        page_number = page["page"]
        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number < 1
        ):
            raise ValueError(
                f"Page number at index {page_index} must be a "
                "positive integer."
            )

        text = text.strip()

        if not text:
            continue

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            chunks.append(
                {
                    "text": text[start:end],
                    "page": page_number,
                    "source": source,
                    "chunk_id": len(chunks),
                }
            )

            if end == len(text):
                break

            start = end - chunk_overlap

    return chunks
