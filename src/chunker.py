def chunk_pages(pages, chunk_size=1000, chunk_overlap=200):
    """
    Splits extracted PDF pages into smaller overlapping chunks.
    """

    # Validation
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    chunks = []

    for page in pages:
        text = page["text"].strip()

        # Skip empty pages
        if not text:
            continue

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            chunk_text = text[start:end]

            chunks.append({
                "text": chunk_text,
                "page": page["page"],
                "source": page["source"],
                "chunk_id": len(chunks)
            })

            # Stop when we reach the end
            if end == len(text):
                break

            start = end - chunk_overlap

    return chunks