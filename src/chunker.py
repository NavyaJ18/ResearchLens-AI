def chunk_pages(pages, chunk_size=1000, chunk_overlap=200):
    """
    Splits extracted PDF pages into smaller overlapping chunks.
    """

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "text": chunk_text,
                "page": page["page"],
                "source": page["source"]
            })

            start = end - chunk_overlap

    return chunks