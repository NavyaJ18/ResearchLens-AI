def format_search_results(results):
    """
    Formats semantic search results for readable terminal output.
    """

    if not results:
        return "No relevant results found."

    output = []

    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]
        score = result["score"]

        output.append(
            f"\n--- Result {rank} ---\n"
            f"Similarity Score: {score:.4f}\n"
            f"Source: {chunk['source']}\n"
            f"Page: {chunk['page']}\n"
            f"Chunk ID: {chunk['chunk_id']}\n"
            f"Text:\n{chunk['text'].strip()}\n"
        )

    return "\n".join(output)