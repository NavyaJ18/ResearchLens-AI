
def format_search_results(results):
    """Format semantic search results for readable terminal output."""

    if not results:
        return "No relevant results found."

    output = []

    for rank, result in enumerate(results, start=1):
        chunk = result.get("chunk", {})
        score = result.get("score")

        if not isinstance(chunk, dict) or score is None:
            continue

        try:
            score = float(score)
        except (TypeError, ValueError):
            continue

        if not 0.0 <= score <= 1.0:
            continue

        source = chunk.get("source", "Unknown source")
        page = chunk.get("page", "Unknown")
        chunk_id = chunk.get("chunk_id", "Unknown")
        text = chunk.get("text", "")

        output.append(
            f"\n--- Result {rank} ---\n"
            f"Similarity Score: {score:.4f}\n"
            f"Similarity Percentage: {score * 100:.2f}%\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Text:\n{str(text).strip()}\n"
        )

    if not output:
        return "No valid search results to display."

    return "\n".join(output)
