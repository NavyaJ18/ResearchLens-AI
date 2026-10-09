
import hashlib
import chromadb


def get_or_create_collection(persist_directory, collection_name):
    """Open or create a persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(persist_directory))

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


def create_chunk_id(chunk):
    """Generate a stable ID for a document chunk."""
    text = chunk["text"]
    source = chunk["source"]
    page = int(chunk["page"])
    chunk_number = int(chunk["chunk_id"])

    content_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:16]

    return f"{source}:{page}:{chunk_number}:{content_hash}"


def store_chunks(collection, chunks, embeddings):
    """Store chunk text, metadata, and embeddings."""
    if len(chunks) != len(embeddings):
        raise ValueError("Each chunk must have one embedding.")

    if not chunks:
        return 0

    collection.upsert(
        ids=[create_chunk_id(chunk) for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {
                "source": chunk["source"],
                "page": int(chunk["page"]),
                "chunk_id": int(chunk["chunk_id"]),
            }
            for chunk in chunks
        ],
        embeddings=[embedding.tolist() for embedding in embeddings],
    )

    return len(chunks)


def get_missing_chunks(collection, chunks):
    """Return chunks that are not already stored."""
    if not chunks:
        return []

    ids = [create_chunk_id(chunk) for chunk in chunks]
    existing = collection.get(ids=ids, include=[])

    existing_ids = set(existing["ids"])

    return [
        chunk
        for chunk, chunk_id in zip(chunks, ids)
        if chunk_id not in existing_ids
    ]


def search_vector_store(collection, query_embedding, top_k=3):
    """Retrieve the most similar chunks from ChromaDB."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []

    if not results["ids"] or not results["ids"][0]:
        return retrieved

    for i, chunk_id in enumerate(results["ids"][0]):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        retrieved.append(
            {
                "chunk": {
                    "text": results["documents"][0][i],
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "chunk_id": metadata["chunk_id"],
                },
                "score": 1.0 - distance,
                "id": chunk_id,
            }
        )

    return retrieved
