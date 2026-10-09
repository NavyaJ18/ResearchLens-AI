
from src.loader import load_pdf
from src.chunker import chunk_pages
from src.embeddings import (
    load_embedding_model,
    create_embeddings,
)
from src.vector_store import (
    get_or_create_collection,
    get_missing_chunks,
    store_chunks,
    search_vector_store,
)
from src.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
    VECTOR_STORE_PATH,
    VECTOR_COLLECTION_NAME,
)

# Configuration
pdf_path = "data/paper/perclos_detection.pdf"


# Step 1: Load PDF
print("\n--- LOADING PDF ---\n")

pages = load_pdf(pdf_path)
print(f"Total pages: {len(pages)}")


# Step 2: Chunk the document
chunks = chunk_pages(
    pages,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

if not chunks:
    raise ValueError("No text chunks were extracted from the PDF.")

print(f"Total chunks created: {len(chunks)}")


# Step 3: Calculate chunk statistics
chunk_lengths = [len(chunk["text"]) for chunk in chunks]

average_length = sum(chunk_lengths) / len(chunk_lengths)
smallest_chunk = min(chunk_lengths)
largest_chunk = max(chunk_lengths)

print("\n--- CHUNK STATISTICS ---\n")
print(f"Average chunk length: {average_length:.2f} characters")
print(f"Smallest chunk: {smallest_chunk} characters")
print(f"Largest chunk: {largest_chunk} characters")


# Step 4: Open the persistent vector database
print("\n--- CONNECTING TO CHROMADB ---\n")

collection = get_or_create_collection(
    VECTOR_STORE_PATH,
    VECTOR_COLLECTION_NAME,
)

print(f"Existing stored chunks: {collection.count()}")


# Step 5: Load the embedding model
print("\n--- LOADING EMBEDDING MODEL ---\n")

model = load_embedding_model()


# Step 6: Embed and store only missing chunks
missing_chunks = get_missing_chunks(collection, chunks)

print(f"Chunks needing embeddings: {len(missing_chunks)}")

if missing_chunks:
    print("\n--- CREATING MISSING EMBEDDINGS ---\n")

    new_embeddings = create_embeddings(model, missing_chunks)

    stored_count = store_chunks(
        collection,
        missing_chunks,
        new_embeddings,
    )

    print(f"New chunks stored: {stored_count}")
else:
    print("All current document chunks are already stored.")
    print("Skipping document embedding generation.")

print(f"Total chunks in vector database: {collection.count()}")


# Step 7: Display embedding information
print("\n--- EMBEDDING INFORMATION ---\n")

print("Embedding dimension:",
      model.get_embedding_dimension())


# Step 8: Perform semantic search through ChromaDB
query = "What does PERCLOS measure?"

print("\n--- SEARCH QUERY ---\n")
print(query)

query_embedding = model.encode(
    query,
    normalize_embeddings=True,
)

results = search_vector_store(
    collection=collection,
    query_embedding=query_embedding,
    top_k=TOP_K_RESULTS,
)


# Step 9: Display search results
print("\n--- TOP RELEVANT CHUNKS ---\n")

for rank, result in enumerate(results, start=1):
    chunk = result["chunk"]

    print(f"\nResult #{rank}")
    print(f"Similarity Score: {result['score']:.4f}")
    print(f"Source: {chunk['source']}")
    print(f"Page: {chunk['page']}")

    print("\nText:")
    print(chunk["text"][:700])

    print("\n" + "=" * 60)


# Step 10: Display first chunk and its metadata
print("\n--- FIRST CHUNK ---\n")
print(chunks[0]["text"])

print("\n--- CHUNK METADATA ---")
print("Chunk ID:", chunks[0]["chunk_id"])
print("Page:", chunks[0]["page"])
print("Source:", chunks[0]["source"])
