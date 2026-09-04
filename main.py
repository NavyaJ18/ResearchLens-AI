from src.loader import load_pdf
from src.chunker import chunk_pages
from src.embeddings import (
    load_embedding_model,
    create_embeddings
)

pdf_path = "data/paper/perclos_detection.pdf"


# Step 1: Load PDF
pages = load_pdf(pdf_path)

print(f"\nTotal pages: {len(pages)}")


# Step 2: Chunk the document
chunks = chunk_pages(
    pages,
    chunk_size=1000,
    chunk_overlap=200
)

print(f"Total chunks created: {len(chunks)}")


# Step 3: Calculate chunk statistics
chunk_lengths = [len(chunk["text"]) for chunk in chunks]

average_length = sum(chunk_lengths) / len(chunk_lengths)
smallest_chunk = min(chunk_lengths)
largest_chunk = max(chunk_lengths)

# Step 4: Load embedding model
print("\n--- LOADING EMBEDDING MODEL ---\n")

model = load_embedding_model()


# Step 5: Create embeddings
print("\n--- CREATING EMBEDDINGS ---\n")

embeddings = create_embeddings(
    model,
    chunks
)


# Step 6: Display embedding information
print("\n--- EMBEDDING INFORMATION ---\n")

print("Number of embeddings:", len(embeddings))
print("Embedding dimension:", len(embeddings[0]))


print("\n--- CHUNK STATISTICS ---\n")

print(f"Average chunk length: {average_length:.2f} characters")
print(f"Smallest chunk: {smallest_chunk} characters")
print(f"Largest chunk: {largest_chunk} characters")


# Step 4: Display first chunk
print("\n--- FIRST CHUNK ---\n")

print(chunks[0]["text"])


print("\n--- CHUNK METADATA ---")

print("Chunk ID:", chunks[0]["chunk_id"])
print("Page:", chunks[0]["page"])
print("Source:", chunks[0]["source"])