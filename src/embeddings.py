
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


def load_embedding_model():
    """Load and return the configured sentence embedding model."""

    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    return model


def create_embeddings(model, chunks):
    """Convert document chunks into normalized vector embeddings."""

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    if not texts:
        raise ValueError("No text chunks available for embedding.")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return embeddings
