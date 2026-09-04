from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model():
    """
    Loads and returns the sentence embedding model.
    """

    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    return model


def create_embeddings(model, chunks):
    """
    Converts document chunks into normalized vector embeddings.
    """

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    if not texts:
        raise ValueError("No text chunks available for embedding.")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return embeddings