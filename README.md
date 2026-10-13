# ResearchLens-AI

A Retrieval-Augmented Generation (RAG) foundation for searching research papers using semantic similarity. The project extracts research text, converts it into vector embeddings, stores those embeddings in ChromaDB, and retrieves relevant passages for a user’s research question.

## Features

- **PDF ingestion:** Extract text from research papers using PyMuPDF.
- **Text chunking:** Split extracted text into manageable, overlapping chunks.
- **Semantic embeddings:** Generate normalized vector embeddings with Sentence Transformers.
- **Persistent vector storage:** Store document chunks and embeddings in ChromaDB.
- **Semantic search:** Retrieve passages based on query similarity rather than exact keyword matching.
- **Source tracking:** Display the source filename, page number, and chunk ID for retrieved passages.
- **Retrieval evaluation:** Evaluate search quality using metrics such as precision@k, recall@k, and F1@k.
- **Automated tests:** Test core retrieval, evaluation, and vector-store functionality.

## Architecture

```text
Research Paper (PDF)
        |
        v
   PDF Loader
        |
        v
   Text Chunker
        |
        v
  Embedding Model
        |
        v
 Persistent ChromaDB
        |
        v
 User Research Question
        |
        v
 Query Embedding
        |
        v
  Semantic Retrieval
        |
        v
 Ranked Passages + Sources
```

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| PDF extraction | PyMuPDF |
| Embeddings | Sentence Transformers |
| Embedding model | all-MiniLM-L6-v2 |
| Vector database | ChromaDB |
| Testing | pytest |

## Project Structure

```text
ResearchLens-AI/
├── data/
│   ├── paper/
│   └── chroma_db/
├── src/
│   ├── chunker.py
│   ├── config.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── loader.py
│   ├── result_formatter.py
│   ├── search.py
│   └── vector_store.py
├── tests/
├── main.py
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/NavyaJ18/ResearchLens-AI.git
cd ResearchLens-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Add a research paper

Place a PDF in `data/paper/`. Ensure the paper is available locally; research PDFs and generated vector-database files are not necessarily tracked by Git.

### 5. Run the application

```bash
python main.py
```

Follow the terminal prompts to search the indexed research content, if interactive querying is enabled in your current version.

### 6. Run the tests

```bash
python -m pytest -q
```

## Retrieval Evaluation

The project includes implementations of:

- **Precision@k:** The proportion of retrieved results that are relevant.
- **Recall@k:** The proportion of all relevant items retrieved within the top-k results.
- **F1@k:** The harmonic mean of precision@k and recall@k.

Automated tests help verify the metric calculations and vector-store behavior.

## Current Scope

ResearchLens-AI currently focuses on document ingestion, embedding generation, persistent vector storage, and semantic retrieval. It provides retrieved passages with source metadata to support verification against the original research paper.

Retrieval is not equivalent to verified question answering: similarity scores indicate semantic closeness, not factual correctness. The quality of results depends on the source documents, chunking strategy, embedding model, and retrieval configuration.

## Future Improvements

- Add a grounded answer-generation layer with citations.
- Support multiple research papers and metadata filters.
- Improve retrieval using hybrid search and reranking.
- Add retrieval-quality benchmarks on a labelled question set.
- Develop a web interface for document upload and search.

## Author

**Navya Jain**

GitHub: [NavyaJ18](https://github.com/NavyaJ18)
