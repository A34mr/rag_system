# RAG System

A minimal, modern Retrieval-Augmented Generation (RAG) pipeline built with
LangChain 1.x-style modular packages — no `langchain-community`, no legacy
`langchain.chains`.

Documents are loaded from `data/`, chunked, embedded locally with
`sentence-transformers`, stored in a persistent Chroma vector database, and
queried through an LLM chain served by Ollama.

## Features

- **Multi-format ingestion** — loads `.pdf`, `.txt`, and `.md` files recursively from `data/`
- **Local embeddings** — `sentence-transformers/all-MiniLM-L6-v2` via `langchain_huggingface` (CPU by default; switchable to GPU)
- **Persistent vector store** — ChromaDB saved to `chroma_db/`
- **LCEL QA chain** — retriever → prompt → `llama3.1` (Ollama) → string output
- **Source attribution** — every answer lists the retrieved documents (with page numbers for PDFs)

## Project Structure

```
rag_system/
├── rag.py          # Core: loading, splitting, embeddings, vector store, QA chain
├── ingest.py       # Build/rebuild the vector store from data/
├── query.py        # Interactive question-answering REPL
├── data/           # Your source documents (.pdf / .txt / .md)
└── chroma_db/      # Persisted Chroma vector store (generated)
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with the `llama3.1` model pulled:

  ```bash
  ollama pull llama3.1
  ```

Python dependencies:

```bash
pip install langchain-core langchain-text-splitters langchain-chroma \
            langchain-huggingface langchain-ollama pypdf
```

## Configuration

All tunables live at the top of `rag.py`:

| Setting       | Default                              | Description                          |
|---------------|--------------------------------------|--------------------------------------|
| `DATA_DIR`    | `data`                               | Directory containing source documents |
| `DB_DIR`      | `chroma_db`                          | Chroma persistence directory         |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model                 |
| `LLM_MODEL`   | `llama3.1`                           | Ollama model name                    |
| `CHUNK_SIZE`  | `800`                                | Chunk size (characters)              |
| `CHUNK_OVER`  | `150`                                | Chunk overlap (characters)           |
| `TOP_K`       | `4`                                  | Number of chunks retrieved per query |

To use a GPU for embeddings, change `model_kwargs={"device": "cpu"}` to `"cuda"`
in `get_embeddings()`.

## Usage

### 1. Ingest documents

Drop your files into `data/`, then build the index:

```bash
python ingest.py
```

This prints the number of pages/docs loaded and chunks created, and saves the
vector store to `chroma_db/`. Re-run it any time you add or change documents.

### 2. Ask questions

```bash
python query.py
```

You'll get an interactive prompt:

```
RAG ready. Type your question (or 'exit').

> What is X?

=== Answer ===
...

=== Sources ===
- data/sample.txt
- data/doc.pdf (page 3)
```

Type `exit`, `quit`, or an empty line to leave.

### Alternative

Running `python rag.py` directly also rebuilds the index (same as `ingest.py`).

## How It Works

1. **Load** — `load_documents()` walks `data/`, extracting text per PDF page
   (with page metadata) or reading whole text/markdown files.
2. **Split** — `RecursiveCharacterTextSplitter` breaks documents into 800-char
   chunks with 150-char overlap.
3. **Embed & store** — normalized MiniLM embeddings are persisted into Chroma.
4. **Retrieve & generate** — the top-4 relevant chunks are formatted into a
   context-augmented prompt; `llama3.1` answers using *only* that context and
   replies "I don't know based on the provided documents." when the answer
   isn't present.

## Notes

- The first run downloads the embedding model from Hugging Face.
- If `data/` contains no supported files, ingestion exits with an error.
- Unreadable files are skipped with a `[warn]` message rather than crashing.
