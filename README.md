# RAG with Retrieval Self-Assessment

A RAG pipeline that grades its retrieved chunks for relevance before answering, rewrites the query and re-retrieves when the context falls short, and refuses to answer rather than guessing.

It comes in two versions:

- **Web app** (`app.py`): Streamlit UI that shows each retrieval round, which chunks were judged relevant or rejected, query rewrites and refusals. Uses Groq for the LLM, so it can be hosted for free. Visitors can also upload their own documents.
- **Terminal version** (`rag_pipeline.py`): the fully local pipeline using Ollama. Documented below.

**Live demo:** _add your Streamlit URL here_

## Web app

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
mkdir -p .streamlit && echo 'GROQ_API_KEY = "gsk_..."' > .streamlit/secrets.toml
streamlit run app.py
```

The web app searches the 15 PDFs in `demo_docs/` by default. Free hosting steps are in [DEPLOY.md](DEPLOY.md).

Changes from the terminal version:

- Pluggable LLM provider: Groq (hosted) or Ollama (local), via `LLM_PROVIDER`.
- Chunk grading batched into one LLM call per round instead of one per chunk (~3 requests per question instead of ~15).
- Embeddings use Chroma's ONNX build of the same all-MiniLM-L6-v2 model, avoiding PyTorch.

---

# Terminal version (local, Ollama)

A local Retrieval-Augmented Generation (RAG) pipeline using ChromaDB for vector storage, sentence-transformers for embeddings, and Ollama for the LLM.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally

## Setup

### 1. Install Ollama and pull a model

```bash
# Install Ollama from https://ollama.com, then pull a model:
ollama pull llama3.2
```

Any of these models work — update `MODEL` in `rag_pipeline.py` to match:
`llama3.2` | `mistral` | `phi3` | `gemma2` | `qwen2.5` | `deepseek-r1`

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements-local.txt
```

### 4. Add your documents

Place the files you want to query into the `docs/` folder. Supported formats:

| Format | Extensions |
|---|---|
| PDF | `.pdf` |
| Word | `.docx` |
| Plain text | `.txt` |
| Excel | `.xlsx`, `.xls` |
| CSV | `.csv` |

```
docs/
├── report.pdf
├── notes.docx
├── data.xlsx
├── log.csv
└── readme.txt
```

> Note: Legacy `.doc` (Word 97–2003) is not supported. Convert to `.docx` first.

## Running the pipeline

```bash
python rag_pipeline.py
```

On first run, the pipeline will embed all documents in `docs/` and store them in `chroma_db/`. Subsequent runs skip unchanged files — only new or modified documents are re-indexed.

Once ready, type your question at the prompt:

```
You: What is the transformer architecture?
```

Type `quit`, `exit`, or `q` to stop.

## How document indexing works

A `doc_index.json` file is maintained in the project root. Each time the pipeline starts it:

1. Computes a SHA-256 hash of every supported file in `docs/`
2. Compares it against the stored hash in `doc_index.json`
3. **New file** — chunks and indexes it, saves the hash
4. **Modified file** — removes old chunks from the DB, re-indexes the new content
5. **Unchanged file** — skips entirely

To force a full re-index, delete `doc_index.json` and `chroma_db/`.

## Project structure

```
RAG/
├── app.py                  # Web UI (Streamlit)
├── rag_core.py             # Pipeline used by the web UI
├── rag_pipeline.py         # Terminal version (Ollama)
├── demo_docs/              # 15 PDFs the web app searches (committed)
├── docs/                   # Your own documents for the terminal version (not committed)
├── chroma_db/              # Vector database (auto-created, not committed)
├── doc_index.json          # File-hash index for change detection (auto-created)
├── requirements.txt        # Web app dependencies
├── requirements-local.txt  # Web app + Ollama + sentence-transformers
├── DEPLOY.md               # Free hosting guide
└── README.md
```

## Configuration

All tuneable settings are at the top of `rag_pipeline.py`:

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `llama3.2` | Ollama model to use |
| `DOCS_DIR` | `./docs` | Folder scanned for documents |
| `CHROMA_PATH` | `./chroma_db` | ChromaDB storage path |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `MAX_RETRIEVAL_ITERATIONS` | `3` | Query rewrite attempts before giving up |
| `MIN_RELEVANT_CHUNKS` | `1` | Minimum relevant chunks needed to answer |
