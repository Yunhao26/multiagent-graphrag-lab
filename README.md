# GraphRAG MVP (FastAPI + Streamlit + NetworkX + Chroma + LangChain + LangGraph)

This repository is a **minimal, offline-friendly GraphRAG skeleton**. It is designed to be stable and easy to extend:

- **Build/Serve separation**
  - **Build**: `ingestion/build_index.py` generates **chunks + Chroma vector index + `graph.json.gz`**
  - **Serve**: FastAPI exposes `/health` and `/qa`; Streamlit UI calls the FastAPI backend
- **Supported source formats**
  - Structured: **CSV** (course rows)
  - Unstructured: **Markdown** and **PDF (text-only)**. PDF pages are extracted via `pypdf` (no OCR).
- **Offline by default**
  - No `OPENAI_API_KEY` is required to **build the index** or to **answer questions**
  - If OpenAI is configured, answers can be generated with `ChatOpenAI` for improved fluency
- **GraphRAG**
  - Graph: **NetworkX** (course prerequisite graph)
  - Vector store: **Chroma** (via `langchain-chroma`)
  - Agent orchestration: **LangGraph**

## Project layout

```
app/          # FastAPI backend
agents/       # LangGraph agent graph/nodes + LLM wrapper
ingestion/    # index building: load -> split -> vectorize -> graph -> persist
ui/           # Streamlit UI + simple backend client
data/
  sources/    # mock source data
  index/      # build outputs (created by build_index)
notebooks/    # evaluation notebook(s)
tests/        # pytest smoke tests
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Optionally copy `env.example` to `.env` and adjust paths:

```bash
cp env.example .env
```

## Build the index (chunks + Chroma + graph)

```bash
python -m ingestion.build_index --data_dir data/sources --out_dir data/index
```

Supported inputs under `data/sources`:
- `*.md` (Markdown)
- `*.csv` (course rows)
- `*.pdf` (PDF, text-only extraction via `pypdf`)

Outputs (by default):

- `data/index/chunks.jsonl`
- `data/index/chroma/` (Chroma persistence directory)
- `data/index/graph.json.gz`
- `data/index/manifest.json`

## Start the backend (FastAPI)

```bash
uvicorn app.main:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Start the frontend (Streamlit)

```bash
streamlit run ui/streamlit_app.py
```

## Call `/qa` via curl

```bash
curl -X POST "http://localhost:8000/qa" -H "Content-Type: application/json" -d "{\"query\":\"What is the prerequisite chain to DL301?\",\"brief\":true,\"top_k\":5,\"k_hop\":2}"
```

## Modes

- **Offline Mode (default)**
  - No `OPENAI_API_KEY` required
  - Hybrid retrieval: Chroma similarity search + NetworkX graph paths + simple fusion
  - Deterministic offline summarization (no external LLM)
  - Embeddings: tries HuggingFace locally; falls back to deterministic hash embeddings if the model is not available

- **OpenAI Enhanced Mode (optional)**
  - Set `OPENAI_API_KEY` (and keep `LLM_PROVIDER=openai`) to enable `ChatOpenAI` answer generation
  - Answers are still grounded in evidence chunks and include chunk-id citations

- **Ollama Local Mode (optional)**
  - Set `LLM_PROVIDER=ollama` and `OLLAMA_MODEL=<local_ollama_model>` (e.g., `mistral:7b`) to generate answers with a local Ollama model
  - Requires a running Ollama server (default: `http://localhost:11434`)
  - If Ollama is unavailable, the system falls back to the deterministic offline summarizer

- **Optional Graph Transformer (Ollama, optional)**
  - Set `GRAPH_LLM_PROVIDER=ollama` and `GRAPH_LLM_MODEL=<local_ollama_model>` to enable `LLMGraphTransformer` during ingestion
  - This implementation is conservative: it only adds prerequisite edges between course codes and attaches chunk-id evidence
  - The rule-based graph builder remains the offline safety net

## Offline fallback vs OpenAI enhancement

- **Offline fallback (default)**
  - If `OPENAI_API_KEY` is missing, the system uses:
    - Retrieval: Chroma similarity search
    - Graph expansion: NetworkX k-hop traversal on prerequisite graph
    - Answer synthesis: deterministic, rule-based summarization (no external API)
  - Embeddings:
    - Tries HuggingFace embeddings first (configurable with `EMBEDDING_MODEL`)
    - If the model cannot be loaded, it falls back to a deterministic hash embedding (stable, fully offline)

- **With OpenAI (optional)**
  - Set `OPENAI_API_KEY` (and keep `LLM_PROVIDER=openai`) to enable `ChatOpenAI` for more natural answers.
  - Optional: `LLMGraphTransformer` can be enabled during ingestion via local Ollama to extract additional prerequisite edges from text. The rule-based graph builder remains the offline safety net.

## Evaluation

After building the index, open and run `notebooks/evaluation.ipynb`. It compares:
- `vector_only` vs `hybrid` retrieval
- latency, validity metrics, and a tiny Hit@k score

## Scoring checklist (deliverables covered)

| Deliverable | Status | Notes |
| --- | --- | --- |
| Build/Serve separation | Yes | `python -m ingestion.build_index ...` produces persisted artifacts; FastAPI serves QA from artifacts |
| Vector index (Chroma) | Yes | `ingestion/vector_index.py` + persisted `data/index/chroma/` |
| Graph index (NetworkX MultiDiGraph) | Yes | `ingestion/graph_index.py` + persisted `data/index/graph.json.gz` |
| Hybrid retrieval (vector + graph) | Yes | LangGraph nodes: retrieve → fuse → generate |
| Citations | Yes | `chunk_id/source/page_or_row/section` returned by `/qa` |
| Graph paths | Yes | ranked `graph_paths` with edge evidence chunk ids |
| Offline run (no OpenAI key) | Yes | offline summarizer + hash-embeddings fallback |
| OpenAI enhancement | Yes | optional `ChatOpenAI` when `OPENAI_API_KEY` is set |
| Demo UI (Streamlit) | Yes | form + QA + expanders, writes `data/index/users.csv` |
| Evaluation notebook | Yes | `notebooks/evaluation.ipynb` |
| Tests | Yes | `tests/test_smoke.py` builds index and runs offline QA |

## Troubleshooting

- **Port already in use**
  - Change the port: `uvicorn app.main:app --port 8010`
  - Or stop the process that is using the port.

- **Index not found / `/qa` returns 400**
  - Run: `python -m ingestion.build_index --data_dir data/sources --out_dir data/index`
  - Ensure `CHROMA_DIR` and `GRAPH_PATH` point to the generated artifacts.

- **Dependency install issues**
  - Upgrade pip: `python -m pip install --upgrade pip`
  - Recreate the venv if needed.
  - If `sentence-transformers` cannot download models, either pre-download models or set `FORCE_HASH_EMBEDDINGS=1`.

- **Embedding configuration**
  - `EMBEDDING_MODEL`: HuggingFace model name (default: `sentence-transformers/all-MiniLM-L6-v2`)
  - `EMBEDDING_DEVICE`: `auto` (default), `cpu`, `cuda`, `cuda:0`, ...
  - `EMBEDDING_BATCH_SIZE`: encoder batch size (default: `64`)
  - `FORCE_HASH_EMBEDDINGS=1`: force the deterministic offline fallback
  - `HASH_EMBEDDING_DIM`: hash embedding dimension (default: `384`)

- **Graph extraction (LLMGraphTransformer + Ollama)**
  - Disabled by default. Enable it during ingestion by setting:
    - `GRAPH_LLM_PROVIDER=ollama`
    - `GRAPH_LLM_MODEL=<local_ollama_model>` (e.g., `mistral:7b`)
    - `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
  - Optional knobs:
    - `GRAPH_LLM_MAX_DOCS` (default: `25`, set `0` to process all chunks)
    - `GRAPH_LLM_TEMPERATURE` / `GRAPH_LLM_NUM_PREDICT` / `GRAPH_LLM_TIMEOUT_S`

- **Chroma dimension mismatch**
  - If you see: `Collection expecting embedding with dimension of X, got Y`
    - Delete your persisted `CHROMA_DIR` (e.g., `data/index/chroma/`) and rebuild the index, or
    - Keep embedding settings consistent across build and serve (`EMBEDDING_MODEL` / `FORCE_HASH_EMBEDDINGS` / `HASH_EMBEDDING_DIM`).

- **GPU embeddings**
  - To use an NVIDIA GPU for embeddings, install a CUDA-enabled PyTorch build. A CPU-only PyTorch install will run embeddings on CPU even if a GPU is present.


