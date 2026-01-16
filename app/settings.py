"""
Application settings loaded from environment variables.

This MVP intentionally keeps settings explicit and lightweight:
- Values are read from environment variables (optionally via a local .env file).
- Paths are stored as ``pathlib.Path``.

The goal is a stable foundation for a GraphRAG-style project that can run without
external LLM credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ingestion.utils import load_dotenv_if_present


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for both ingestion and serving."""

    index_dir: Path
    chroma_dir: Path
    graph_path: Path

    llm_provider: str
    openai_api_key: Optional[str]

    graph_llm_provider: str
    graph_llm_model: str
    ollama_base_url: str
    graph_llm_max_docs: int
    graph_llm_temperature: float
    graph_llm_num_predict: int
    graph_llm_timeout_s: int

    embedding_model: str
    embedding_device: str
    embedding_batch_size: int
    hash_embedding_dim: int
    top_k: int
    k_hop: int
    force_hash_embeddings: bool

    @property
    def chunks_path(self) -> Path:
        """Path to the persisted chunk store (JSONL)."""

        return self.index_dir / "chunks.jsonl"

    @property
    def manifest_path(self) -> Path:
        """Path to the build manifest JSON."""

        return self.index_dir / "manifest.json"

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables (and optional .env)."""

        load_dotenv_if_present()

        def _path(name: str, default: str) -> Path:
            return Path(os.environ.get(name, default))

        def _int(name: str, default: int) -> int:
            try:
                v = int(os.environ.get(name, str(default)))
                return v if v > 0 else default
            except Exception:
                return default

        def _float(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, str(default)))
            except Exception:
                return default

        index_dir = _path("INDEX_DIR", "data/index")
        chroma_dir = _path("CHROMA_DIR", str(index_dir / "chroma"))
        graph_path = _path("GRAPH_PATH", str(index_dir / "graph.json.gz"))

        llm_provider = os.environ.get("LLM_PROVIDER", "openai").strip()
        openai_api_key = (os.environ.get("OPENAI_API_KEY") or "").strip() or None

        graph_llm_provider = os.environ.get("GRAPH_LLM_PROVIDER", "none").strip()
        graph_llm_model = os.environ.get("GRAPH_LLM_MODEL", "mistral:7b").strip()
        ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").strip()
        graph_llm_max_docs = _int("GRAPH_LLM_MAX_DOCS", 25)
        graph_llm_temperature = _float("GRAPH_LLM_TEMPERATURE", 0.0)
        graph_llm_num_predict = _int("GRAPH_LLM_NUM_PREDICT", 256)
        graph_llm_timeout_s = _int("GRAPH_LLM_TIMEOUT_S", 120)

        embedding_model = os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip()
        embedding_device = (os.environ.get("EMBEDDING_DEVICE", "auto") or "auto").strip()
        embedding_batch_size = _int("EMBEDDING_BATCH_SIZE", 64)
        hash_embedding_dim = _int("HASH_EMBEDDING_DIM", 384)

        top_k = int(os.environ.get("TOP_K", "5"))
        k_hop = int(os.environ.get("K_HOP", "2"))

        force_hash_embeddings = (
            os.environ.get("FORCE_HASH_EMBEDDINGS", "0").strip().lower()
            in {"1", "true", "yes"}
        )

        return cls(
            index_dir=index_dir,
            chroma_dir=chroma_dir,
            graph_path=graph_path,
            llm_provider=llm_provider,
            openai_api_key=openai_api_key,
            graph_llm_provider=graph_llm_provider,
            graph_llm_model=graph_llm_model,
            ollama_base_url=ollama_base_url,
            graph_llm_max_docs=graph_llm_max_docs,
            graph_llm_temperature=graph_llm_temperature,
            graph_llm_num_predict=graph_llm_num_predict,
            graph_llm_timeout_s=graph_llm_timeout_s,
            embedding_model=embedding_model,
            embedding_device=embedding_device,
            embedding_batch_size=embedding_batch_size,
            hash_embedding_dim=hash_embedding_dim,
            top_k=top_k,
            k_hop=k_hop,
            force_hash_embeddings=force_hash_embeddings,
        )

    def index_ready(self) -> bool:
        """Return True if the persisted index artifacts are present."""

        return (
            self.chroma_dir.exists()
            and self.graph_path.exists()
            and self.chunks_path.exists()
            and self.manifest_path.exists()
        )


