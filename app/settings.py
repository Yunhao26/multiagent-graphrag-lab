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

    embedding_model: str
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

        index_dir = _path("INDEX_DIR", "data/index")
        chroma_dir = _path("CHROMA_DIR", str(index_dir / "chroma"))
        graph_path = _path("GRAPH_PATH", str(index_dir / "graph.json.gz"))

        llm_provider = os.environ.get("LLM_PROVIDER", "openai").strip()
        openai_api_key = (os.environ.get("OPENAI_API_KEY") or "").strip() or None

        embedding_model = os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip()

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
            embedding_model=embedding_model,
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


