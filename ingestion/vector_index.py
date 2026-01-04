"""
Vector index utilities (Chroma via LangChain).

Design goals for MVP:
- Use HuggingFace embeddings by default (SentenceTransformers).
- If embeddings cannot be loaded (e.g., offline without cached models), fall back to a
  deterministic hash embedding to keep the pipeline runnable.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, List, Tuple

try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    from langchain.schema import Document  # type: ignore

try:
    from langchain_core.embeddings import Embeddings
except Exception:  # pragma: no cover
    from langchain.embeddings.base import Embeddings  # type: ignore

from ingestion.utils import ensure_dir


class DeterministicHashEmbeddings(Embeddings):
    """A tiny, fully offline embedding implementation (not semantic, but stable)."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        vec: List[float] = []
        counter = 0
        base = text or ""
        while len(vec) < self.dim:
            digest = hashlib.sha256(f"{base}|{counter}".encode("utf-8")).digest()
            for b in digest:
                # Map byte [0,255] -> [-1, 1]
                vec.append((b / 255.0) * 2.0 - 1.0)
                if len(vec) >= self.dim:
                    break
            counter += 1
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def get_embeddings(model_name: str, *, force_hash: bool = False) -> Embeddings:
    """Return an Embeddings implementation based on settings and availability."""

    if force_hash:
        return DeterministicHashEmbeddings()

    try:
        # Prefer the newest LangChain integration.
        from langchain_huggingface import HuggingFaceEmbeddings

        # Default to fully offline behavior unless the user explicitly allows downloads.
        allow_download = os.environ.get("ALLOW_MODEL_DOWNLOAD", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if not allow_download:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        emb = HuggingFaceEmbeddings(model_name=model_name)
        # Warm-up call to ensure the model can be loaded. If this fails (e.g., truly
        # offline without cached weights), fall back to a deterministic embedding.
        _ = emb.embed_query("ping")
        return emb
    except Exception:
        try:
            # Legacy fallback location.
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

            emb = HuggingFaceEmbeddings(model_name=model_name)
            _ = emb.embed_query("ping")
            return emb
        except Exception:
            return DeterministicHashEmbeddings()


def _import_chroma_class():
    try:
        from langchain_chroma import Chroma

        return Chroma
    except Exception:  # pragma: no cover
        try:
            from langchain_community.vectorstores import Chroma  # type: ignore

            return Chroma
        except Exception:  # pragma: no cover
            from langchain.vectorstores import Chroma  # type: ignore

            return Chroma


def build_vector_store(
    *, chroma_dir: Path, docs: List[Document], embeddings: Embeddings, collection_name: str = "graphrag_mvp"
):
    """Create a persistent Chroma vector store from documents."""

    Chroma = _import_chroma_class()

    ensure_dir(chroma_dir)
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(chroma_dir),
        collection_name=collection_name,
    )
    # Older versions require explicit persist(). Newer versions persist automatically.
    try:  # pragma: no cover
        vs.persist()
    except Exception:
        pass
    return vs


def load_vector_store(
    *, chroma_dir: Path, embeddings: Embeddings, collection_name: str = "graphrag_mvp"
):
    """Load a persistent Chroma vector store."""

    Chroma = _import_chroma_class()

    return Chroma(
        persist_directory=str(chroma_dir),
        collection_name=collection_name,
        embedding_function=embeddings,
    )


def similarity_search_with_score(
    vectorstore: Any, query: str, *, k: int
) -> List[Tuple[Document, float]]:
    """
    Return (Document, distance/score) pairs.

    Chroma typically returns distance where lower is more similar.
    """

    fn = getattr(vectorstore, "similarity_search_with_score", None)
    if callable(fn):
        return list(fn(query, k=k))

    # Fallback: if only similarity_search exists, emulate a score-less API.
    docs = vectorstore.similarity_search(query, k=k)
    return [(d, 0.0) for d in docs]


