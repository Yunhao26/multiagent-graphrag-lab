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
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple

try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    from langchain.schema import Document  # type: ignore

try:
    from langchain_core.embeddings import Embeddings
except Exception:  # pragma: no cover
    from langchain.embeddings.base import Embeddings  # type: ignore

from ingestion.utils import ensure_dir


_DIM_MISMATCH_RE = re.compile(r"expecting embedding with dimension of (\d+), got (\d+)")


def _int_env(name: str, default: int) -> int:
    try:
        v = int(os.environ.get(name, str(default)))
        return v if v > 0 else default
    except Exception:
        return default


def _hash_embedding_dim(requested: int | None) -> int:
    """Resolve the hash embedding dimension (defaults to 384 for MiniLM alignment)."""

    if requested is None:
        return _int_env("HASH_EMBEDDING_DIM", 384)
    try:
        v = int(requested)
        return v if v > 0 else _int_env("HASH_EMBEDDING_DIM", 384)
    except Exception:
        return _int_env("HASH_EMBEDDING_DIM", 384)


def _resolve_embedding_device(requested: str | None) -> str:
    """
    Resolve embedding device.

    - "auto" -> "cuda" if available, else "cpu"
    - Otherwise return the provided string ("cpu", "cuda", "cuda:0", ...)
    """

    raw = (requested or os.environ.get("EMBEDDING_DEVICE", "auto") or "auto").strip()
    if raw.lower() != "auto":
        return raw
    try:
        import torch

        return "cuda" if bool(getattr(torch, "cuda", None)) and torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_embedding_batch_size(requested: int | None) -> Optional[int]:
    raw = None if requested is None else str(requested)
    if raw is None:
        raw = (os.environ.get("EMBEDDING_BATCH_SIZE") or "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
        return v if v > 0 else None
    except Exception:
        return None


def _probe_embedding_dim(embeddings: Embeddings) -> Optional[int]:
    """Best-effort probe to infer embedding dimension."""

    try:
        vec = embeddings.embed_query("dimension_probe")
        return int(len(vec))
    except Exception:
        return None


def _doc_ids(docs: List[Document]) -> List[str]:
    """Derive stable ids for vector store entries (prefer chunk_id)."""

    ids: List[str] = []
    seen: set[str] = set()
    for i, d in enumerate(docs):
        meta = dict(getattr(d, "metadata", {}) or {})
        cid = str(meta.get("chunk_id") or meta.get("id") or f"doc{i}").strip()
        if not cid:
            cid = f"doc{i}"
        if cid in seen:
            cid = f"{cid}::{i}"
        seen.add(cid)
        ids.append(cid)
    return ids


def _raise_friendly_dim_mismatch(
    exc: Exception, *, chroma_dir: Path, embeddings: Embeddings, collection_name: str
) -> None:
    msg = str(exc)
    m = _DIM_MISMATCH_RE.search(msg)
    if not m:
        return

    expected = int(m.group(1))
    got = int(m.group(2))
    probed = _probe_embedding_dim(embeddings)

    hint = ""
    if probed is not None and probed != got:
        hint = f" (probed_dim={probed})"

    raise RuntimeError(
        "Chroma embedding dimension mismatch.\n"
        f"- expected: {expected}\n"
        f"- got:      {got}{hint}\n"
        f"- CHROMA_DIR: {chroma_dir}\n"
        f"- collection: {collection_name}\n\n"
        "This usually happens when the index was built with a different embedding model/dimension, "
        "or when switching FORCE_HASH_EMBEDDINGS / HASH_EMBEDDING_DIM.\n\n"
        "Fix options:\n"
        "1) Delete the persisted CHROMA_DIR and rebuild the index.\n"
        "2) Keep embedding settings consistent across build and serve: EMBEDDING_MODEL / "
        "FORCE_HASH_EMBEDDINGS / HASH_EMBEDDING_DIM.\n"
        "Tip: If you rely on the hash fallback with the default MiniLM model, set HASH_EMBEDDING_DIM=384.\n"
    ) from exc


class DeterministicHashEmbeddings(Embeddings):
    """A tiny, fully offline embedding implementation (not semantic, but stable)."""

    def __init__(self, dim: int | None = None):
        self.dim = _hash_embedding_dim(dim)

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


def get_embeddings(
    model_name: str,
    *,
    force_hash: bool = False,
    hash_dim: int | None = None,
    device: str | None = None,
    batch_size: int | None = None,
) -> Embeddings:
    """Return an Embeddings implementation based on settings and availability."""

    if force_hash:
        return DeterministicHashEmbeddings(dim=_hash_embedding_dim(hash_dim))

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
        else:
            # Honor the explicit allow flag.
            os.environ["HF_HUB_OFFLINE"] = "0"
            os.environ["TRANSFORMERS_OFFLINE"] = "0"

        resolved_device = _resolve_embedding_device(device)
        resolved_batch_size = _resolve_embedding_batch_size(batch_size)
        model_kwargs = {"device": resolved_device} if resolved_device else {}
        encode_kwargs = {"batch_size": resolved_batch_size} if resolved_batch_size else {}

        kwargs: dict[str, Any] = {"model_name": model_name}
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs
        if encode_kwargs:
            kwargs["encode_kwargs"] = encode_kwargs

        emb = HuggingFaceEmbeddings(**kwargs)
        # Warm-up call to ensure the model can be loaded. If this fails (e.g., truly
        # offline without cached weights), fall back to a deterministic embedding.
        _ = emb.embed_query("ping")
        return emb
    except Exception:
        try:
            # Legacy fallback location.
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

            resolved_device = _resolve_embedding_device(device)
            resolved_batch_size = _resolve_embedding_batch_size(batch_size)
            model_kwargs = {"device": resolved_device} if resolved_device else {}
            encode_kwargs = {"batch_size": resolved_batch_size} if resolved_batch_size else {}

            kwargs2: dict[str, Any] = {"model_name": model_name}
            if model_kwargs:
                kwargs2["model_kwargs"] = model_kwargs
            if encode_kwargs:
                kwargs2["encode_kwargs"] = encode_kwargs

            emb = HuggingFaceEmbeddings(**kwargs2)
            _ = emb.embed_query("ping")
            return emb
        except Exception:
            return DeterministicHashEmbeddings(dim=_hash_embedding_dim(hash_dim))


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
    """
    Create a persistent Chroma vector store from documents.

    Notes:
    - Uses stable ids (chunk_id) to avoid duplication on repeated builds.
    - If embedding dimension changes between runs, Chroma will reject upserts; we surface a
      friendly error with recovery steps.
    """

    Chroma = _import_chroma_class()

    ensure_dir(chroma_dir)
    ids = _doc_ids(docs)

    try:
        try:
            vs = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                ids=ids,
                persist_directory=str(chroma_dir),
                collection_name=collection_name,
            )
        except TypeError:
            # Compatibility fallback: some versions don't accept ids in from_documents.
            try:
                vs = Chroma.from_texts(
                    texts=[d.page_content for d in docs],
                    embedding=embeddings,
                    metadatas=[dict(getattr(d, "metadata", {}) or {}) for d in docs],
                    ids=ids,
                    persist_directory=str(chroma_dir),
                    collection_name=collection_name,
                )
            except TypeError:
                vs = Chroma.from_documents(
                    documents=docs,
                    embedding=embeddings,
                    persist_directory=str(chroma_dir),
                    collection_name=collection_name,
                )
    except Exception as exc:
        _raise_friendly_dim_mismatch(
            exc, chroma_dir=chroma_dir, embeddings=embeddings, collection_name=collection_name
        )
        raise
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


