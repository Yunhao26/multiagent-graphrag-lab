"""Text splitters for chunking source Documents."""

from __future__ import annotations

from typing import List

try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    from langchain.schema import Document  # type: ignore


def get_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 150):
    """Create a reasonable default text splitter for MVP usage."""

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except Exception:  # pragma: no cover
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )


def split_documents(
    docs: List[Document], *, chunk_size: int = 1000, chunk_overlap: int = 150
) -> List[Document]:
    """
    Split source docs into smaller chunks with deterministic chunk ids.

    Chunk id rule:
    - ``chunk_id = f"{doc_id}::c{idx}"``

    The chunk metadata is guaranteed to include:
    - chunk_id, doc_id, source
    - section (optional)
    - page_or_row (optional)
    """

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks: List[Document] = []

    for doc in docs:
        doc_meta = dict(getattr(doc, "metadata", {}) or {})
        doc_id = str(doc_meta.get("doc_id") or doc_meta.get("source") or "doc")
        source = str(doc_meta.get("source") or "")

        pieces = splitter.split_documents([doc])
        for idx, ch in enumerate(pieces):
            meta = dict(getattr(ch, "metadata", {}) or {})
            meta["doc_id"] = str(meta.get("doc_id") or doc_id)
            meta["source"] = str(meta.get("source") or source)
            meta["chunk_id"] = f"{meta['doc_id']}::c{idx}"

            # Optional metadata fields
            if "section" in doc_meta and "section" not in meta:
                meta["section"] = doc_meta["section"]
            if "page_or_row" in doc_meta and "page_or_row" not in meta:
                meta["page_or_row"] = doc_meta["page_or_row"]
            if "row" in doc_meta and "page_or_row" not in meta:
                meta["page_or_row"] = doc_meta["row"]

            ch.metadata = meta
            all_chunks.append(ch)

    return all_chunks


