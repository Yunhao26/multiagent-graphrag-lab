"""
Index builder entry point.

Build outputs (default under ``data/index``):
- ``chunks.jsonl``: chunk texts + metadata
- ``chroma/``: Chroma persistent vector store
- ``graph.json.gz``: prerequisite graph (NetworkX) in gzipped JSON
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    from langchain.schema import Document  # type: ignore

from app.settings import Settings
from ingestion.graph_index import build_graph_from_chunks, maybe_enhance_graph_with_llm, save_graph
from ingestion.loaders import load_sources, scan_source_files
from ingestion.splitters import split_documents
from ingestion.utils import ensure_dir, now_iso_utc, write_json, write_jsonl
from ingestion.vector_index import build_vector_store, get_embeddings


def _chunk_records(chunks: List[Document]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for d in chunks:
        records.append(
            {
                "chunk_id": d.metadata.get("chunk_id"),
                "text": d.page_content,
                "metadata": dict(d.metadata),
            }
        )
    return records


def build_index(
    *,
    data_dir: Path,
    out_dir: Path,
    settings: Settings,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> Dict[str, Any]:
    """Build and persist GraphRAG artifacts."""

    ensure_dir(out_dir)
    chroma_dir = out_dir / "chroma"
    chunks_path = out_dir / "chunks.jsonl"
    graph_path = out_dir / "graph.json.gz"
    manifest_path = out_dir / "manifest.json"

    selected_files = scan_source_files(data_dir)
    selected_sources = [str(p.relative_to(data_dir).as_posix()) for p in selected_files]

    docs = load_sources(data_dir, selected_sources=selected_files)
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Persist chunks for inspection/debugging.
    write_jsonl(_chunk_records(chunks), chunks_path)

    embeddings = get_embeddings(
        model_name=settings.embedding_model, force_hash=settings.force_hash_embeddings
    )
    _ = build_vector_store(chroma_dir=chroma_dir, docs=chunks, embeddings=embeddings)

    # Build graph (rule-based safety net), with chunk evidence ids.
    g = build_graph_from_chunks(chunks)
    g = maybe_enhance_graph_with_llm(
        graph=g,
        documents_text="\n".join(d.page_content for d in docs),
        llm=None,
    )
    save_graph(g, graph_path)

    manifest: Dict[str, Any] = {
        "created_at": now_iso_utc(),
        "data_dir": str(data_dir.as_posix()),
        "out_dir": str(out_dir.as_posix()),
        "selected_sources": selected_sources,
        "config": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": settings.embedding_model,
            "force_hash_embeddings": settings.force_hash_embeddings,
            "collection_name": "graphrag_mvp",
        },
    }
    write_json(manifest, manifest_path)

    return {
        "chunks_path": str(chunks_path),
        "chroma_dir": str(chroma_dir),
        "graph_path": str(graph_path),
        "manifest_path": str(manifest_path),
        "selected_sources": selected_sources,
    }


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Build GraphRAG MVP index artifacts.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/sources",
        help="Directory containing source files (md/csv).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/index",
        help="Output directory for index artifacts.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Chunk size for text splitting.",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=150,
        help="Chunk overlap for text splitting.",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    result = build_index(
        data_dir=data_dir,
        out_dir=out_dir,
        settings=settings,
        chunk_size=int(args.chunk_size),
        chunk_overlap=int(args.chunk_overlap),
    )
    print("Index build complete.")
    print(f"- chunks:   {result['chunks_path']}")
    print(f"- chroma:   {result['chroma_dir']}")
    print(f"- graph:    {result['graph_path']}")
    print(f"- manifest: {result['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


