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
from ingestion.graph_index import (
    build_community_reports,
    build_graph_from_chunks,
    maybe_enhance_graph_with_llm,
    save_graph,
)
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


def _build_entity_index(chunks: List[Document]) -> Dict[str, Any]:
    """
    Build a lightweight entity index for runtime entity linking.

    MVP focus:
    - Entities are course codes from CSV rows.
    - Aliases include course code and course name.
    - Evidence chunk ids point to the originating course row chunks.
    """

    by_code: Dict[str, Dict[str, Any]] = {}
    for ch in chunks:
        meta = dict(getattr(ch, "metadata", {}) or {})
        if str(meta.get("type") or "") != "course_row":
            continue

        code = str(meta.get("course_code") or "").strip().upper()
        if not code:
            continue
        name = str(meta.get("course_name") or "").strip()
        program = str(meta.get("program") or "").strip()
        year = meta.get("year")
        chunk_id = str(meta.get("chunk_id") or "").strip()

        rec = by_code.get(code) or {
            "id": code,
            "type": "COURSE",
            "name": name,
            "program": program,
            "year": year,
            "aliases": [],
            "evidence_chunk_ids": [],
        }

        aliases = list(rec.get("aliases", []) or [])
        for a in [code, name]:
            a = str(a or "").strip()
            if a and a not in aliases:
                aliases.append(a)
        rec["aliases"] = aliases

        ev = list(rec.get("evidence_chunk_ids", []) or [])
        if chunk_id and chunk_id not in ev:
            ev.append(chunk_id)
        rec["evidence_chunk_ids"] = ev

        by_code[code] = rec

    entities = list(by_code.values())
    entities.sort(key=lambda x: str(x.get("id") or ""))
    return {"created_at": now_iso_utc(), "entities": entities}


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
    entities_path = out_dir / "entities.json"
    community_reports_path = out_dir / "community_reports.json"

    selected_files = scan_source_files(data_dir)
    selected_sources = [str(p.relative_to(data_dir).as_posix()) for p in selected_files]

    docs = load_sources(data_dir, selected_sources=selected_files)
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Persist a small entity index for better runtime entity linking.
    write_json(_build_entity_index(chunks), entities_path)

    # Persist chunks for inspection/debugging.
    write_jsonl(_chunk_records(chunks), chunks_path)

    embeddings = get_embeddings(
        model_name=settings.embedding_model,
        force_hash=settings.force_hash_embeddings,
        hash_dim=settings.hash_embedding_dim,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    _ = build_vector_store(chroma_dir=chroma_dir, docs=chunks, embeddings=embeddings)

    # Build graph (rule-based safety net), with chunk evidence ids.
    g = build_graph_from_chunks(chunks)

    graph_llm = None
    if settings.graph_llm_provider.strip().lower() == "ollama":
        try:
            from langchain_ollama import ChatOllama  # type: ignore
        except Exception:  # pragma: no cover
            from langchain_community.chat_models.ollama import ChatOllama  # type: ignore

        graph_llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.graph_llm_model,
            temperature=float(settings.graph_llm_temperature),
            num_predict=int(settings.graph_llm_num_predict),
            timeout=int(settings.graph_llm_timeout_s),
            format="json",
        )

    g = maybe_enhance_graph_with_llm(
        graph=g,
        documents=chunks,
        llm=graph_llm,
        max_docs=int(settings.graph_llm_max_docs),
    )
    save_graph(g, graph_path)

    # Build community reports ("global" graph summaries) for GraphRAG-style retrieval.
    community_reports = build_community_reports(g)
    community_reports["created_at"] = now_iso_utc()
    write_json(community_reports, community_reports_path)

    manifest: Dict[str, Any] = {
        "created_at": now_iso_utc(),
        "data_dir": str(data_dir.as_posix()),
        "out_dir": str(out_dir.as_posix()),
        "selected_sources": selected_sources,
        "config": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_model": settings.embedding_model,
            "embedding_device": settings.embedding_device,
            "embedding_batch_size": settings.embedding_batch_size,
            "force_hash_embeddings": settings.force_hash_embeddings,
            "hash_embedding_dim": settings.hash_embedding_dim,
            "graph_llm": {
                "provider": settings.graph_llm_provider,
                "model": settings.graph_llm_model if settings.graph_llm_provider.strip().lower() != "none" else "",
                "max_docs": settings.graph_llm_max_docs,
            },
            "entity_index_path": "entities.json",
            "community_reports_path": "community_reports.json",
            "community_algo": str(community_reports.get("algo") or ""),
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
        "entities_path": str(entities_path),
        "community_reports_path": str(community_reports_path),
    }


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Build GraphRAG MVP index artifacts.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/sources",
        help="Directory containing source files (md/csv/pdf).",
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


