"""Persistence-related types and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ingestion.utils import ensure_dir


@dataclass(frozen=True)
class IndexArtifacts:
    """Paths to persisted artifacts produced by the ingestion pipeline."""

    index_dir: Path
    chroma_dir: Path
    graph_path: Path
    chunks_path: Path

    def ensure_layout(self) -> None:
        """Ensure the required directory layout exists."""

        ensure_dir(self.index_dir)
        ensure_dir(self.chroma_dir)
        ensure_dir(self.graph_path.parent)
        ensure_dir(self.chunks_path.parent)

    def ready(self) -> bool:
        """Return True if artifacts appear to exist on disk."""

        return self.chroma_dir.exists() and self.graph_path.exists() and self.chunks_path.exists()


def default_artifacts(index_dir: Path) -> IndexArtifacts:
    """Default artifact paths under the index directory."""

    chroma_dir = index_dir / "chroma"
    graph_path = index_dir / "graph.json.gz"
    chunks_path = index_dir / "chunks.jsonl"
    return IndexArtifacts(
        index_dir=index_dir,
        chroma_dir=chroma_dir,
        graph_path=graph_path,
        chunks_path=chunks_path,
    )


