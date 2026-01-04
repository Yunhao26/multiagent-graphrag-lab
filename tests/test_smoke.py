"""Smoke tests for importability and the minimal offline QA path."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_imports() -> None:
    import app.main  # noqa: F401
    import app.schemas  # noqa: F401
    import app.settings  # noqa: F401
    import agents.graph  # noqa: F401
    import agents.nodes  # noqa: F401
    import agents.llm  # noqa: F401
    import ingestion.build_index  # noqa: F401
    import ingestion.loaders  # noqa: F401
    import ingestion.splitters  # noqa: F401
    import ingestion.vector_index  # noqa: F401
    import ingestion.graph_index  # noqa: F401
    import ingestion.stores  # noqa: F401
    import ui.client  # noqa: F401


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_health_and_qa_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Configure index locations in a temp directory.
    index_dir = tmp_path / "index"
    chroma_dir = index_dir / "chroma"
    graph_path = index_dir / "graph.json.gz"
    chunks_path = index_dir / "chunks.jsonl"
    manifest_path = index_dir / "manifest.json"

    monkeypatch.setenv("INDEX_DIR", str(index_dir))
    monkeypatch.setenv("CHROMA_DIR", str(chroma_dir))
    monkeypatch.setenv("GRAPH_PATH", str(graph_path))

    # Force the fully offline embedding fallback (no model download).
    monkeypatch.setenv("FORCE_HASH_EMBEDDINGS", "1")

    from app.settings import Settings
    from ingestion.build_index import build_index

    settings = Settings.from_env()
    _ = build_index(data_dir=Path("data/sources"), out_dir=index_dir, settings=settings)
    assert chunks_path.exists()
    assert manifest_path.exists()
    assert graph_path.exists()
    assert chroma_dir.exists()

    # Direct offline QA (no server).
    from agents.graph import answer_query

    result = answer_query(
        query="AI101 的先修课是什么？",
        brief=True,
        top_k=5,
        k_hop=2,
        settings=settings,
    )
    assert isinstance(result.get("answer"), str)
    assert (result.get("citations") or []) or (result.get("graph_paths") or [])

    from app.main import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

    r = client.post(
        "/qa",
        json={"query": "What is the prerequisite chain to DL301?", "brief": True, "top_k": 5, "k_hop": 2},
    )
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload.get("answer"), str)
    assert "citations" in payload
    assert "graph_paths" in payload
    assert "debug" in payload


