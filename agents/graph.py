"""
LangGraph-based agent graph for the GraphRAG MVP.

This module wires together:
- Vector retrieval (Chroma)
- Graph expansion (NetworkX)
- Answer synthesis (offline fallback or ChatOpenAI)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import networkx as nx

from app.settings import Settings
from ingestion.graph_index import load_graph
from ingestion.utils import read_jsonl
from ingestion.vector_index import get_embeddings, load_vector_store


class AgentError(RuntimeError):
    """Raised when the agent cannot load artifacts or execute a run."""


@dataclass(frozen=True)
class AgentRuntime:
    """Runtime dependencies required by the agent."""

    vectorstore: Any
    graph: nx.MultiDiGraph
    chunk_store: Dict[str, Dict[str, Any]]


def _load_chunk_store(chunks_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load ``chunks.jsonl`` into a dict keyed by chunk_id.

    Each record:
    - chunk_id
    - text
    - metadata
    """

    store: Dict[str, Dict[str, Any]] = {}
    for r in read_jsonl(chunks_path):
        cid = str(r.get("chunk_id") or "")
        if not cid:
            continue
        store[cid] = {"text": r.get("text", ""), "metadata": r.get("metadata", {})}
    return store


def _build_langgraph(runtime: AgentRuntime, settings: Settings) -> Any:
    """Build and compile a LangGraph state machine."""

    try:
        from langgraph.graph import END, START, StateGraph
    except Exception as exc:  # pragma: no cover
        raise AgentError("langgraph is not available or incompatible.") from exc

    from agents.nodes import GraphRAGState, fuse_node, generate_node, retrieve_node, route_node

    graph = StateGraph(GraphRAGState)

    def _route(state: GraphRAGState) -> GraphRAGState:
        return route_node(state)

    def _retrieve(state: GraphRAGState) -> GraphRAGState:
        return retrieve_node(state, runtime=runtime)

    def _fuse(state: GraphRAGState) -> GraphRAGState:
        return fuse_node(state, runtime=runtime)

    def _generate(state: GraphRAGState) -> GraphRAGState:
        return generate_node(state, settings=settings)

    graph.add_node("route", _route)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("fuse", _fuse)
    graph.add_node("generate", _generate)

    graph.add_edge(START, "route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "fuse")
    graph.add_edge("fuse", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


class GraphRAGAgent:
    """A small wrapper around the compiled LangGraph application."""

    def __init__(self, runtime: AgentRuntime, settings: Settings):
        self._runtime = runtime
        self._settings = settings
        self._app = _build_langgraph(runtime=runtime, settings=settings)

    def invoke(
        self,
        *,
        query: str,
        brief: bool,
        top_k: int,
        k_hop: int,
        route: str | None = None,
    ) -> Dict[str, Any]:
        """Run the agent for a single query."""

        try:
            return self._app.invoke(
                {
                    "query": query,
                    "brief": brief,
                    "top_k": int(top_k),
                    "k_hop": int(k_hop),
                    "route": route,
                }
            )
        except Exception as exc:
            raise AgentError(f"Agent invocation failed: {exc}") from exc


@lru_cache(maxsize=4)
def _load_runtime_cached(
    chroma_dir: str,
    graph_path: str,
    chunks_path: str,
    embedding_model: str,
    force_hash_embeddings: bool,
) -> AgentRuntime:
    """Load persisted artifacts once per unique configuration."""

    embeddings = get_embeddings(
        model_name=embedding_model, force_hash=force_hash_embeddings
    )
    vectorstore = load_vector_store(chroma_dir=Path(chroma_dir), embeddings=embeddings)
    graph = load_graph(Path(graph_path))
    chunk_store = _load_chunk_store(Path(chunks_path))
    return AgentRuntime(vectorstore=vectorstore, graph=graph, chunk_store=chunk_store)


def get_agent(settings: Settings) -> GraphRAGAgent:
    """Create (or reuse) a configured agent instance."""

    if not settings.index_ready():
        raise AgentError("Index artifacts are missing; build the index first.")

    runtime = _load_runtime_cached(
        chroma_dir=str(settings.chroma_dir),
        graph_path=str(settings.graph_path),
        chunks_path=str(settings.chunks_path),
        embedding_model=settings.embedding_model,
        force_hash_embeddings=settings.force_hash_embeddings,
    )
    return GraphRAGAgent(runtime=runtime, settings=settings)


def answer_query(
    *,
    query: str,
    brief: bool = True,
    top_k: int = 5,
    k_hop: int = 2,
    route: str | None = None,
    settings: Settings | None = None,
) -> Dict[str, Any]:
    """Convenience function used by FastAPI and tests."""

    import time

    t0 = time.perf_counter()
    settings_obj = settings or Settings.from_env()
    agent = get_agent(settings_obj)
    state = agent.invoke(query=query, brief=brief, top_k=top_k, k_hop=k_hop, route=route)

    debug = dict(state.get("debug", {}) or {})
    latency = dict(debug.get("latency_ms", {}) or {})
    latency["total"] = (time.perf_counter() - t0) * 1000.0
    debug["latency_ms"] = latency

    return {
        "answer": state.get("answer", ""),
        "citations": state.get("citations", []),
        "graph_paths": state.get("graph_paths", []),
        "debug": debug,
    }


