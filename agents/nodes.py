"""
LangGraph node implementations for the GraphRAG MVP.

This MVP implements a stable, offline-first Hybrid GraphRAG pipeline:
- Vector retrieval: Chroma similarity search (with score)
- Graph retrieval: k-hop expansion + path extraction on a NetworkX MultiDiGraph
- Fusion: dedupe + re-rank with a simple graph bonus
- Generation: offline extractive summary, optionally enhanced with ChatOpenAI
"""

from __future__ import annotations

import itertools
import time
from typing import Any, Dict, List, Optional, Protocol, Set, TypedDict, Tuple

import networkx as nx

from agents.llm import generate_answer, llm_mode
from app.settings import Settings
from ingestion.graph_index import COURSE_CODE_PATTERN
from ingestion.vector_index import similarity_search_with_score


class Runtime(Protocol):
    """Protocol for runtime dependencies used by nodes."""

    vectorstore: Any
    graph: nx.MultiDiGraph
    chunk_store: Dict[str, Dict[str, Any]]


class GraphRAGState(TypedDict, total=False):
    """State carried across LangGraph nodes."""

    query: str
    brief: bool
    top_k: int
    k_hop: int

    route: str
    seed_entities: List[str]

    vector_hits: List[Dict[str, Any]]
    graph_paths: List[Dict[str, Any]]

    evidence_pack: List[Dict[str, Any]]
    answer: str
    citations: List[Dict[str, Any]]
    debug: Dict[str, Any]


def extract_course_codes(text: str) -> List[str]:
    """Extract course codes like AI101, ML201 from free-form text."""

    codes = COURSE_CODE_PATTERN.findall((text or "").upper())
    # Preserve order while deduping.
    seen: Set[str] = set()
    out: List[str] = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _ensure_debug(state: GraphRAGState) -> Dict[str, Any]:
    debug = dict(state.get("debug", {}) or {})
    debug.setdefault("latency_ms", {})
    debug.setdefault("evidence_counts", {})
    return debug


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def route_node(state: GraphRAGState) -> GraphRAGState:
    """Route the query (MVP: always 'hybrid') and do entity linking."""

    t0 = time.perf_counter()
    query = state.get("query", "")
    seeds = extract_course_codes(query)
    requested = str(state.get("route") or "").strip().lower()
    route = requested if requested in {"hybrid", "vector_only"} else "hybrid"
    debug = _ensure_debug(state)
    debug["route"] = route
    debug["latency_ms"]["route"] = _ms(t0)
    return {**state, "route": route, "seed_entities": seeds, "debug": debug}


def _vector_score(distance: float) -> float:
    """Convert a distance-like number to a 0..1 similarity score (higher is better)."""

    try:
        d = float(distance)
    except Exception:
        return 0.0
    if d < 0:
        return 1.0
    return 1.0 / (1.0 + d)


def _resolve_chunk(runtime: Runtime, chunk_id: str) -> Tuple[str, Dict[str, Any]]:
    rec = runtime.chunk_store.get(chunk_id, {})
    text = str(rec.get("text") or "")
    meta = dict(rec.get("metadata", {}) or {})
    return text, meta


def _best_edge_attrs(graph: nx.MultiDiGraph, u: str, v: str) -> Dict[str, Any]:
    data = graph.get_edge_data(u, v) or {}
    if not data:
        return {"type": "PREREQUISITE_OF", "evidence_chunk_ids": []}
    if "PREREQUISITE_OF" in data:
        return dict(data["PREREQUISITE_OF"] or {})
    # Otherwise choose the edge with the highest confidence.
    best_key = None
    best_conf = -1.0
    for k, attrs in data.items():
        conf = float((attrs or {}).get("confidence", 0.0))
        if conf > best_conf:
            best_conf = conf
            best_key = k
    return dict(data.get(best_key) or {})


def _path_to_steps(graph: nx.MultiDiGraph, nodes: List[str]) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    for u, v in zip(nodes, nodes[1:]):
        attrs = _best_edge_attrs(graph, u, v)
        steps.append(
            {
                "source": u,
                "relation": str(attrs.get("type") or "PREREQUISITE_OF"),
                "target": v,
                "evidence_chunk_ids": list(attrs.get("evidence_chunk_ids", []) or []),
            }
        )
    return steps


def _score_path(steps: List[Dict[str, Any]]) -> float:
    path_len = max(1, len(steps))
    evidence = sum(len(s.get("evidence_chunk_ids", []) or []) for s in steps)
    return (1.0 / (1.0 + path_len)) + 0.05 * float(evidence)


def retrieve_graph_paths(
    graph: nx.MultiDiGraph, *, seeds: List[str], k_hop: int, max_paths: int = 5
) -> List[Dict[str, Any]]:
    """Generate ranked graph paths from seeds using simple BFS/shortest-path logic."""

    seeds_in_graph = [s for s in seeds if s in graph]
    if not seeds_in_graph or k_hop <= 0:
        return []

    candidates: List[Tuple[float, Tuple[str, ...], List[Dict[str, Any]]]] = []
    seen_paths: Set[Tuple[str, ...]] = set()

    rev = graph.reverse(copy=False)

    for s in seeds_in_graph:
        # Prerequisite chain candidates (ancestors -> seed).
        try:
            lengths = nx.single_source_shortest_path_length(rev, s, cutoff=k_hop)
        except Exception:
            lengths = {}
        for a in lengths.keys():
            if a == s:
                continue
            try:
                nodes = nx.shortest_path(graph, a, s)
            except Exception:
                continue
            key = tuple(nodes)
            if key in seen_paths:
                continue
            steps = _path_to_steps(graph, nodes)
            score = _score_path(steps)
            seen_paths.add(key)
            candidates.append((score, key, steps))

        # Forward chain candidates (seed -> advanced courses).
        try:
            lengths_f = nx.single_source_shortest_path_length(graph, s, cutoff=k_hop)
        except Exception:
            lengths_f = {}
        for b in lengths_f.keys():
            if b == s:
                continue
            try:
                nodes = nx.shortest_path(graph, s, b)
            except Exception:
                continue
            key = tuple(nodes)
            if key in seen_paths:
                continue
            steps = _path_to_steps(graph, nodes)
            score = _score_path(steps)
            seen_paths.add(key)
            candidates.append((score, key, steps))

    # Between seed pairs.
    for a, b in itertools.combinations(seeds_in_graph, 2):
        for src, dst in ((a, b), (b, a)):
            try:
                nodes = nx.shortest_path(graph, src, dst)
            except Exception:
                continue
            key = tuple(nodes)
            if key in seen_paths:
                continue
            steps = _path_to_steps(graph, nodes)
            score = _score_path(steps)
            seen_paths.add(key)
            candidates.append((score, key, steps))

    # Rank and return.
    candidates.sort(key=lambda x: x[0], reverse=True)
    out: List[Dict[str, Any]] = []
    for score, _, steps in candidates[:max_paths]:
        out.append({"path": steps, "score": float(score)})
    return out


def retrieve_node(state: GraphRAGState, *, runtime: Runtime) -> GraphRAGState:
    """Retrieve vector hits and graph paths."""

    t0 = time.perf_counter()
    query = state.get("query", "")
    top_k = int(state.get("top_k", 5))
    k_hop = int(state.get("k_hop", 2))
    route = str(state.get("route") or "hybrid").strip().lower()

    pairs = similarity_search_with_score(runtime.vectorstore, query, k=top_k)
    vector_hits: List[Dict[str, Any]] = []

    for doc, dist in pairs:
        meta = dict(getattr(doc, "metadata", {}) or {})
        chunk_id = str(meta.get("chunk_id") or meta.get("id") or "")
        text, stored_meta = _resolve_chunk(runtime, chunk_id)
        if not stored_meta:
            stored_meta = meta
        if not text:
            text = str(getattr(doc, "page_content", "") or "")
        vector_hits.append(
            {
                "chunk_id": chunk_id,
                "distance": float(dist),
                "vector_score": _vector_score(float(dist)),
                "text": text,
                "metadata": stored_meta,
            }
        )

    # If no explicit entities in the query, bootstrap seeds from retrieved docs.
    seeds = list(state.get("seed_entities", []) or [])
    if not seeds:
        for h in vector_hits[: max(1, min(3, len(vector_hits)))]:
            seeds.extend(extract_course_codes(h.get("text", "")))
        # De-dupe
        seeds = extract_course_codes(" ".join(seeds))

    graph_paths: List[Dict[str, Any]] = []
    if route != "vector_only":
        graph_paths = retrieve_graph_paths(
            runtime.graph, seeds=seeds, k_hop=k_hop, max_paths=5
        )

    debug = _ensure_debug(state)
    debug["latency_ms"]["retrieve"] = _ms(t0)
    debug["evidence_counts"]["vector_hits"] = len(vector_hits)
    debug["evidence_counts"]["graph_paths"] = len(graph_paths)

    return {**state, "seed_entities": seeds, "vector_hits": vector_hits, "graph_paths": graph_paths, "debug": debug}


def fuse_node(state: GraphRAGState, *, runtime: Runtime) -> GraphRAGState:
    """Fuse vector hits and graph evidence into a small evidence pack."""

    t0 = time.perf_counter()
    vector_hits = list(state.get("vector_hits", []) or [])
    graph_paths = list(state.get("graph_paths", []) or [])

    graph_evidence_ids: Set[str] = set()
    for p in graph_paths:
        for step in p.get("path", []) or []:
            for cid in step.get("evidence_chunk_ids", []) or []:
                graph_evidence_ids.add(str(cid))

    alpha = 1.0
    beta = 0.75

    by_chunk: Dict[str, Dict[str, Any]] = {}
    for h in vector_hits:
        cid = str(h.get("chunk_id") or "")
        if not cid:
            continue
        by_chunk[cid] = dict(h)

    # Add graph-only evidence chunks that were not retrieved by vector search.
    for cid in graph_evidence_ids:
        if cid in by_chunk:
            continue
        text, meta = _resolve_chunk(runtime, cid)
        if not meta and not text:
            continue
        by_chunk[cid] = {
            "chunk_id": cid,
            "distance": None,
            "vector_score": 0.0,
            "text": text,
            "metadata": meta,
        }

    fused: List[Dict[str, Any]] = []
    for cid, h in by_chunk.items():
        vscore = float(h.get("vector_score") or 0.0)
        gb = 1.0 if cid in graph_evidence_ids else 0.0
        final_score = alpha * vscore + beta * gb
        fused.append(
            {
                "chunk_id": cid,
                "text": h.get("text", ""),
                "metadata": h.get("metadata", {}),
                "vector_score": vscore,
                "graph_bonus": gb,
                "final_score": float(final_score),
            }
        )

    fused.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
    evidence_pack = fused[:8]

    citations: List[Dict[str, Any]] = []
    for ev in evidence_pack:
        meta = dict(ev.get("metadata", {}) or {})
        por = meta.get("page_or_row")
        if por is None:
            por = meta.get("row")
        try:
            por_i: Optional[int] = int(por) if por is not None and str(por).strip() else None
        except Exception:
            por_i = None
        citations.append(
            {
                "chunk_id": str(ev.get("chunk_id") or ""),
                "source": str(meta.get("source") or ""),
                "page_or_row": por_i,
                "section": meta.get("section"),
            }
        )

    debug = _ensure_debug(state)
    debug["latency_ms"]["fuse"] = _ms(t0)
    debug["evidence_counts"]["graph_evidence_chunk_ids"] = len(graph_evidence_ids)
    debug["evidence_counts"]["fused_unique_chunks"] = len(by_chunk)
    debug["evidence_counts"]["evidence_pack"] = len(evidence_pack)

    return {**state, "evidence_pack": evidence_pack, "citations": citations, "debug": debug}


def generate_node(state: GraphRAGState, *, settings: Settings) -> GraphRAGState:
    """Generate the final answer in offline mode or optional OpenAI mode."""

    t0 = time.perf_counter()
    query = state.get("query", "")
    brief = bool(state.get("brief", True))
    evidence_pack = list(state.get("evidence_pack", []) or [])
    graph_paths = list(state.get("graph_paths", []) or [])

    mode = llm_mode(settings)
    answer = generate_answer(
        query=query,
        brief=brief,
        evidence_pack=evidence_pack,
        graph_paths=graph_paths,
        settings=settings,
    )

    debug = _ensure_debug(state)
    debug["mode"] = mode
    debug["latency_ms"]["generate"] = _ms(t0)

    return {**state, "answer": answer, "debug": debug}


