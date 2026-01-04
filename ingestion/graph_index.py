"""
Graph index utilities (NetworkX).

This MVP focuses on a course prerequisite graph:
- Nodes: course_code
- Edges: prerequisite -> course

Rule-based extraction is always available (offline safety net).
Optional future enhancement:
- When an online LLM is configured, ``LLMGraphTransformer`` can extract a richer graph
  from unstructured text.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

import networkx as nx
from networkx.readwrite import json_graph

try:
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    from langchain.schema import Document  # type: ignore

from ingestion.utils import gzip_json_dump, gzip_json_load

COURSE_CODE_PATTERN = re.compile(r"\b[A-Z]{2,6}\d{3}\b")
ARROW_CHAIN_PATTERN = re.compile(
    r"\b(?P<chain>(?:[A-Z]{2,6}\d{3}\s*->\s*)+[A-Z]{2,6}\d{3})\b"
)
PREREQ_SENTENCE_PATTERN = re.compile(
    r"\b(?P<src>[A-Z]{2,6}\d{3})\b\s+(?:is\s+)?prerequisite\s+(?:for|of)\s+\b(?P<dst>[A-Z]{2,6}\d{3})\b",
    re.IGNORECASE,
)


def parse_prereq_field(value: object) -> List[str]:
    """Parse a prerequisite field into a list of course codes."""

    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "none":
        return []
    parts = re.split(r"[;,]\s*|\s+", text)
    codes: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = COURSE_CODE_PATTERN.search(p.upper())
        if m:
            codes.append(m.group(0))
    return codes


def _upsert_prereq_edge(
    graph: nx.MultiDiGraph, src: str, dst: str, *, evidence_chunk_id: str, confidence: float
) -> None:
    """
    Insert or update a PREREQUISITE_OF edge.

    Edge key is fixed to ``PREREQUISITE_OF`` to allow evidence merging.
    """

    src = src.upper()
    dst = dst.upper()
    key = "PREREQUISITE_OF"

    if graph.has_edge(src, dst, key=key):
        attrs = graph.get_edge_data(src, dst, key) or {}
        evidence = list(attrs.get("evidence_chunk_ids", []))
        if evidence_chunk_id and evidence_chunk_id not in evidence:
            evidence.append(evidence_chunk_id)
        attrs["evidence_chunk_ids"] = evidence
        attrs["confidence"] = max(float(attrs.get("confidence", 0.0)), float(confidence))
        attrs["type"] = "PREREQUISITE_OF"
        graph[src][dst][key].update(attrs)
        return

    graph.add_edge(
        src,
        dst,
        key=key,
        type="PREREQUISITE_OF",
        evidence_chunk_ids=[evidence_chunk_id] if evidence_chunk_id else [],
        confidence=float(confidence),
    )


def _edges_from_text(text: str) -> Set[Tuple[str, str]]:
    """Extract prerequisite edges from free-form text using simple rules."""

    edges: Set[Tuple[str, str]] = set()

    # Arrow chains: AI101 -> ML201 -> DL301
    for m in ARROW_CHAIN_PATTERN.finditer(text or ""):
        chain = m.group("chain")
        codes = COURSE_CODE_PATTERN.findall(chain.upper())
        for a, b in zip(codes, codes[1:]):
            edges.add((a, b))

    # "X is prerequisite for Y" / "X prerequisite of Y"
    for m in PREREQ_SENTENCE_PATTERN.finditer(text or ""):
        edges.add((m.group("src").upper(), m.group("dst").upper()))

    return edges


def build_graph_from_chunks(chunks: Sequence[Document]) -> nx.MultiDiGraph:
    """
    Build a prerequisite graph from indexed chunks.

    Rules (offline safety net):
    1) From course CSV rows (chunk metadata prerequisite): add PREREQUISITE_OF edges
       with confidence 0.8.
    2) From chunk text: regex-based extraction of prerequisite relations with
       confidence 0.6.
    """

    g = nx.MultiDiGraph()

    for ch in chunks:
        meta = dict(getattr(ch, "metadata", {}) or {})
        chunk_id = str(meta.get("chunk_id") or "")

        # Node attributes from course rows.
        course_code = meta.get("course_code")
        if isinstance(course_code, str) and course_code:
            cc = course_code.upper()
            g.add_node(
                cc,
                year=meta.get("year"),
                program=meta.get("program"),
                course_name=meta.get("course_name"),
            )
            prereq_field = meta.get("prerequisite")
            for p in parse_prereq_field(prereq_field):
                g.add_node(p)
                _upsert_prereq_edge(
                    g, p, cc, evidence_chunk_id=chunk_id, confidence=0.8
                )

        # Text-based extraction across all chunks.
        for src, dst in _edges_from_text(getattr(ch, "page_content", "") or ""):
            g.add_node(src)
            g.add_node(dst)
            _upsert_prereq_edge(
                g, src, dst, evidence_chunk_id=chunk_id, confidence=0.6
            )

    return g


def save_graph(graph: nx.MultiDiGraph, path: Path) -> None:
    """Persist the graph as node-link JSON compressed with gzip."""

    # Explicit edges key to avoid NetworkX future warnings and keep the format stable.
    try:
        data = json_graph.node_link_data(graph, edges="edges")
    except TypeError:  # pragma: no cover (older NetworkX)
        data = json_graph.node_link_data(graph)
    gzip_json_dump(data, path)


def load_graph(path: Path) -> nx.MultiDiGraph:
    """Load a graph produced by ``save_graph``."""

    data = gzip_json_load(path)
    # Support both "edges" and legacy "links" keys, and avoid future warnings by
    # always passing the edges kwarg when supported.
    try:
        g = json_graph.node_link_graph(data, directed=True, multigraph=True, edges="edges")
    except TypeError:  # pragma: no cover (older NetworkX)
        g = json_graph.node_link_graph(data, directed=True, multigraph=True)
    except Exception:
        g = json_graph.node_link_graph(data, directed=True, multigraph=True, edges="links")
    if not isinstance(g, nx.MultiDiGraph):
        g = nx.MultiDiGraph(g)  # pragma: no cover
    return g


def k_hop_nodes(*, graph: nx.MultiDiGraph, seeds: Iterable[str], k: int) -> Set[str]:
    """Return nodes within k hops (undirected) from the seed set."""

    if k <= 0:
        return set(seeds)
    undirected = graph.to_undirected()
    nodes: Set[str] = set()
    for s in seeds:
        if s not in undirected:
            continue
        lengths = nx.single_source_shortest_path_length(undirected, s, cutoff=k)
        nodes.update(lengths.keys())
    return nodes


def edges_as_facts(*, graph: nx.MultiDiGraph, nodes: Set[str], limit: int = 200) -> List[str]:
    """Render edges in a subgraph as simple 'A -> B' strings."""

    facts: List[str] = []
    count = 0
    for u, v, k in graph.edges(keys=True):
        if u in nodes and v in nodes:
            facts.append(f"{u} -> {v} ({k})")
            count += 1
            if count >= limit:
                break
    return facts


def maybe_enhance_graph_with_llm(
    graph: nx.MultiDiGraph,
    *,
    documents_text: str,
    llm: object | None,
) -> nx.MultiDiGraph:
    """
    Optional hook for LLM-based graph extraction (future enhancement).

    If an online LLM is available, ``LLMGraphTransformer`` can extract additional
    nodes/edges from free-form text. This function is intentionally conservative and
    returns the original graph if enhancement is not possible.
    """

    if llm is None:
        return graph

    try:
        from langchain_experimental.graph_transformers import LLMGraphTransformer
    except Exception:
        return graph

    try:
        transformer = LLMGraphTransformer(llm=llm)
        _ = transformer  # placeholder: connect extracted relations to NetworkX later
        # MVP: keep as no-op until the next iteration.
        return graph
    except Exception:
        return graph


