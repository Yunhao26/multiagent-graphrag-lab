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
import logging
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

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

logger = logging.getLogger(__name__)


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


def _upsert_struct_edge(
    graph: nx.MultiDiGraph,
    src: str,
    dst: str,
    *,
    key: str,
    edge_type: str,
    evidence_chunk_id: str = "",
    confidence: float = 1.0,
) -> None:
    """Insert or update a structural edge with merged evidence chunk ids."""

    if not src or not dst:
        return
    if graph.has_edge(src, dst, key=key):
        attrs = graph.get_edge_data(src, dst, key) or {}
        evidence = list(attrs.get("evidence_chunk_ids", []) or [])
        if evidence_chunk_id and evidence_chunk_id not in evidence:
            evidence.append(evidence_chunk_id)
        attrs["evidence_chunk_ids"] = evidence
        attrs["confidence"] = max(float(attrs.get("confidence", 0.0)), float(confidence))
        attrs["type"] = edge_type
        graph[src][dst][key].update(attrs)
        return

    graph.add_edge(
        src,
        dst,
        key=key,
        type=edge_type,
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

    # Document structure tracking for PDF-first GraphRAG:
    # - doc nodes:  doc:<source>
    # - page nodes: page:<source>#p<page>
    # - chunk nodes: <chunk_id>
    # - edges: doc->page, page->chunk, chunk->next_chunk
    doc_chunk_order: Dict[str, List[Tuple[int, int, str]]] = {}
    page_to_first_chunk: Dict[Tuple[str, int], str] = {}

    for ch in chunks:
        meta = dict(getattr(ch, "metadata", {}) or {})
        chunk_id = str(meta.get("chunk_id") or "")
        source = str(meta.get("source") or "").strip()
        doc_type = str(meta.get("type") or "").strip()

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

        # PDF document structure graph (enables graph paths for PDF QA).
        if source and chunk_id:
            # Chunk node
            g.add_node(
                chunk_id,
                node_type="CHUNK",
                source=source,
                page_or_row=meta.get("page_or_row"),
                section=meta.get("section"),
            )

            # Maintain an ordering index for NEXT edges.
            # For PDFs we use page number; otherwise we fall back to 0.
            try:
                page = int(meta.get("page_or_row") or meta.get("page") or 0)
            except Exception:
                page = 0
            m = re.search(r"::c(\d+)\s*$", chunk_id)
            try:
                cidx = int(m.group(1)) if m else 0
            except Exception:
                cidx = 0
            doc_chunk_order.setdefault(source, [])
            doc_chunk_order[source].append((page, cidx, chunk_id))

            # Only build the explicit DOC->PAGE->CHUNK structure for PDF pages.
            if doc_type == "pdf_page":
                doc_node = f"doc:{source}"
                page_node = f"page:{source}#p{page}"
                g.add_node(doc_node, node_type="DOC", source=source)
                g.add_node(page_node, node_type="PAGE", source=source, page=page)

                _upsert_struct_edge(
                    g,
                    doc_node,
                    page_node,
                    key="CONTAINS_PAGE",
                    edge_type="CONTAINS_PAGE",
                    evidence_chunk_id=chunk_id,
                    confidence=1.0,
                )
                _upsert_struct_edge(
                    g,
                    page_node,
                    chunk_id,
                    key="CONTAINS_CHUNK",
                    edge_type="CONTAINS_CHUNK",
                    evidence_chunk_id=chunk_id,
                    confidence=1.0,
                )

                # Track the first chunk seen for each page (useful for page transitions later).
                page_key = (source, page)
                if page_key not in page_to_first_chunk:
                    page_to_first_chunk[page_key] = chunk_id

    # Add NEXT edges between sequential chunks within the same source document.
    for source, items in doc_chunk_order.items():
        items.sort(key=lambda t: (t[0], t[1], t[2]))
        for (_p1, _c1, a), (_p2, _c2, b) in zip(items, items[1:]):
            if not a or not b or a == b:
                continue
            _upsert_struct_edge(
                g,
                a,
                b,
                key="NEXT",
                edge_type="NEXT",
                evidence_chunk_id=b,
                confidence=0.9,
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
    documents: Sequence[Document],
    llm: object | None,
    max_docs: int = 25,
) -> nx.MultiDiGraph:
    """
    Optional hook for LLM-based prerequisite extraction (local or online LLM).

    This implementation is intentionally conservative:
    - It only keeps relations that look like course prerequisites.
    - It attaches the originating chunk_id as edge evidence.
    - If enhancement is not possible, it returns the original graph.
    """

    if llm is None:
        return graph

    if not documents:
        return graph

    try:
        from langchain_experimental.graph_transformers import LLMGraphTransformer
    except Exception:
        return graph

    try:
        transformer = LLMGraphTransformer(
            llm=llm,
            strict_mode=False,
            node_properties=False,
            relationship_properties=False,
            ignore_tool_usage=True,
            additional_instructions=(
                "You are extracting a course prerequisite graph.\n"
                "- Only extract course codes like AI101, ML201, DL301.\n"
                "- Use node id as the exact course code (uppercase) and node type COURSE.\n"
                "- Only extract prerequisite relations.\n"
                "- Use relationship type PREREQUISITE_OF where source is the prerequisite and target is the course.\n"
                "- Ignore all other entities and relations.\n"
            ),
        )

        # Filter: skip structured CSV course rows and focus on unstructured text where an LLM
        # is more likely to add incremental value.
        candidates: List[Document] = []
        for d in documents:
            meta = dict(getattr(d, "metadata", {}) or {})
            if str(meta.get("type") or "") == "course_row":
                continue
            text = str(getattr(d, "page_content", "") or "").strip()
            if not text:
                continue
            # Heuristic: prerequisite relations require at least two course codes in context.
            codes = COURSE_CODE_PATTERN.findall(text.upper())
            if len(set(codes)) < 2:
                continue
            candidates.append(d)

        if not candidates:
            return graph

        # Prefer docs that clearly contain prerequisite chains or keywords.
        def _score(doc: Document) -> Tuple[int, int, int, int]:
            t = str(getattr(doc, "page_content", "") or "")
            up = t.upper()
            has_arrow = 1 if bool(ARROW_CHAIN_PATTERN.search(up)) else 0
            has_prereq = 1 if ("PREREQ" in up or "PREREQUIS" in up) else 0
            n_codes = len(set(COURSE_CODE_PATTERN.findall(up)))
            return (has_arrow, has_prereq, n_codes, len(t))

        candidates.sort(key=_score, reverse=True)
        docs = candidates
        if int(max_docs) > 0:
            docs = candidates[: int(max_docs)]

        graph_docs = transformer.convert_to_graph_documents(docs)

        added = 0
        for gd in graph_docs:
            src_doc = gd.source
            chunk_id = str((src_doc.metadata or {}).get("chunk_id") or "")
            if not chunk_id:
                continue

            for rel in gd.relationships or []:
                src = str(getattr(rel.source, "id", "") or "").upper()
                dst = str(getattr(rel.target, "id", "") or "").upper()
                rtype = str(getattr(rel, "type", "") or "").strip().upper()
                if not src or not dst or src == dst:
                    continue
                if not COURSE_CODE_PATTERN.fullmatch(src) or not COURSE_CODE_PATTERN.fullmatch(dst):
                    continue
                if "PREREQ" not in rtype and "PREREQUISITE" not in rtype and rtype != "PREREQUISITE_OF":
                    continue

                # Direction sanity check:
                # If the opposite direction already exists in the base graph and the proposed
                # direction does not, flip it to match the existing prerequisite direction.
                if graph.has_edge(dst, src, key="PREREQUISITE_OF") and not graph.has_edge(
                    src, dst, key="PREREQUISITE_OF"
                ):
                    logger.info(
                        "Flipping LLM prerequisite direction to match base graph: %s -> %s",
                        dst,
                        src,
                    )
                    src, dst = dst, src

                _upsert_prereq_edge(graph, src, dst, evidence_chunk_id=chunk_id, confidence=0.5)
                added += 1

        if added:
            logger.info("LLM graph enhancement added %s prerequisite edges.", added)

        return graph
    except Exception:
        return graph


def build_community_reports(
    graph: nx.MultiDiGraph,
    *,
    max_communities: int = 12,
    max_edges_per_community: int = 50,
    max_edges_in_summary: int = 15,
) -> Dict[str, Any]:
    """
    Build lightweight community reports for GraphRAG-style "global search".

    This is a deterministic, offline-friendly approximation of the GraphRAG idea:
    - Detect communities on the undirected projection of the prerequisite graph.
    - Create a short textual report per community (stored as a pseudo-chunk).
    - Attach aggregated evidence chunk ids for provenance.
    """

    # Build an undirected graph ONLY for the course prerequisite subgraph.
    # This avoids structural DOC/PAGE/CHUNK nodes polluting the communities.
    course_nodes = [str(n) for n in graph.nodes() if COURSE_CODE_PATTERN.fullmatch(str(n or ""))]
    course_set = set(course_nodes)

    ug = nx.Graph()
    ug.add_nodes_from(course_nodes)
    for u, v, k in graph.edges(keys=True):
        if k != "PREREQUISITE_OF":
            continue
        su = str(u)
        sv = str(v)
        if su in course_set and sv in course_set and su != sv:
            ug.add_edge(su, sv)

    algo = "trivial"
    comm_sets: List[Set[str]] = []
    if ug.number_of_nodes() <= 1:
        comm_sets = [set(ug.nodes())]
        algo = "trivial"
    else:
        try:
            from networkx.algorithms.community import greedy_modularity_communities

            comm_sets = [set(c) for c in greedy_modularity_communities(ug)]
            algo = "greedy_modularity"
        except Exception:
            # Fallback: connected components (stable, fast).
            comm_sets = [set(c) for c in nx.connected_components(ug)]
            algo = "connected_components"

    # Stable ordering: largest first, then lexical.
    comm_sets.sort(key=lambda c: (-len(c), sorted([str(x) for x in c])[:3]))
    comm_sets = comm_sets[: max(1, int(max_communities))]

    communities: List[Dict[str, Any]] = []
    for i, nodes in enumerate(comm_sets):
        community_id = f"c{i}"
        chunk_id = f"community:{community_id}"
        node_list = sorted([str(n) for n in nodes if str(n)])

        # Collect prerequisite edges and aggregated evidence ids within this community.
        edges: List[Dict[str, Any]] = []
        evidence_ids: Set[str] = set()
        for u, v, k, attrs in graph.edges(keys=True, data=True):
            if k != "PREREQUISITE_OF":
                continue
            if u not in nodes or v not in nodes:
                continue
            ev = [str(x) for x in list((attrs or {}).get("evidence_chunk_ids", []) or []) if str(x)]
            for cid in ev:
                evidence_ids.add(cid)
            edges.append(
                {
                    "source": str(u),
                    "target": str(v),
                    "relation": "PREREQUISITE_OF",
                    "confidence": float((attrs or {}).get("confidence", 0.0) or 0.0),
                    "evidence_chunk_ids": ev[:5],
                }
            )
        edges.sort(key=lambda e: (e.get("source", ""), e.get("target", "")))
        if int(max_edges_per_community) > 0:
            edges = edges[: int(max_edges_per_community)]

        # Deterministic report text.
        lines: List[str] = []
        lines.append(f"Community {community_id} (size={len(node_list)})")
        if node_list:
            preview = node_list[: min(10, len(node_list))]
            lines.append(
                "Courses: "
                + ", ".join(preview)
                + (" ..." if len(node_list) > len(preview) else "")
            )
        if edges:
            lines.append("")
            lines.append("Prerequisite edges:")
            for e in edges[: max(1, int(max_edges_in_summary))]:
                lines.append(f"- {e['source']} -> {e['target']}")

        report_text = "\n".join(lines).strip()
        communities.append(
            {
                "community_id": community_id,
                "chunk_id": chunk_id,
                "title": f"Community {community_id}",
                "text": report_text,
                "nodes": node_list,
                "edges": edges,
                "evidence_chunk_ids": sorted(list(evidence_ids)),
                "metadata": {
                    "type": "community_report",
                    "source": "community_reports.json",
                    "section": f"Community {community_id}",
                    "community_id": community_id,
                    "nodes": node_list,
                },
            }
        )

    return {"algo": algo, "communities": communities}

