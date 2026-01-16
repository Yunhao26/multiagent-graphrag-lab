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
import os
import re
import time
from functools import lru_cache
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
    entity_index: Dict[str, Any]
    community_index: Dict[str, Any]


class GraphRAGState(TypedDict, total=False):
    """State carried across LangGraph nodes."""

    query: str
    brief: bool
    top_k: int
    k_hop: int

    route: str
    seed_entities: List[str]
    retrieval_query: str

    vector_hits: List[Dict[str, Any]]
    graph_paths: List[Dict[str, Any]]
    graph_hits: List[Dict[str, Any]]

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


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(str(raw).strip() or default)
    except Exception:
        return int(default)


def _sanitize_one_line(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    # Prefer the first non-empty line.
    for line in s.splitlines():
        line = line.strip()
        if line:
            s = line
            break
    # Remove surrounding quotes.
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s


@lru_cache(maxsize=256)
def _rewrite_query_ollama_cached(base_url: str, model: str, query: str) -> str:
    """Rewrite a query for retrieval using a local Ollama model (cached)."""

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_ollama import ChatOllama  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Ollama query rewrite dependencies are not available.") from exc

    llm = ChatOllama(
        base_url=base_url,
        model=model,
        temperature=0.0,
        num_predict=96,
        client_kwargs={"timeout": float(os.environ.get("QUERY_REWRITE_TIMEOUT_S", "30") or 30)},
    )
    system = SystemMessage(
        content=(
            "You rewrite user questions into short, high-recall search queries for semantic retrieval.\n"
            "Context: the corpus is mainly French academic regulations (PDF).\n"
            "Rules:\n"
            "- Output ONLY the rewritten query string. No quotes. No explanations.\n"
            "- If the input is Chinese, translate to French and add relevant French keywords.\n"
            "- If the question asks about consequences/penalties of absences, include keywords like:\n"
            "  absence, assiduité, non-assiduité, sanction, avertissement, exclusion, quota.\n"
            "- Include numeric thresholds if they appear or are strongly implied.\n"
        )
    )
    human = HumanMessage(content=f"User question: {query}\nRewritten search query:")
    msg = llm.invoke([system, human])
    return _sanitize_one_line(getattr(msg, "content", str(msg)))


def _rewrite_query_for_retrieval(query: str, *, settings: Settings) -> Tuple[str, bool]:
    """
    Conditionally rewrite the query for better cross-lingual retrieval.

    Returns (retrieval_query, rewritten_flag).
    """

    if not _bool_env("ENABLE_QUERY_REWRITE", True):
        return query, False
    if settings.llm_provider.strip().lower() != "ollama":
        return query, False
    if not _contains_cjk(query):
        return query, False

    base_url = (os.environ.get("OLLAMA_BASE_URL") or "").strip() or settings.ollama_base_url
    model = (os.environ.get("OLLAMA_MODEL") or "").strip() or "mistral:7b"
    try:
        rewritten = _rewrite_query_ollama_cached(base_url, model, query)
        rewritten = _sanitize_one_line(rewritten)
        if rewritten and rewritten.lower() != (query or "").strip().lower():
            return rewritten, True
        return query, False
    except Exception:
        return query, False


def _allow_model_download() -> bool:
    return os.environ.get("ALLOW_MODEL_DOWNLOAD", "0").strip().lower() in {"1", "true", "yes"}


@lru_cache(maxsize=2)
def _load_cross_encoder_cached(model_name: str, device: str) -> Any:
    """
    Load a sentence-transformers CrossEncoder for reranking (cached across queries).

    Note: We keep this optional; if dependencies/models are missing, callers should
    catch and skip reranking.
    """

    # Respect the project's offline-first defaults unless explicitly allowed.
    if not _allow_model_download():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    else:
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"

    from sentence_transformers import CrossEncoder  # type: ignore

    return CrossEncoder(model_name, device=device)


def _resolve_rerank_device(requested: str | None) -> str:
    raw = (requested or os.environ.get("CE_RERANK_DEVICE") or "auto").strip()
    if raw.lower() != "auto":
        return raw
    try:
        import torch

        return "cuda" if bool(getattr(torch, "cuda", None)) and torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _cross_encoder_rerank(
    hits: List[Dict[str, Any]],
    *,
    query: str,
    alt_query: str | None,
    model_name: str,
    device: str,
    top_n: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Rerank hits using a cross-encoder. Returns (reranked_hits, debug_info).

    Strategy:
    - Score (query, chunk_text) pairs.
    - If alt_query is provided and differs, score both and take max per chunk.
    """

    debug: Dict[str, Any] = {"enabled": True, "model": model_name, "device": device, "top_n": int(top_n)}
    if not hits:
        debug["skipped_reason"] = "no_hits"
        return hits, debug

    n = max(0, min(int(top_n), len(hits)))
    if n <= 0:
        debug["skipped_reason"] = "top_n<=0"
        return hits, debug

    ce = _load_cross_encoder_cached(model_name, device)

    texts = [str(h.get("text") or "") for h in hits[:n]]
    pairs1 = [(str(query or ""), t) for t in texts]
    scores1 = ce.predict(pairs1)
    try:
        scores1_list = [float(x) for x in list(scores1)]
    except Exception:
        scores1_list = [float(x) for x in scores1]  # type: ignore

    scores = scores1_list
    used_alt = False
    aq = str(alt_query or "").strip()
    if aq and aq.lower() != str(query or "").strip().lower():
        pairs2 = [(aq, t) for t in texts]
        scores2 = ce.predict(pairs2)
        try:
            scores2_list = [float(x) for x in list(scores2)]
        except Exception:
            scores2_list = [float(x) for x in scores2]  # type: ignore
        scores = [max(a, b) for a, b in zip(scores1_list, scores2_list)]
        used_alt = True

    debug["used_alt_query"] = bool(used_alt)

    subset = []
    for h, s in zip(hits[:n], scores):
        hh = dict(h)
        # Keep the previous retrieval score for debugging.
        hh["vector_score_pre_rerank"] = float(h.get("vector_score", 0.0) or 0.0)
        hh["ce_score"] = float(s)
        # Make rerank influence downstream fusion/evidence selection.
        hh["vector_score"] = float(s)
        subset.append(hh)

    subset.sort(key=lambda x: float(x.get("ce_score", 0.0)), reverse=True)
    out = subset + hits[n:]
    debug["reranked"] = int(n)
    debug["ce_score_max"] = float(max(scores)) if scores else 0.0
    debug["ce_score_min"] = float(min(scores)) if scores else 0.0
    return out, debug


def _keyword_weights_for_query(original_query: str, retrieval_query: str) -> Dict[str, float]:
    """
    Build weighted keywords for fallback re-ranking of candidate chunks.

    This is intentionally lightweight and offline-friendly.
    """

    q = (original_query or "").lower()
    rq = (retrieval_query or "").lower()
    weights: Dict[str, float] = {}

    # Detect if the user is asking about attendance/absence/penalties.
    about_absence = any(
        k in q
        for k in [
            "absence",
            "absen",
            "attendance",
            "assiduit",
            "sanction",
            "penalt",
            "consequence",
            "缺席",
            "缺勤",
            "缺课",
            "出勤",
            "后果",
            "处罚",
            "惩罚",
        ]
    ) or any(k in rq for k in ["absence", "assiduit", "sanction"])

    if about_absence:
        for kw, w in [
            ("absence", 2.0),
            ("absences", 2.0),
            ("assiduit", 2.0),  # matches assiduité / assiduite
            ("non-assiduit", 2.5),
            ("sanction", 2.0),
            ("sanctions", 2.0),
            ("quota", 1.8),
            ("avertissement", 1.8),
            ("exclusion", 2.2),
            ("rattrapage", 1.5),
            ("retard", 1.0),
            ("19e", 2.0),
            ("29e", 2.0),
            ("39e", 2.0),
        ]:
            weights[kw] = max(weights.get(kw, 0.0), float(w))

    # Add tokens from the rewritten query (low weight).
    toks = [t for t in re.split(r"[^\w]+", rq) if t]
    for t in toks:
        if len(t) < 3:
            continue
        if t.isdigit():
            continue
        weights[t] = max(weights.get(t, 0.0), 0.6)

    return weights


def _keyword_score(text: str, weights: Dict[str, float]) -> float:
    if not text or not weights:
        return 0.0
    tl = text.lower()
    total = float(sum(weights.values())) if weights else 0.0
    if total <= 0:
        return 0.0
    hit = 0.0
    for kw, w in weights.items():
        if kw and kw in tl:
            hit += float(w)
    return max(0.0, min(1.0, hit / total))


def _top_keywords_query(weights: Dict[str, float], *, limit: int = 8) -> str:
    """Build a short keyword query from the highest-weighted keywords."""

    if not weights:
        return ""
    items = sorted(weights.items(), key=lambda x: (-float(x[1]), str(x[0])))
    toks: List[str] = []
    for kw, _w in items:
        s = str(kw or "").strip()
        if not s:
            continue
        if len(s) < 3:
            continue
        toks.append(s)
        if len(toks) >= int(limit):
            break
    return " ".join(toks).strip()


def _ensure_debug(state: GraphRAGState) -> Dict[str, Any]:
    debug = dict(state.get("debug", {}) or {})
    debug.setdefault("latency_ms", {})
    debug.setdefault("evidence_counts", {})
    return debug


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _entity_link_query(query: str, *, runtime: Runtime | None, limit: int = 10) -> List[str]:
    """
    Link a query to entity ids (MVP: course codes) using a lightweight alias index.

    Priority:
    1) Explicit course codes in text (AI101)
    2) Alias substring matches (e.g., "Deep Learning" -> DL301) from entities.json
    """

    seeds = extract_course_codes(query)
    if not runtime:
        return seeds

    ent = dict(getattr(runtime, "entity_index", {}) or {})
    alias_to_ids = dict(ent.get("alias_to_ids", {}) or {})
    if not alias_to_ids:
        return seeds

    q = (query or "").lower()
    matched: List[str] = []
    # Conservative: skip very short aliases to avoid false positives.
    for alias, ids in alias_to_ids.items():
        if not isinstance(alias, str):
            continue
        a = alias.strip()
        if len(a) < 4:
            continue
        if a and a in q:
            if isinstance(ids, list):
                matched.extend([str(x) for x in ids if str(x)])
            else:
                matched.append(str(ids))

    # Preserve order while deduping.
    out: List[str] = []
    seen: Set[str] = set()
    for x in seeds + matched:
        x = str(x or "").strip().upper()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)

    return out[: max(1, int(limit))]


def route_node(state: GraphRAGState, *, runtime: Runtime | None = None) -> GraphRAGState:
    """Route the query (MVP: always 'hybrid') and do entity linking."""

    t0 = time.perf_counter()
    query = state.get("query", "")
    seeds = _entity_link_query(query, runtime=runtime, limit=10)
    requested = str(state.get("route") or "").strip().lower()
    route = requested if requested in {"hybrid", "vector_only"} else "hybrid"
    debug = _ensure_debug(state)
    debug["route"] = route
    debug.setdefault("entity_linking", {})
    debug["entity_linking"]["seed_entities"] = list(seeds)
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


def _get_entity_evidence_ids(runtime: Runtime, entity_id: str) -> List[str]:
    """Return evidence chunk ids attached to an entity (if available)."""

    ent = dict(getattr(runtime, "entity_index", {}) or {})
    by_id = dict(ent.get("by_id", {}) or {})
    rec = dict(by_id.get(entity_id, {}) or {})
    out: List[str] = []
    for cid in list(rec.get("evidence_chunk_ids", []) or []):
        s = str(cid or "").strip()
        if s:
            out.append(s)
    return out


def _graph_neighborhood_nodes(graph: nx.MultiDiGraph, *, seeds: List[str], k: int) -> Set[str]:
    """Return nodes within k hops (undirected) from seeds."""

    if k <= 0:
        return set(seeds)
    undirected = graph.to_undirected()
    nodes: Set[str] = set()
    for s in seeds:
        if s not in undirected:
            continue
        try:
            lengths = nx.single_source_shortest_path_length(undirected, s, cutoff=k)
        except Exception:
            continue
        nodes.update(lengths.keys())
    return nodes


def _edge_evidence_ids(graph: nx.MultiDiGraph, u: str, v: str) -> List[str]:
    """Collect evidence chunk ids from the best edge attrs between u->v."""

    attrs = _best_edge_attrs(graph, u, v)
    return [str(x) for x in list(attrs.get("evidence_chunk_ids", []) or []) if str(x)]


def retrieve_graph_evidence(
    runtime: Runtime,
    *,
    seeds: List[str],
    k_hop: int,
    max_edge_evidence_chunks: int = 12,
    max_node_evidence_chunks: int = 8,
    max_community_reports: int = 2,
) -> List[Dict[str, Any]]:
    """
    Graph-driven evidence retrieval.

    - Node evidence: entity index evidence for seed entities (e.g., course rows)
    - Edge evidence: evidence chunk ids attached to edges within the k-hop neighborhood
    """

    seeds_in_graph = [s for s in seeds if s in runtime.graph]
    if not seeds_in_graph:
        return []

    hits: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    # 0) Community reports for "global" evidence (GraphRAG-style).
    comm = dict(getattr(runtime, "community_index", {}) or {})
    node_to = dict(comm.get("node_to_communities", {}) or {})
    if node_to and int(max_community_reports) > 0:
        counts: Dict[str, int] = {}
        for s in seeds_in_graph:
            for cid in list(node_to.get(s, []) or []):
                c = str(cid or "").strip()
                if not c:
                    continue
                counts[c] = int(counts.get(c, 0)) + 1
        ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        for cid, cnt in ranked[: int(max_community_reports)]:
            if cid in seen:
                continue
            seen.add(cid)
            hits.append(
                {
                    "chunk_id": cid,
                    "kind": "community",
                    "graph_score": 1.5 + float(cnt),
                }
            )

    # 1) Node evidence for seed entities.
    node_budget = int(max_node_evidence_chunks)
    for s in seeds_in_graph:
        for cid in _get_entity_evidence_ids(runtime, s):
            if cid in seen:
                continue
            seen.add(cid)
            hits.append(
                {
                    "chunk_id": cid,
                    "kind": "node",
                    "entity_id": s,
                    "graph_score": 2.0,
                }
            )
            node_budget -= 1
            if node_budget <= 0:
                break
        if node_budget <= 0:
            break

    # 2) Edge evidence within k-hop neighborhood.
    edge_budget = int(max_edge_evidence_chunks)
    nodes = _graph_neighborhood_nodes(runtime.graph, seeds=seeds_in_graph, k=int(k_hop))
    if nodes:
        candidates: List[Tuple[float, str, str, str]] = []  # score, cid, u, v
        for u, v, k in runtime.graph.edges(keys=True):
            if u not in nodes or v not in nodes:
                continue
            if k != "PREREQUISITE_OF":
                continue
            try:
                attrs = runtime.graph.get_edge_data(u, v, k) or {}
            except Exception:
                attrs = {}
            conf = float((attrs or {}).get("confidence", 0.0) or 0.0)
            evidence = list((attrs or {}).get("evidence_chunk_ids", []) or [])
            if not evidence:
                continue
            # Prioritize edges adjacent to seeds and with higher confidence.
            seed_adj = 1.0 if (u in seeds_in_graph or v in seeds_in_graph) else 0.0
            base = 1.0 + 0.5 * seed_adj + 0.1 * conf + 0.01 * len(evidence)
            for cid in evidence:
                cid_s = str(cid or "").strip()
                if not cid_s:
                    continue
                candidates.append((base, cid_s, u, v))

        candidates.sort(key=lambda x: x[0], reverse=True)
        for score, cid, u, v in candidates:
            if edge_budget <= 0:
                break
            if cid in seen:
                continue
            seen.add(cid)
            hits.append(
                {
                    "chunk_id": cid,
                    "kind": "edge",
                    "edge": {"source": u, "target": v, "relation": "PREREQUISITE_OF"},
                    "graph_score": float(score),
                }
            )
            edge_budget -= 1

    # 3) Structural neighborhood evidence for PDF/doc chunk seeds.
    # If seeds are chunk ids, pull a few neighboring chunk ids via the DOC/PAGE/CHUNK/NEXT
    # structure edges. This improves recall for broad PDF questions.
    chunk_seeds = [s for s in seeds_in_graph if s in runtime.chunk_store]
    if chunk_seeds:
        try:
            seed_meta = dict((runtime.chunk_store.get(chunk_seeds[0], {}) or {}).get("metadata", {}) or {})
            seed_source = str(seed_meta.get("source") or "").strip()
            seed_page = int(seed_meta.get("page_or_row") or seed_meta.get("page") or 0)
        except Exception:
            seed_source = ""
            seed_page = 0

        neigh_k = min(2, max(1, int(k_hop)))
        neigh_nodes = _graph_neighborhood_nodes(runtime.graph, seeds=chunk_seeds, k=neigh_k)
        cand_chunks: List[Tuple[int, str]] = []
        for n in neigh_nodes:
            if n in chunk_seeds:
                continue
            if n not in runtime.chunk_store:
                continue
            meta = dict((runtime.chunk_store.get(n, {}) or {}).get("metadata", {}) or {})
            src = str(meta.get("source") or "").strip()
            if seed_source and src and src != seed_source:
                continue
            try:
                p = int(meta.get("page_or_row") or meta.get("page") or 0)
            except Exception:
                p = 0
            cand_chunks.append((abs(p - seed_page), str(n)))
        cand_chunks.sort(key=lambda x: (x[0], x[1]))

        struct_budget = 8
        for _dist, cid in cand_chunks:
            if struct_budget <= 0:
                break
            if cid in seen:
                continue
            seen.add(cid)
            hits.append({"chunk_id": cid, "kind": "struct", "graph_score": 1.2})
            struct_budget -= 1

    return hits


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
        # Optimization: compute all shortest paths once per seed instead of calling
        # nx.shortest_path repeatedly.
        try:
            # Paths in the reversed graph: seed -> ancestor. Reverse each path to get
            # ancestor -> seed in the original graph direction.
            rev_paths = nx.single_source_shortest_path(rev, s, cutoff=k_hop)
        except Exception:
            rev_paths = {}
        for a, path_rev in (rev_paths or {}).items():
            if a == s:
                continue
            nodes = list(reversed(path_rev))
            key = tuple(nodes)
            if key in seen_paths:
                continue
            steps = _path_to_steps(graph, nodes)
            score = _score_path(steps)
            seen_paths.add(key)
            candidates.append((score, key, steps))

        # Forward chain candidates (seed -> advanced courses).
        try:
            fwd_paths = nx.single_source_shortest_path(graph, s, cutoff=k_hop)
        except Exception:
            fwd_paths = {}
        for b, nodes in (fwd_paths or {}).items():
            if b == s:
                continue
            key = tuple(nodes)
            if key in seen_paths:
                continue
            steps = _path_to_steps(graph, list(nodes))
            score = _score_path(steps)
            seen_paths.add(key)
            candidates.append((score, key, steps))

    # Between seed pairs.
    # Keep this conservative: seeds are few, so the overhead is bounded. Add a small
    # cache to avoid recomputing the same pair twice.
    sp_cache: Dict[Tuple[str, str], List[str]] = {}
    for a, b in itertools.combinations(seeds_in_graph, 2):
        for src, dst in ((a, b), (b, a)):
            try:
                key2 = (src, dst)
                nodes = sp_cache.get(key2)
                if nodes is None:
                    nodes = nx.shortest_path(graph, src, dst)
                    sp_cache[key2] = nodes
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


def retrieve_node(state: GraphRAGState, *, runtime: Runtime, settings: Settings) -> GraphRAGState:
    """Retrieve vector hits and graph paths."""

    t0 = time.perf_counter()
    query = state.get("query", "")
    top_k = int(state.get("top_k", 5))
    k_hop = int(state.get("k_hop", 2))
    route = str(state.get("route") or "hybrid").strip().lower()

    retrieval_query, rewritten = _rewrite_query_for_retrieval(query, settings=settings)
    candidate_k = int(min(50, max(top_k, top_k * 6)))

    pairs = similarity_search_with_score(runtime.vectorstore, retrieval_query, k=candidate_k)

    # Collect hits into a dict keyed by chunk_id so we can merge multiple retrieval passes.
    by_chunk: Dict[str, Dict[str, Any]] = {}

    keyword_rerank_on = _bool_env("ENABLE_KEYWORD_RERANK", True)
    kw_weights = _keyword_weights_for_query(query, retrieval_query) if keyword_rerank_on else {}

    def _upsert_hit(doc: Any, dist: float, *, source_query: str) -> None:
        meta = dict(getattr(doc, "metadata", {}) or {})
        chunk_id = str(meta.get("chunk_id") or meta.get("id") or "").strip()
        if not chunk_id:
            return
        text, stored_meta = _resolve_chunk(runtime, chunk_id)
        if not stored_meta:
            stored_meta = meta
        if not text:
            text = str(getattr(doc, "page_content", "") or "")
        base_vscore = _vector_score(float(dist))
        kw_score = _keyword_score(text, kw_weights) if keyword_rerank_on else 0.0

        # Combine semantic similarity with a lightweight keyword match score.
        # This helps for cross-lingual queries where the embedding match may be weak.
        combined = base_vscore + 0.65 * kw_score if keyword_rerank_on else base_vscore

        # Heuristic penalty for very early PDF pages when the query is about absences/sanctions.
        try:
            por = stored_meta.get("page_or_row")
            page = int(por) if por is not None and str(por).strip() else None
        except Exception:
            page = None
        if page is not None and page <= 3 and any(k in kw_weights for k in ("absence", "sanction", "non-assiduit")):
            combined *= 0.85

        hit = {
            "chunk_id": chunk_id,
            "distance": float(dist),
            "vector_score": float(combined),
            "vector_score_base": float(base_vscore),
            "keyword_score": float(kw_score),
            "text": text,
            "metadata": stored_meta,
            "retrieved_by": [str(source_query or "").strip() or "query"],
        }

        prev = by_chunk.get(chunk_id)
        if prev is None:
            by_chunk[chunk_id] = hit
            return

        # Merge provenance and keep the strongest score.
        try:
            rb = set([str(x) for x in list(prev.get("retrieved_by", []) or []) if str(x)])
            rb.add(str(source_query or "").strip() or "query")
            prev["retrieved_by"] = sorted(rb)
        except Exception:
            prev["retrieved_by"] = list(dict.fromkeys((prev.get("retrieved_by", []) or []) + [source_query]))

        try:
            prev_dist = float(prev.get("distance")) if prev.get("distance") is not None else float("inf")
            prev["distance"] = float(min(prev_dist, float(dist)))
        except Exception:
            pass

        if float(hit.get("vector_score", 0.0)) > float(prev.get("vector_score", 0.0)):
            by_chunk[chunk_id] = hit

    for doc, dist in pairs:
        _upsert_hit(doc, float(dist), source_query=retrieval_query)

    # Optional auto-expand: if the first pass looks weak, run 1-2 fallback queries and merge.
    auto_expand_on = _bool_env("ENABLE_AUTO_EXPAND_RETRIEVAL", True)
    initial_hits = list(by_chunk.values())
    initial_hits.sort(key=lambda x: float(x.get("vector_score", 0.0)), reverse=True)
    probe = initial_hits[: max(1, min(8, len(initial_hits)))]
    top_base = max([float(h.get("vector_score_base") or 0.0) for h in probe], default=0.0)
    top_kw = max([float(h.get("keyword_score") or 0.0) for h in probe], default=0.0)

    # Heuristic: treat retrieval as weak when top semantic similarity is low AND keyword
    # coverage is also low (when applicable). This avoids unnecessary extra searches.
    weak_retrieval = False
    if not initial_hits:
        weak_retrieval = True
    elif top_base < 0.23:
        weak_retrieval = True
    elif keyword_rerank_on and kw_weights and top_kw < 0.08 and top_base < 0.40:
        weak_retrieval = True

    fallback_queries: List[str] = []
    fallback_candidate_k = None
    if auto_expand_on and weak_retrieval:
        fallback_candidate_k = int(min(120, max(candidate_k, top_k * 12)))
        candidates: List[str] = []
        if rewritten and query.strip():
            candidates.append(query.strip())
        kw_query = _top_keywords_query(kw_weights, limit=8) if kw_weights else ""
        if kw_query:
            candidates.append(kw_query)

        seen_q: Set[str] = set([str(retrieval_query or "").strip().lower()])
        for cq in candidates:
            cq2 = str(cq or "").strip()
            if not cq2:
                continue
            key = cq2.lower()
            if key in seen_q:
                continue
            seen_q.add(key)
            fallback_queries.append(cq2)

        for fq in fallback_queries[:2]:
            pairs2 = similarity_search_with_score(runtime.vectorstore, fq, k=int(fallback_candidate_k))
            for doc, dist in pairs2:
                _upsert_hit(doc, float(dist), source_query=fq)

    vector_hits: List[Dict[str, Any]] = list(by_chunk.values())

    # Re-rank and keep a slightly larger pool than top_k.
    vector_hits.sort(key=lambda x: float(x.get("vector_score", 0.0)), reverse=True)
    keep_mult = 6 if (auto_expand_on and bool(fallback_queries)) else 4
    keep_k = int(min(len(vector_hits), max(top_k, top_k * keep_mult)))
    vector_hits = vector_hits[:keep_k]

    # Optional cross-encoder rerank for higher precision (quality-first).
    ce_on = _bool_env("ENABLE_CE_RERANK", False)
    ce_debug: Dict[str, Any] = {"enabled": False}
    if ce_on and vector_hits:
        try:
            ce_model = (os.environ.get("CE_RERANK_MODEL") or "").strip() or "BAAI/bge-reranker-v2-m3"
            ce_top_n = _int_env("CE_RERANK_TOP_N", 80)
            ce_top_n = max(1, min(int(ce_top_n), 200))
            ce_device = _resolve_rerank_device(os.environ.get("CE_RERANK_DEVICE"))
            alt_q = retrieval_query if bool(rewritten) else None
            vector_hits, ce_debug = _cross_encoder_rerank(
                vector_hits,
                query=query,
                alt_query=alt_q,
                model_name=ce_model,
                device=ce_device,
                top_n=int(ce_top_n),
            )
        except Exception as exc:
            # Degrade gracefully to keep the system usable offline/without downloads.
            ce_debug = {"enabled": True, "error": f"{type(exc).__name__}: {exc}"}

    # If no explicit entities in the query, bootstrap seeds from retrieved docs.
    seeds = list(state.get("seed_entities", []) or [])
    if not seeds:
        # 1) Try course-code extraction from retrieved text.
        for h in vector_hits[: max(1, min(3, len(vector_hits)))]:
            seeds.extend(extract_course_codes(h.get("text", "")))
        seeds = extract_course_codes(" ".join(seeds))

        # 2) If still empty (common for PDF policy documents), fall back to using the
        # top retrieved chunk ids as graph seeds. This enables DOC/PAGE/CHUNK graph paths.
        if not seeds:
            for h in vector_hits[: max(1, min(2, len(vector_hits)))]:
                cid = str(h.get("chunk_id") or "").strip()
                if cid:
                    seeds.append(cid)

    graph_paths: List[Dict[str, Any]] = []
    if route != "vector_only":
        graph_paths = retrieve_graph_paths(
            runtime.graph, seeds=seeds, k_hop=k_hop, max_paths=5
        )

    graph_hits: List[Dict[str, Any]] = []
    if route != "vector_only":
        graph_hits = retrieve_graph_evidence(runtime, seeds=seeds, k_hop=k_hop)

    debug = _ensure_debug(state)
    debug["latency_ms"]["retrieve"] = _ms(t0)
    debug["evidence_counts"]["vector_hits"] = len(vector_hits)
    debug["evidence_counts"]["graph_paths"] = len(graph_paths)
    debug["evidence_counts"]["graph_hits"] = len(graph_hits)
    debug.setdefault("retrieval", {})
    debug["retrieval"]["candidate_k"] = int(candidate_k)
    debug["retrieval"]["keep_k"] = int(keep_k)
    debug["retrieval"]["retrieval_query"] = retrieval_query
    debug["retrieval"]["query_rewritten"] = bool(rewritten)
    debug["retrieval"]["keyword_rerank"] = bool(keyword_rerank_on)
    debug["retrieval"]["auto_expand"] = bool(auto_expand_on)
    debug["retrieval"]["weak_retrieval"] = bool(weak_retrieval)
    debug["retrieval"]["top_base_vscore"] = float(top_base)
    debug["retrieval"]["top_keyword_score"] = float(top_kw)
    debug["retrieval"]["fallback_queries"] = list(fallback_queries)
    if fallback_candidate_k is not None:
        debug["retrieval"]["fallback_candidate_k"] = int(fallback_candidate_k)
    debug["retrieval"]["ce_rerank"] = dict(ce_debug)

    return {
        **state,
        "seed_entities": seeds,
        "retrieval_query": retrieval_query,
        "vector_hits": vector_hits,
        "graph_paths": graph_paths,
        "graph_hits": graph_hits,
        "debug": debug,
    }


def fuse_node(state: GraphRAGState, *, runtime: Runtime) -> GraphRAGState:
    """Fuse vector hits and graph evidence into a small evidence pack."""

    t0 = time.perf_counter()
    brief = bool(state.get("brief", True))
    vector_hits = list(state.get("vector_hits", []) or [])
    graph_paths = list(state.get("graph_paths", []) or [])
    graph_hits = list(state.get("graph_hits", []) or [])

    graph_evidence_ids: Set[str] = set()
    for p in graph_paths:
        for step in p.get("path", []) or []:
            for cid in step.get("evidence_chunk_ids", []) or []:
                graph_evidence_ids.add(str(cid))
    for h in graph_hits:
        cid = str(h.get("chunk_id") or "").strip()
        if cid:
            graph_evidence_ids.add(cid)

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
    # For report-style answers, provide more evidence when brief=False.
    evidence_limit = 8 if brief else 12
    evidence_pack = fused[: int(evidence_limit)]

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


