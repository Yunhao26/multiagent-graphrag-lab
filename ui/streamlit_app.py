"""
Streamlit frontend for the GraphRAG MVP.

This UI is intentionally minimal: it calls the FastAPI backend for QA.
"""

from __future__ import annotations

import csv
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st

from ui.client import health, safe_qa


_COURSE_CODE_RE = re.compile(r"^[A-Z]{2,6}\d{3}$")


def _dot_escape(s: str) -> str:
    return (s or "").replace("\\", "\\\\").replace('"', '\\"')


def _shorten(s: str, max_len: int = 40) -> str:
    s = str(s or "")
    if len(s) <= max_len:
        return s
    return s[: max(0, max_len - 1)] + "…"


def _node_kind(node_id: str) -> str:
    nid = str(node_id or "")
    if nid.startswith("doc:"):
        return "DOC"
    if nid.startswith("page:"):
        return "PAGE"
    if nid.startswith(("pdf:", "md:", "csv:", "community:")):
        return "CHUNK"
    if _COURSE_CODE_RE.fullmatch(nid):
        return "COURSE"
    return "NODE"


def _node_label(node_id: str) -> str:
    nid = str(node_id or "")
    kind = _node_kind(nid)

    if kind == "DOC":
        src = nid[len("doc:") :]
        fname = src.replace("\\", "/").split("/")[-1]
        return _shorten(fname, 44)
    if kind == "PAGE":
        # page:<source>#p17
        try:
            src, p = nid[len("page:") :].rsplit("#p", 1)
        except Exception:
            src, p = nid, ""
        fname = src.replace("\\", "/").split("/")[-1]
        return _shorten(f"{fname} p{p}", 46)
    if kind == "COURSE":
        return nid
    if kind == "CHUNK":
        if nid.startswith("pdf:"):
            m = re.search(r"::p(\d+)::c(\d+)\s*$", nid)
            if m:
                return f"chunk p{m.group(1)} c{m.group(2)}"
        if nid.startswith("md:"):
            m = re.search(r"::s(\d+)::c(\d+)\s*$", nid)
            if m:
                return f"chunk s{m.group(1)} c{m.group(2)}"
        if nid.startswith("csv:"):
            m = re.search(r"::row(\d+)::.*::c(\d+)\s*$", nid)
            if m:
                return f"row {m.group(1)} c{m.group(2)}"
        if nid.startswith("community:"):
            return nid
        return _shorten(nid, 46)
    return _shorten(nid, 46)


def _node_style(kind: str) -> Tuple[str, str]:
    # (shape, fillcolor)
    if kind == "DOC":
        return "box", "#D7ECFF"
    if kind == "PAGE":
        return "ellipse", "#EDEDED"
    if kind == "CHUNK":
        return "note", "#FFF2CC"
    if kind == "COURSE":
        return "oval", "#FCE5CD"
    return "oval", "#FFFFFF"


def build_dot_from_graph_paths(graph_paths: List[Dict[str, Any]], *, max_paths: int = 1) -> str:
    """Build a Graphviz DOT graph from GraphRAG path steps."""

    nodes: Dict[str, Dict[str, str]] = {}
    edges: List[Tuple[str, str, str]] = []

    for p in (graph_paths or [])[: max(1, int(max_paths))]:
        for step in list(p.get("path", []) or []):
            u = str(step.get("source") or "")
            v = str(step.get("target") or "")
            rel = str(step.get("relation") or "")
            if not u or not v:
                continue
            for nid in (u, v):
                if nid in nodes:
                    continue
                kind = _node_kind(nid)
                shape, color = _node_style(kind)
                nodes[nid] = {"label": _node_label(nid), "shape": shape, "color": color}
            edges.append((u, v, rel))

    lines: List[str] = []
    lines.append("digraph G {")
    lines.append('  rankdir="LR";')
    lines.append('  graph [bgcolor="transparent"];')
    lines.append('  node [style="filled", fontname="Inter"];')
    lines.append('  edge [fontname="Inter"];')

    for nid, attrs in nodes.items():
        lines.append(
            f'  "{_dot_escape(nid)}" [label="{_dot_escape(attrs["label"])}", shape="{attrs["shape"]}", fillcolor="{attrs["color"]}"];'
        )
    for u, v, rel in edges:
        label = _dot_escape(rel)
        lines.append(f'  "{_dot_escape(u)}" -> "{_dot_escape(v)}" [label="{label}"];')

    lines.append("}")
    return "\n".join(lines)


def _append_user_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    st.set_page_config(page_title="GraphRAG MVP", layout="wide")
    st.title("GraphRAG MVP")

    default_backend = os.environ.get("BACKEND_URL", "http://localhost:8000").strip()
    index_dir = Path(os.environ.get("INDEX_DIR", "data/index"))
    users_csv = index_dir / "users.csv"

    st.sidebar.header("Settings")
    backend_url = st.sidebar.text_input("Backend URL", value=default_backend)
    brief = st.sidebar.checkbox("Brief answer", value=False)
    top_k = st.sidebar.slider("top_k (vector)", min_value=1, max_value=20, value=5)
    k_hop = st.sidebar.slider("k_hop (graph)", min_value=0, max_value=5, value=2)

    st.sidebar.divider()
    st.sidebar.header("User info")
    name = st.sidebar.text_input("Name", value="")
    email = st.sidebar.text_input("Email", value="")
    use_case = st.sidebar.text_area("Use case", value="", height=80)
    feedback = st.sidebar.text_area("Feedback", value="", height=80)
    if st.sidebar.button("Save to users.csv"):
        row = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "name": name.strip(),
            "email": email.strip(),
            "use_case": use_case.strip(),
            "feedback": feedback.strip(),
        }
        _append_user_row(users_csv, row)
        st.sidebar.success(f"Saved to {users_csv.as_posix()}")

    st.sidebar.divider()
    if st.sidebar.button("Check /health"):
        try:
            h = health(backend_url)
            st.sidebar.success("Backend is reachable.")
            st.sidebar.json(h)
        except Exception as exc:
            st.sidebar.error(str(exc))

    query = st.text_area(
        "Query",
        value="What is the prerequisite chain to DL301?",
        height=120,
    )
    if st.button("Ask", type="primary"):
        result = safe_qa(
            backend_url,
            query=query,
            brief=brief,
            top_k=top_k,
            k_hop=k_hop,
        )
        if not result.get("answer") and result.get("debug", {}).get("error"):
            st.error("Backend request failed. Please check the backend URL and that the index is built.")
            st.code(str(result.get("debug", {}).get("error")))
            return

        st.subheader("Answer")
        st.write(result.get("answer", ""))

        with st.expander("Citations", expanded=False):
            st.json(result.get("citations", []))

        graph_paths = list(result.get("graph_paths", []) or [])
        with st.expander("Graph (visual)", expanded=True):
            if not graph_paths:
                st.info("No graph paths were produced for this query.")
            else:
                max_n = min(5, len(graph_paths))
                mode = st.radio(
                    "View mode",
                    options=["Top-1 path", "Union of top-N paths"],
                    horizontal=True,
                )
                if mode == "Top-1 path":
                    idx = st.selectbox("Path", options=list(range(len(graph_paths))), index=0)
                    dot = build_dot_from_graph_paths([graph_paths[int(idx)]], max_paths=1)
                else:
                    n = st.slider("N paths", min_value=1, max_value=max_n, value=min(3, max_n))
                    dot = build_dot_from_graph_paths(graph_paths, max_paths=int(n))

                st.graphviz_chart(dot, use_container_width=True)

                # Lightweight edge table for the top path.
                top_path = graph_paths[0] if graph_paths else {}
                steps = list(top_path.get("path", []) or [])
                if steps:
                    rows: List[Dict[str, Any]] = []
                    for s in steps:
                        rows.append(
                            {
                                "source": s.get("source"),
                                "relation": s.get("relation"),
                                "target": s.get("target"),
                                "evidence_chunks": len(list(s.get("evidence_chunk_ids", []) or [])),
                            }
                        )
                    st.caption("Top path steps (table)")
                    st.dataframe(rows, use_container_width=True, hide_index=True)

        with st.expander("Graph paths (raw)", expanded=False):
            st.json(graph_paths)

        with st.expander("Debug", expanded=False):
            st.json(result.get("debug", {}))


if __name__ == "__main__":
    main()


