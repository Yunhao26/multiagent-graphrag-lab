"""
Streamlit frontend for the GraphRAG MVP.

This UI is intentionally minimal: it calls the FastAPI backend for QA.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from ui.client import health, safe_qa


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
    brief = st.sidebar.checkbox("Brief answer", value=True)
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

        with st.expander("Graph paths", expanded=False):
            st.json(result.get("graph_paths", []))

        with st.expander("Debug", expanded=False):
            st.json(result.get("debug", {}))


if __name__ == "__main__":
    main()


