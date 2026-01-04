"""A tiny HTTP client for the FastAPI backend used by Streamlit."""

from __future__ import annotations

from typing import Any, Dict

import requests


def health(backend_url: str, *, timeout_s: int = 10) -> Dict[str, Any]:
    """Call ``GET /health``."""

    url = backend_url.rstrip("/") + "/health"
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def qa(
    backend_url: str,
    *,
    query: str,
    brief: bool,
    top_k: int,
    k_hop: int,
    timeout_s: int = 60,
) -> Dict[str, Any]:
    """Call ``POST /qa``."""

    url = backend_url.rstrip("/") + "/qa"
    r = requests.post(
        url,
        json={"query": query, "brief": bool(brief), "top_k": int(top_k), "k_hop": int(k_hop)},
        timeout=timeout_s,
    )
    r.raise_for_status()
    return r.json()


def safe_qa(
    backend_url: str,
    *,
    query: str,
    brief: bool,
    top_k: int,
    k_hop: int,
    timeout_s: int = 60,
) -> Dict[str, Any]:
    """QA call that returns an error dict instead of raising."""

    try:
        return qa(
            backend_url=backend_url,
            query=query,
            brief=brief,
            top_k=top_k,
            k_hop=k_hop,
            timeout_s=timeout_s,
        )
    except Exception as exc:
        return {"answer": "", "citations": [], "graph_paths": [], "debug": {"error": str(exc)}}


