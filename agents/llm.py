"""
LLM wrapper utilities.

This module provides:
- An optional OpenAI chat model (when ``OPENAI_API_KEY`` is configured)
- A deterministic offline fallback synthesizer (no external API)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from app.settings import Settings

def llm_mode(settings: Settings) -> str:
    """Return 'openai' if configured, otherwise 'offline'."""

    if settings.llm_provider.lower() == "openai" and bool(settings.openai_api_key):
        return "openai"
    return "offline"


def _format_graph_paths(graph_paths: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    lines: List[str] = []
    for p in graph_paths[:limit]:
        steps = p.get("path", []) or []
        if not steps:
            continue
        chain = [steps[0].get("source")]
        for s in steps:
            chain.append(s.get("target"))
        chain_str = " -> ".join([c for c in chain if c])
        lines.append(f"- {chain_str}")
    return lines


def _offline_answer(
    *, query: str, brief: bool, evidence_pack: List[Dict[str, Any]], graph_paths: List[Dict[str, Any]]
) -> str:
    """Deterministic, offline extractive summary."""

    lines: List[str] = []

    if graph_paths:
        lines.append("Prerequisite paths (graph):")
        lines.extend(_format_graph_paths(graph_paths, limit=3))
        lines.append("")

    # Extract course facts from evidence metadata when available.
    facts: List[str] = []
    for ev in evidence_pack:
        meta = dict(ev.get("metadata", {}) or {})
        cc = meta.get("course_code")
        year = meta.get("year")
        program = meta.get("program")
        if cc:
            parts = [str(cc)]
            if year:
                parts.append(f"year={year}")
            if program:
                parts.append(f"program={program}")
            facts.append("- " + ", ".join(parts))
    facts = list(dict.fromkeys(facts))  # stable dedupe

    if facts:
        lines.append("Course facts (evidence metadata):")
        lines.extend(facts[: (4 if brief else 8)])
        lines.append("")

    lines.append("Evidence highlights:")
    for ev in evidence_pack[: (4 if brief else 8)]:
        cid = str(ev.get("chunk_id") or "")
        text = str(ev.get("text") or "").strip().replace("\n", " ")
        if len(text) > 180:
            text = text[:180] + " ...[truncated]"
        lines.append(f"- [{cid}] {text}")

    cited = ", ".join([str(ev.get("chunk_id")) for ev in evidence_pack if ev.get("chunk_id")])
    if cited:
        lines.append("")
        lines.append(f"Citations: {cited}")

    return "\n".join(lines).strip()


def _openai_answer(
    *, query: str, brief: bool, evidence_pack: List[Dict[str, Any]], graph_paths: List[Dict[str, Any]], settings: Settings
) -> str:
    """Generate an answer using ChatOpenAI (requires OPENAI_API_KEY)."""

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("OpenAI dependencies are not available.") from exc

    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    llm = ChatOpenAI(model=model_name, api_key=settings.openai_api_key)

    evidence_lines: List[str] = []
    for ev in evidence_pack:
        cid = str(ev.get("chunk_id") or "")
        text = str(ev.get("text") or "").strip()
        if len(text) > 800:
            text = text[:800] + " ...[truncated]"
        evidence_lines.append(f"[{cid}]\n{text}")
    evidence_block = "\n\n".join(evidence_lines).strip()

    graph_block = "\n".join(_format_graph_paths(graph_paths, limit=5)).strip()

    system = SystemMessage(
        content=(
            "You are a careful assistant for a course handbook.\n"
            "- Answer strictly based on the provided EVIDENCE.\n"
            "- If evidence is insufficient, say what is missing.\n"
            "- Always include citations using chunk ids in square brackets, e.g. [md:...::c0].\n"
            "- At the end, add a final line: 'Citations: <comma-separated chunk ids>'.\n"
        )
    )

    human = HumanMessage(
        content=(
            f"Query: {query}\n"
            f"Brief: {brief}\n\n"
            f"Graph paths (optional):\n{graph_block or '[none]'}\n\n"
            f"EVIDENCE:\n{evidence_block or '[none]'}\n\n"
            "Answer:"
        )
    )
    msg = llm.invoke([system, human])
    return getattr(msg, "content", str(msg)).strip()


def generate_answer(
    *,
    query: str,
    brief: bool,
    evidence_pack: List[Dict[str, Any]],
    graph_paths: List[Dict[str, Any]],
    settings: Settings,
) -> str:
    """Generate an answer in offline mode, optionally enhanced with OpenAI."""

    if llm_mode(settings) == "openai":
        try:
            return _openai_answer(
                query=query,
                brief=brief,
                evidence_pack=evidence_pack,
                graph_paths=graph_paths,
                settings=settings,
            )
        except Exception:
            return _offline_answer(
                query=query, brief=brief, evidence_pack=evidence_pack, graph_paths=graph_paths
            )

    return _offline_answer(query=query, brief=brief, evidence_pack=evidence_pack, graph_paths=graph_paths)


