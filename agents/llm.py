"""
LLM wrapper utilities.

This module provides:
- An optional OpenAI chat model (when ``OPENAI_API_KEY`` is configured)
- An optional local Ollama chat model (when ``LLM_PROVIDER=ollama``)
- A deterministic offline fallback synthesizer (no external API)
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List

from app.settings import Settings


_CITATIONS_LINE_RE = re.compile(r"^\s*Citations\s*:\s*(?P<ids>.*)\s*$", re.IGNORECASE)
_BRACKET_RE = re.compile(r"\[[^\[\]]+\]")


def _report_style_system_prompt(*, brief: bool) -> str:
    """
    System prompt used for both OpenAI and Ollama answer generation.

    Goal: produce longer, report-like answers (when brief=False) while staying strictly
    grounded in the provided evidence.
    """

    detail = "concise" if brief else "detailed"
    length_hint = (
        "Aim for ~6-10 bullet points total and 2-4 short paragraphs."
        if brief
        else "Aim for ~250-500 words and use multiple short sections."
    )

    return (
        "You are a careful assistant answering questions about academic regulations (PDF).\n"
        "Output language: English.\n\n"
        "Grounding rules:\n"
        "- Answer strictly based on the provided EVIDENCE.\n"
        "- Do not assume facts that are not explicitly stated.\n"
        "- If the evidence is insufficient, say what is missing and what would be needed.\n"
        "- Use citations with chunk ids in square brackets, e.g. [pdf:...::p20::c4].\n"
        "- Only cite chunk ids that appear in the EVIDENCE block; never invent ids.\n"
        "- At the end, add a final line: 'Citations: <comma-separated chunk ids>'.\n\n"
        "Write a report-style answer with these sections (use headings):\n"
        "1) Direct answer\n"
        "2) Key rules / clauses\n"
        "3) Thresholds & consequences (use a small table if numbers/thresholds appear)\n"
        "4) Practical interpretation (what this means in practice)\n"
        "5) Not specified / limitations\n\n"
        f"Style: {detail}. {length_hint}\n"
        "- Prefer short sentences and clear bullet points.\n"
        "- When listing consequences, keep them concrete and tied to citations.\n"
    )


def _allowed_citation_ids(evidence_pack: List[Dict[str, Any]]) -> List[str]:
    """Return allowed chunk ids in stable order (evidence_pack order)."""

    out: List[str] = []
    seen: set[str] = set()
    for ev in evidence_pack or []:
        cid = str(ev.get("chunk_id") or "").strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
    return out


def _strip_trailing_citations_lines(text: str) -> str:
    """Remove trailing 'Citations: ...' line(s) if present."""

    lines = (text or "").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and _CITATIONS_LINE_RE.match(lines[-1]):
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
    return "\n".join(lines).rstrip()


def _looks_like_chunk_id(token: str) -> bool:
    """
    Heuristic: chunk ids in this project contain ':' (md:/csv:/pdf:/community:).
    Avoid touching brackets like [none] or [1].
    """

    t = str(token or "").strip()
    if not t:
        return False
    if ":" not in t:
        return False
    # Very permissive; keep it robust for custom chunk id schemes.
    return True


def _sanitize_inline_citations(text: str, *, allowed: set[str]) -> str:
    """
    Remove/trim bracket citations that reference chunk ids not present in `allowed`.
    """

    if not text:
        return ""
    if not allowed:
        return _BRACKET_RE.sub("", text)

    def _repl(m: re.Match[str]) -> str:
        inner = m.group(0)[1:-1].strip()
        if not inner:
            return m.group(0)
        # Split on commas/whitespace to support "[id1, id2]" or "[id1 id2]".
        toks = [t for t in re.split(r"[,\s]+", inner) if t]
        if not toks:
            return m.group(0)

        # Only treat as a citation bracket if it contains something that looks like a chunk id.
        if not any(_looks_like_chunk_id(t) for t in toks):
            return m.group(0)

        kept = [t for t in toks if t in allowed]
        if not kept:
            return ""
        if len(kept) == 1:
            return f"[{kept[0]}]"
        return "[" + ", ".join(kept) + "]"

    out = _BRACKET_RE.sub(_repl, text)
    # Clean up leftover double spaces from removed brackets.
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _enforce_citations(answer: str, *, evidence_pack: List[Dict[str, Any]]) -> str:
    """
    Enforce that citations in `answer` reference only chunk ids present in `evidence_pack`.

    Strategy:
    - Strip any trailing 'Citations:' lines produced by the model.
    - Remove or trim inline bracket citations that reference non-allowed chunk ids.
    - Append a canonical 'Citations:' line with allowed ids.
    """

    allowed_list = _allowed_citation_ids(evidence_pack)
    allowed_set = set(allowed_list)
    base = _strip_trailing_citations_lines(answer or "")
    base = _sanitize_inline_citations(base, allowed=allowed_set)
    if not allowed_list:
        return base
    return (base + "\n\n" + "Citations: " + ", ".join(allowed_list)).strip()


def llm_mode(settings: Settings) -> str:
    """Return 'openai', 'ollama', or 'offline' based on configuration."""

    if settings.llm_provider.lower() == "openai" and bool(settings.openai_api_key):
        return "openai"
    if settings.llm_provider.lower() == "ollama":
        return "ollama"
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
        content=_report_style_system_prompt(brief=bool(brief))
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


def _ollama_answer(
    *, query: str, brief: bool, evidence_pack: List[Dict[str, Any]], graph_paths: List[Dict[str, Any]], settings: Settings
) -> str:
    """Generate an answer using a local Ollama model (requires a running Ollama server)."""

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        try:
            from langchain_ollama import ChatOllama  # type: ignore
        except Exception:  # pragma: no cover
            # Fallback for older environments.
            from langchain_community.chat_models.ollama import ChatOllama  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Ollama dependencies are not available.") from exc

    model_name = (os.environ.get("OLLAMA_MODEL") or "").strip() or settings.graph_llm_model or "mistral:7b"
    base_url = (os.environ.get("OLLAMA_BASE_URL") or "").strip() or settings.ollama_base_url

    def _float_env(name: str, default: float) -> float:
        try:
            return float((os.environ.get(name) or "").strip() or default)
        except Exception:
            return default

    def _int_env(name: str, default: int) -> int:
        try:
            v = int((os.environ.get(name) or "").strip() or default)
            return v if v > 0 else default
        except Exception:
            return default

    temperature = _float_env("OLLAMA_TEMPERATURE", float(settings.graph_llm_temperature))
    # If the user explicitly sets OLLAMA_NUM_PREDICT, honor it. Otherwise, increase
    # the default budget for non-brief, report-style answers.
    raw_np = (os.environ.get("OLLAMA_NUM_PREDICT") or "").strip()
    if raw_np:
        num_predict = _int_env("OLLAMA_NUM_PREDICT", int(settings.graph_llm_num_predict))
    else:
        base_np = int(settings.graph_llm_num_predict)
        num_predict = max(base_np, 512) if not bool(brief) else base_np
    timeout_s = _int_env("OLLAMA_TIMEOUT_S", int(settings.graph_llm_timeout_s))

    llm = ChatOllama(
        base_url=base_url,
        model=model_name,
        temperature=float(temperature),
        num_predict=int(num_predict),
        client_kwargs={"timeout": float(timeout_s)},
    )

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
        content=_report_style_system_prompt(brief=bool(brief))
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
    """Generate an answer in offline mode, optionally enhanced with OpenAI or Ollama."""

    mode = llm_mode(settings)
    if mode == "openai":
        try:
            ans = _openai_answer(
                query=query,
                brief=brief,
                evidence_pack=evidence_pack,
                graph_paths=graph_paths,
                settings=settings,
            )
            return _enforce_citations(ans, evidence_pack=evidence_pack)
        except Exception:
            ans = _offline_answer(
                query=query, brief=brief, evidence_pack=evidence_pack, graph_paths=graph_paths
            )
            return _enforce_citations(ans, evidence_pack=evidence_pack)

    if mode == "ollama":
        try:
            ans = _ollama_answer(
                query=query,
                brief=brief,
                evidence_pack=evidence_pack,
                graph_paths=graph_paths,
                settings=settings,
            )
            return _enforce_citations(ans, evidence_pack=evidence_pack)
        except Exception:
            ans = _offline_answer(
                query=query, brief=brief, evidence_pack=evidence_pack, graph_paths=graph_paths
            )
            return _enforce_citations(ans, evidence_pack=evidence_pack)

    ans = _offline_answer(query=query, brief=brief, evidence_pack=evidence_pack, graph_paths=graph_paths)
    return _enforce_citations(ans, evidence_pack=evidence_pack)


