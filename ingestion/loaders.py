"""
Source loaders.

Requirements for this MVP:
- Use LangChain ``Document`` (newest import preferred, fallback supported).
- Markdown: load as section-aware Documents with metadata including ``source``,
  ``doc_id`` and ``title``. A lightweight heading tracker populates ``section``.
- CSV: one Document per row with metadata including ``source``, ``row``,
  ``course_code``, ``year``, ``program`` and ``prerequisite``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

try:  # LangChain modern import
    from langchain_core.documents import Document
except Exception:  # pragma: no cover
    from langchain.schema import Document  # type: ignore


_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")


def scan_source_files(data_dir: Path) -> List[Path]:
    """Scan a directory for supported source files (Markdown/CSV)."""

    files: List[Path] = []
    for pattern in ("*.md", "*.csv"):
        files.extend(data_dir.rglob(pattern))
    return sorted([p for p in files if p.is_file()])


def _rel_source(path: Path, data_dir: Path) -> str:
    """Return a stable source identifier relative to the data_dir."""

    try:
        rel = path.relative_to(data_dir)
        return rel.as_posix()
    except Exception:
        return path.as_posix()


def _doc_id(prefix: str, source: str, suffix: str = "") -> str:
    """Create a deterministic doc_id safe to embed into chunk_id."""

    base = f"{prefix}:{source}".replace(" ", "_")
    return f"{base}{suffix}"


def _markdown_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m and len(m.group("hashes")) == 1:
            return m.group("title").strip()
    return fallback


def split_markdown_by_headings(text: str, *, title: str) -> List[Tuple[str, str]]:
    """
    Split markdown text into section chunks using a lightweight heading tracker.

    Returns a list of (section, section_text) pairs.
    """

    sections: List[Tuple[str, str]] = []
    h1: str | None = None
    h2: str | None = None
    current_section: str | None = None
    buf: List[str] = []

    def _flush() -> None:
        nonlocal buf, current_section
        content = "\n".join(buf).strip()
        if content:
            sections.append(((current_section or title).strip(), content))
        buf = []

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        m = _HEADING_RE.match(line.strip())
        if m:
            level = len(m.group("hashes"))
            heading = m.group("title").strip()

            if level == 1:
                _flush()
                h1, h2 = heading, None
            elif level == 2:
                _flush()
                h2 = heading
            else:
                # Do not create new Documents for deep headings; keep them in the same section.
                pass

            parts = [p for p in (h1, h2) if p]
            current_section = " > ".join(parts) if parts else heading

        buf.append(line)

    _flush()
    return sections or [(title, text)]


def load_markdown_file(path: Path, *, data_dir: Path) -> List[Document]:
    """Load a Markdown file into section-aware Documents."""

    text = path.read_text(encoding="utf-8")
    source = _rel_source(path, data_dir)
    title = _markdown_title(text, fallback=path.stem)

    docs: List[Document] = []
    for i, (section, section_text) in enumerate(split_markdown_by_headings(text, title=title)):
        doc_id = _doc_id("md", source, suffix=f"::s{i}")
        docs.append(
            Document(
                page_content=section_text,
                metadata={
                    "type": "markdown",
                    "source": source,
                    "doc_id": doc_id,
                    "title": title,
                    "section": section,
                },
            )
        )
    return docs


def load_courses_csv(path: Path, *, data_dir: Path) -> List[Document]:
    """Load a course CSV file into one Document per course row."""

    df = pd.read_csv(path)
    source = _rel_source(path, data_dir)
    docs: List[Document] = []
    for i, row in enumerate(df.itertuples(index=False), start=1):
        course_code = str(getattr(row, "course_code"))
        course_name = str(getattr(row, "course_name"))
        program = str(getattr(row, "program"))
        year = int(getattr(row, "year"))
        prerequisite = getattr(row, "prerequisite", "")
        prerequisite_str = "" if pd.isna(prerequisite) else str(prerequisite)
        doc_id = _doc_id("csv", source, suffix=f"::row{i}::{course_code}")

        content = (
            f"Course: {course_code} - {course_name}\n"
            f"Program: {program}\n"
            f"Year: {year}\n"
            f"Prerequisite: {prerequisite_str or 'None'}\n"
        )
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "type": "course_row",
                    "source": source,
                    "doc_id": doc_id,
                    "title": course_code,
                    "row": i,
                    "page_or_row": i,
                    "course_code": course_code,
                    "course_name": course_name,
                    "program": program,
                    "year": year,
                    "prerequisite": prerequisite_str,
                },
            )
        )
    return docs


def load_sources(data_dir: Path, *, selected_sources: Iterable[Path] | None = None) -> List[Document]:
    """Load all supported source files from a directory."""

    docs: List[Document] = []

    files = list(selected_sources) if selected_sources is not None else scan_source_files(data_dir)

    for p in files:
        if p.suffix.lower() == ".md":
            docs.extend(load_markdown_file(p, data_dir=data_dir))
        elif p.suffix.lower() == ".csv":
            docs.extend(load_courses_csv(p, data_dir=data_dir))

    return docs


