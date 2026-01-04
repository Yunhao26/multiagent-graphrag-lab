"""Small utility helpers for the ingestion and serving layers."""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_dotenv_if_present(dotenv_path: str | Path = ".env") -> None:
    """Load environment variables from a local .env file if it exists."""

    try:
        from dotenv import load_dotenv
    except Exception:
        return

    path = Path(dotenv_path)
    if path.exists():
        load_dotenv(dotenv_path=str(path), override=False)


def ensure_dir(path: Path) -> None:
    """Create a directory (including parents) if it does not exist."""

    path.mkdir(parents=True, exist_ok=True)


def gzip_json_dump(obj: Any, path: Path) -> None:
    """Write JSON to a gzip-compressed file."""

    ensure_dir(path.parent)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def gzip_json_load(path: Path) -> Any:
    """Read JSON from a gzip-compressed file."""

    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    """Write JSON lines to disk."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSON lines from disk."""

    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_json(obj: Any, path: Path) -> None:
    """Write pretty JSON to disk."""

    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def now_iso_utc() -> str:
    """Return the current timestamp in ISO 8601 format (UTC)."""

    return datetime.now(timezone.utc).isoformat()


