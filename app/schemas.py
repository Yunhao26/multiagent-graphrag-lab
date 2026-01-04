"""Pydantic models for the FastAPI backend."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response schema for ``GET /health``."""

    status: str = Field(default="ok")


class QARequest(BaseModel):
    """Request schema for ``POST /qa``."""

    query: str = Field(min_length=1, description="User query to answer.")
    brief: bool = Field(default=True, description="If true, generate a concise answer.")
    top_k: int = Field(default=5, ge=1, le=50, description="Vector retrieval top-k.")
    k_hop: int = Field(default=2, ge=0, le=10, description="Graph expansion k-hop.")


class Citation(BaseModel):
    """A minimal citation payload for an evidence chunk."""

    chunk_id: str
    source: str = Field(description="Original data source path.")
    page_or_row: Optional[int] = None
    section: Optional[str] = None


class GraphPathEdge(BaseModel):
    """One edge in a graph path."""

    source: str
    relation: str
    target: str
    evidence_chunk_ids: List[str] = Field(default_factory=list)


class GraphPath(BaseModel):
    """A ranked path result."""

    path: List[GraphPathEdge]
    score: float


class QAResponse(BaseModel):
    """Response schema for ``POST /qa``."""

    answer: str
    citations: List[Citation] = Field(default_factory=list)
    graph_paths: List[GraphPath] = Field(default_factory=list)
    debug: Dict[str, Any] = Field(default_factory=dict)


