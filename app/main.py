"""
FastAPI backend for the GraphRAG MVP.

Endpoints:
- ``GET /health``: basic health + index readiness
- ``POST /qa``: GraphRAG-style QA over the built index
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agents.graph import AgentError, answer_query
from app.schemas import HealthResponse, QARequest, QAResponse
from app.settings import Settings

app = FastAPI(title="GraphRAG MVP", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process."""

    return Settings.from_env()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health endpoint."""

    return HealthResponse(status="ok")


@app.post("/qa", response_model=QAResponse)
def qa(req: QARequest) -> QAResponse:
    """Answer a query using the built GraphRAG artifacts."""

    settings = get_settings()
    if not settings.index_ready():
        raise HTTPException(
            status_code=400,
            detail=(
                "Index artifacts not found. Run "
                "`python -m ingestion.build_index --data_dir data/sources --out_dir data/index` "
                "and verify INDEX_DIR / CHROMA_DIR / GRAPH_PATH."
            ),
        )

    try:
        result = answer_query(
            query=req.query,
            brief=req.brief,
            top_k=req.top_k,
            k_hop=req.k_hop,
            settings=settings,
        )
        return QAResponse(
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
            graph_paths=result.get("graph_paths", []),
            debug=result.get("debug", {}),
        )
    except AgentError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


