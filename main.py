"""
GoingGlobal — India's export regulation intelligence hub.
FastAPI backend with Voyage AI embeddings, Qdrant retrieval, and MCP support.

Endpoints:
  GET  /          — metadata
  GET  /health    — liveness check
  POST /search    — semantic search over ingested export regulation docs
  /mcp            — MCP server (auto-mounted by fastapi-mcp)
"""
import os
from contextlib import asynccontextmanager
from typing import Optional

import truststore
truststore.inject_into_ssl()

import voyageai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mcp import FastApiMCP
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, QueryRequest

load_dotenv()

COLLECTION = "goingglobal"
MODEL = "voyage-3-lite"

_voyage: voyageai.Client | None = None
_qdrant: QdrantClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _voyage, _qdrant
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY not set")
    _voyage = voyageai.Client(api_key=api_key)
    _qdrant = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    yield


app = FastAPI(
    title="GoingGlobal",
    version="1.0.0",
    description="India's export regulation intelligence. Citation-first, retrieval-only.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Models ────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    top_k: int = Field(5, ge=1, le=20)
    filter_type: Optional[str] = Field(
        None,
        description="Filter by doc_type: RBI, FEMA, DGFT, GST, EXIM, ECGC, Customs",
    )


class SearchResult(BaseModel):
    text: str
    doc_type: str
    title: str
    source_url: str
    page: int
    score: float


# ── Helpers ───────────────────────────────────────────────────────────────────

def _corpus_stats() -> dict:
    try:
        info = _qdrant.get_collection(COLLECTION)
        total = info.vectors_count or 0
        by_type: dict[str, int] = {}
        results, _ = _qdrant.scroll(
            collection_name=COLLECTION,
            limit=10_000,
            with_payload=["doc_type"],
            with_vectors=False,
        )
        for r in results:
            dt = r.payload.get("doc_type", "Other")
            by_type[dt] = by_type.get(dt, 0) + 1
        return {"total_documents": total, "by_type": by_type}
    except Exception:
        return {"total_documents": 0, "by_type": {}}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    corpus = _corpus_stats() if _qdrant else {"total_documents": 0, "by_type": {}}
    return {
        "name": "goingglobal",
        "version": "1.0.0",
        "tagline": "India's export regulation intelligence. Citation-first, retrieval-only.",
        "endpoints": {
            "mcp": "/mcp",
            "health": "/health",
            "search": "/search",
        },
        "corpus": corpus,
        "disclaimer": (
            "Unofficial resource. Not affiliated with RBI, DGFT, or any government body. "
            "Verify all information with a qualified CA before acting."
        ),
        "license": "Apache-2.0",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/search", response_model=list[SearchResult])
async def search(req: SearchRequest):
    """
    Semantic search over ingested Indian export regulation documents.
    Returns retrieved source chunks with citations. Never generates answers.
    """
    if _voyage is None or _qdrant is None:
        raise HTTPException(503, "Service not initialised")

    try:
        result = _voyage.embed([req.query], model=MODEL, input_type="query")
        vector = result.embeddings[0]
    except Exception as e:
        raise HTTPException(502, f"Embedding failed: {e}")

    qdrant_filter = None
    if req.filter_type:
        qdrant_filter = Filter(
            must=[FieldCondition(key="doc_type", match=MatchValue(value=req.filter_type))]
        )

    try:
        response = _qdrant.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=req.top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        hits = response.points
    except Exception as e:
        raise HTTPException(502, f"Vector search failed: {e}")

    return [
        SearchResult(
            text=h.payload.get("text", ""),
            doc_type=h.payload.get("doc_type", "Other"),
            title=h.payload.get("title", ""),
            source_url=h.payload.get("source_url", ""),
            page=h.payload.get("page", 0),
            score=round(h.score, 4),
        )
        for h in hits
    ]


# ── MCP ───────────────────────────────────────────────────────────────────────

mcp = FastApiMCP(app)
mcp.mount()
