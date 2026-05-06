"""
GoingGlobal — India's export regulation intelligence hub.
FastAPI + MCP backend backed by Voyage AI embeddings and Qdrant.

Endpoints:
  GET  /          — corpus metadata
  GET  /health    — liveness check
  POST /search    — semantic search, returns cited source chunks
  POST /ingest    — (protected) embed all chunks in data/ into Qdrant
  /mcp            — MCP server (auto-mounted by fastapi-mcp)
"""
import asyncio
import glob
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import truststore
truststore.inject_into_ssl()

import anthropic
import voyageai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mcp import FastApiMCP
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, FieldCondition, MatchValue, PointStruct, VectorParams

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("goingglobal")

COLLECTION = "goingglobal"
MODEL = "voyage-3-lite"
VECTOR_SIZE = 512
INGEST_BATCH = 8

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
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )
    log.info("Connected to Qdrant at %s", os.getenv("QDRANT_URL", "http://localhost:6333"))

    # Auto-ingest if collection empty and chunk files exist
    if os.getenv("AUTO_INGEST", "false").lower() == "true":
        log.info("AUTO_INGEST=true — starting background ingestion...")
        asyncio.create_task(_run_ingest())

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


# ── Models ─────────────────────────────────────────────────────────────────────

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


class IngestResult(BaseModel):
    status: str
    chunks_ingested: int
    by_type: dict

class AskResponse(BaseModel):
    doc_results: list[SearchResult]
    ai_answer: Optional[str] = None
    from_docs: bool


# ── Ingestion ──────────────────────────────────────────────────────────────────

async def _run_ingest() -> IngestResult:
    """Embed all *_chunks.json files in data/ and upsert into Qdrant."""
    chunk_files = sorted(glob.glob("data/*_chunks.json"))
    if not chunk_files:
        raise RuntimeError("No chunk files found in data/")

    all_chunks: list[dict] = []
    for f in chunk_files:
        with open(f) as fh:
            all_chunks.extend(json.load(fh))
    log.info("Ingesting %d chunks from %d files", len(all_chunks), len(chunk_files))

    # Ensure collection exists
    existing = [c.name for c in _qdrant.get_collections().collections]
    if COLLECTION not in existing:
        _qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        log.info("Created collection '%s'", COLLECTION)

    upserted = 0
    by_type: dict[str, int] = {}

    for i in range(0, len(all_chunks), INGEST_BATCH):
        batch = all_chunks[i: i + INGEST_BATCH]
        texts = [c["text"] for c in batch]

        for attempt in range(5):
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda t=texts: _voyage.embed(t, model=MODEL, input_type="document"),
                )
                break
            except Exception as e:
                wait = 20 * (attempt + 1)
                log.warning("Voyage retry %d in %ds: %s", attempt + 1, wait, e)
                await asyncio.sleep(wait)
        else:
            log.error("Batch %d failed after 5 attempts, skipping", i // INGEST_BATCH)
            continue

        points = []
        for chunk, vector in zip(batch, result.embeddings):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "doc_type": chunk.get("doc_type", "Other"),
                    "title": chunk.get("title", ""),
                    "source_url": chunk.get("source_url", ""),
                    "page": chunk.get("page", 0),
                    "source": chunk.get("source", ""),
                },
            ))
            dt = chunk.get("doc_type", "Other")
            by_type[dt] = by_type.get(dt, 0) + 1

        _qdrant.upsert(collection_name=COLLECTION, points=points)
        upserted += len(points)
        log.info("Ingested %d/%d chunks", upserted, len(all_chunks))

        if i + INGEST_BATCH < len(all_chunks):
            await asyncio.sleep(22)  # 3 RPM free-tier pacing

    log.info("Ingestion complete — %d vectors in '%s'", upserted, COLLECTION)
    return IngestResult(status="ok", chunks_ingested=upserted, by_type=by_type)


# ── Helpers ────────────────────────────────────────────────────────────────────

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


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    corpus = _corpus_stats() if _qdrant else {"total_documents": 0, "by_type": {}}
    return {
        "name": "goingglobal",
        "version": "1.0.0",
        "tagline": "India's export regulation intelligence. Citation-first, retrieval-only.",
        "endpoints": {"mcp": "/mcp", "health": "/health", "search": "/search", "ingest": "/ingest"},
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


@app.post("/ingest", response_model=IngestResult)
async def ingest(x_ingest_key: Optional[str] = Header(None)):
    """
    Embed all chunk files in data/ and upsert into Qdrant.
    Protected by INGEST_KEY env var (if set). Call once after deployment.
    """
    if _voyage is None or _qdrant is None:
        raise HTTPException(503, "Service not initialised")

    ingest_key = os.getenv("INGEST_KEY")
    if ingest_key and x_ingest_key != ingest_key:
        raise HTTPException(401, "Missing or invalid X-Ingest-Key header")

    try:
        return await _run_ingest()
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/search", response_model=list[SearchResult])
async def search(req: SearchRequest):
    """
    Semantic search over ingested Indian export regulation documents.
    Returns retrieved source chunks with citations. Never generates answers.
    """
    if _voyage is None or _qdrant is None:
        raise HTTPException(503, "Service not initialised")

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _voyage.embed([req.query], model=MODEL, input_type="query"),
        )
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


async def _ai_answer(query: str, chunks: list[SearchResult]) -> Optional[str]:
    # Azure AI Services endpoint takes priority; falls back to direct Anthropic key
    azure_endpoint = os.getenv("AZURE_ANTHROPIC_ENDPOINT")
    azure_key = os.getenv("AZURE_ANTHROPIC_KEY")
    direct_key = os.getenv("ANTHROPIC_API_KEY")

    if azure_endpoint and azure_key:
        # Azure AI Services uses 'api-key' header, not 'x-api-key'
        client = anthropic.Anthropic(
            base_url=azure_endpoint,
            api_key=azure_key,
            default_headers={"api-key": azure_key},
        )
        model = os.getenv("AZURE_MODEL_NAME", "claude-sonnet-4-6")
        log.info("Using Azure AI endpoint: %s", azure_endpoint)
    elif direct_key:
        client = anthropic.Anthropic(api_key=direct_key)
        model = "claude-haiku-4-5-20251001"
        log.info("Using direct Anthropic API")
    else:
        log.warning("No AI credentials configured (AZURE_ANTHROPIC_KEY or ANTHROPIC_API_KEY)")
        return None

    context = "\n\n".join(
        f"[{c.doc_type} · p.{c.page}]: {c.text[:350]}" for c in chunks[:3]
    ) if chunks else "No relevant documents found."
    prompt = (
        f"You are an expert on Indian export regulations, helping SME exporters.\n\n"
        f"Context from official documents:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer in 2–4 plain sentences. If the context contains the answer, use it. "
        "If not, answer from general knowledge and start with 'Based on general knowledge:'. "
        "Never invent regulation numbers or cite rules you are not certain of."
    )
    try:
        msg = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.messages.create(
                model=model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            ),
        )
        return msg.content[0].text
    except Exception as e:
        log.warning("AI answer failed: %s", e)
        return None


@app.post("/ask", response_model=AskResponse)
async def ask(req: SearchRequest):
    """
    Semantic search with LLM fallback.
    Returns doc excerpts + an AI-generated answer (if ANTHROPIC_API_KEY is set).
    """
    if _voyage is None or _qdrant is None:
        raise HTTPException(503, "Service not initialised")

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _voyage.embed([req.query], model=MODEL, input_type="query"),
        )
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

    doc_results = [
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

    best_score = doc_results[0].score if doc_results else 0.0
    from_docs = best_score >= 0.58

    ai_answer = await _ai_answer(req.query, doc_results)

    return AskResponse(doc_results=doc_results, ai_answer=ai_answer, from_docs=from_docs)


# ── MCP ────────────────────────────────────────────────────────────────────────

mcp = FastApiMCP(app)
mcp.mount()
