"""
Embed chunks with Voyage AI and upsert into Qdrant.
Usage:
  python3 scripts/embed.py <chunks_json> [<chunks_json2> ...]

Requires env vars:
  VOYAGE_API_KEY   — Voyage AI API key
  QDRANT_URL       — defaults to http://localhost:6333
  QDRANT_API_KEY   — optional (required for Qdrant Cloud)

Collection name: goingglobal
Vector size: 512 (voyage-3-lite)
"""
import truststore
truststore.inject_into_ssl()

import json
import os
import sys
import time
import uuid
from pathlib import Path

import voyageai
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

load_dotenv()

COLLECTION = "goingglobal"
MODEL = "voyage-3-lite"
VECTOR_SIZE = 512
BATCH_SIZE = 128
LOG_EVERY = 100


def get_qdrant() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    return QdrantClient(url=url, api_key=api_key)


def ensure_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Created collection '{COLLECTION}'")
    else:
        print(f"Collection '{COLLECTION}' already exists")


def embed_and_upsert(chunks: list[dict], vc: voyageai.Client, qc: QdrantClient) -> int:
    total = len(chunks)
    upserted = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        # Voyage AI embedding with retry
        for attempt in range(3):
            try:
                result = vc.embed(texts, model=MODEL, input_type="document")
                vectors = result.embeddings
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  Voyage retry {attempt+1}: {e}")
                time.sleep(2 ** attempt)

        points = []
        for chunk, vector in zip(batch, vectors):
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

        qc.upsert(collection_name=COLLECTION, points=points)
        upserted += len(points)

        if upserted % LOG_EVERY == 0 or upserted == total:
            print(f"  Upserted {upserted}/{total} chunks ...")

    return upserted


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/embed.py <chunks_json> [<chunks_json2> ...]")
        sys.exit(1)

    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        print("ERROR: VOYAGE_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    vc = voyageai.Client(api_key=api_key)
    qc = get_qdrant()
    ensure_collection(qc)

    all_chunks = []
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks from {path}")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks to embed: {len(all_chunks)}")
    n = embed_and_upsert(all_chunks, vc, qc)
    print(f"\nDone. Upserted {n} vectors into collection '{COLLECTION}'.")

    info = qc.get_collection(COLLECTION)
    print(f"Collection now has {info.vectors_count} vectors.")


if __name__ == "__main__":
    main()
