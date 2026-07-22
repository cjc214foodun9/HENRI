"""Local Obsidian vault vector indexer + query server for HENRI V2.

Indexes Markdown notes (YAML frontmatter aware) into a persistent ChromaDB
collection using local sentence-transformer embeddings (no API tokens), and
serves semantic search on http://127.0.0.1:8000/query.

Usage:
    python scripts/local_vault_search_server.py [--vault ./Obsidian_Vault]
                                                [--port 8000]

Endpoints:
    GET /health            -> {"status": "ok", "indexed": <chunk count>}
    GET /query?q=...&top_k=3&module=Anisotropic Langevin&created_after=2026-06-01

Reindex: restart the process (index built at startup), or POST /reindex.
"""
import argparse
import glob
import hashlib
import os
import re

import chromadb
import frontmatter
import uvicorn
from chromadb.utils import embedding_functions
from fastapi import FastAPI, Query

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION = "henri_vault_embeddings"

app = FastAPI(title="HENRI Local Vault Context Engine")

_state: dict = {"collection": None, "vault_dir": None}


def _chunk_sections(content: str) -> list[str]:
    """Split a note into chunks on '## ' section boundaries (first chunk is
    everything before the first '##')."""
    parts = content.split("\n## ")
    return [p.strip() for p in parts if p.strip()]


def index_obsidian_vault(vault_dir: str) -> int:
    collection = _state["collection"]
    md_files = glob.glob(os.path.join(vault_dir, "**", "*.md"), recursive=True)
    documents, metadatas, ids = [], [], []

    for filepath in md_files:
        norm = filepath.replace("\\", "/")
        if "/.obsidian/" in norm or "/.vault_vector_db/" in norm or "/.trash/" in norm:
            continue
        try:
            post = frontmatter.load(filepath)
            meta = dict(post.metadata)
            created = str(meta.get("created_at", ""))
            # ChromaDB $gte/$lte only accept numeric operands — store an
            # integer YYYYMMDD alongside the ISO string for range filters.
            try:
                created_ymd = int(re.sub(r"[^0-9]", "", created)[:8])
            except ValueError:
                created_ymd = 0
            base_meta = {
                "file_path": norm,
                "title": str(meta.get("id") or os.path.basename(filepath)),
                "module": str(meta.get("module", "General")),
                "created_at": created,
                "created_ymd": created_ymd,
                "status": str(meta.get("status", "draft")),
            }
            for sec_idx, section in enumerate(_chunk_sections(post.content)):
                # Content-addressed IDs: stable across re-index, dedupes edits.
                doc_id = hashlib.sha1(f"{norm}::{sec_idx}::{section}".encode()).hexdigest()
                documents.append(section)
                metadatas.append(base_meta)
                ids.append(doc_id)
        except Exception as e:  # noqa: BLE001 — skip malformed notes, keep indexing
            print(f"Skipping {filepath}: {e}")

    # Rebuild: drop stale chunks for files that changed or vanished.
    existing = collection.get(include=[])
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    if documents:
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Indexed {len(documents)} chunks from {len(md_files)} files in {vault_dir}")
    return len(documents)


@app.get("/health")
def health() -> dict:
    count = _state["collection"].count() if _state["collection"] else 0
    return {"status": "ok", "indexed": count}


@app.post("/reindex")
def reindex() -> dict:
    n = index_obsidian_vault(_state["vault_dir"])
    return {"status": "ok", "indexed": n}


def _to_ymd(date_str: str) -> int:
    """'2026-07-16' or ISO datetime -> 20260716 for numeric range filters."""
    return int(re.sub(r"[^0-9]", "", date_str)[:8])


@app.get("/query")
def query_vault(
    q: str = Query(..., description="Semantic search query"),
    top_k: int = 3,
    module: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
) -> dict:
    where = None
    clauses = []
    if module:
        clauses.append({"module": module})
    if created_after:
        clauses.append({"created_ymd": {"$gte": _to_ymd(created_after)}})
    if created_before:
        clauses.append({"created_ymd": {"$lte": _to_ymd(created_before)}})
    if len(clauses) == 1:
        where = clauses[0]
    elif clauses:
        where = {"$and": clauses}

    results = _state["collection"].query(
        query_texts=[q], n_results=top_k, where=where
    )
    context = []
    for i in range(len(results["ids"][0])):
        context.append({
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i] if "distances" in results else None,
        })
    return {"query": q, "context": context}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", default="./Obsidian_Vault")
    ap.add_argument("--db", default="./.vault_vector_db")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    client = chromadb.PersistentClient(path=args.db)
    _state["collection"] = client.get_or_create_collection(
        name=COLLECTION, embedding_function=embedding_fn
    )
    _state["vault_dir"] = args.vault
    index_obsidian_vault(args.vault)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
