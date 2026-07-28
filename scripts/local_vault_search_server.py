"""Local Obsidian graph projection and vector index server for HENRI.

The external Obsidian vault is the source of local agentic graph memory. The
append-only event store under ``<vault>/_agentic`` is authoritative. Chroma is
only a derived semantic index over Markdown projections.

This server never connects to Zone C and never stores HENRI wave checkpoints.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import sys
from pathlib import Path

import chromadb
import frontmatter
import uvicorn
from chromadb.utils import embedding_functions
from fastapi import FastAPI, Query

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from agentic_event_store import query_events, verify_local_events, write_projection  # noqa: E402

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION = "henri_vault_embeddings"
app = FastAPI(title="HENRI Local Agentic Graph Memory")
_state: dict = {"collection": None, "vault_dir": None, "db_dir": None}


def _chunk_sections(content: str) -> list[str]:
    parts = content.split("\n## ")
    return [p.strip() for p in parts if p.strip()]


def index_obsidian_vault(vault_dir: str) -> int:
    collection = _state["collection"]
    md_files = glob.glob(os.path.join(vault_dir, "**", "*.md"), recursive=True)
    documents, metadatas, ids = [], [], []
    for filepath in md_files:
        norm = filepath.replace("\\", "/")
        if any(part in norm for part in ("/.obsidian/", "/.vault_vector_db/", "/.trash/", "/_agentic/")):
            continue
        try:
            post = frontmatter.load(filepath)
            meta = dict(post.metadata)
            created = str(meta.get("created_at", meta.get("event_time", "")))
            digits = re.sub(r"[^0-9]", "", created)
            created_ymd = int(digits[:8]) if len(digits) >= 8 else 0
            base_meta = {
                "file_path": norm,
                "title": str(meta.get("id") or os.path.basename(filepath)),
                "module": str(meta.get("module", "General")),
                "created_at": created,
                "created_ymd": created_ymd,
                "status": str(meta.get("status", "draft")),
                "causal_status": str(meta.get("causal_status", "unclassified")),
                "audit_hash": str(meta.get("audit_hash", "")),
            }
            for sec_idx, section in enumerate(_chunk_sections(post.content)):
                doc_id = hashlib.sha256(f"{norm}::{sec_idx}::{section}".encode()).hexdigest()
                documents.append(section)
                metadatas.append(base_meta)
                ids.append(doc_id)
        except Exception as exc:  # keep one malformed note from blocking reindex
            print(f"Skipping {filepath}: {type(exc).__name__}: {exc}")
    existing = collection.get(include=[])
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
    if documents:
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    return len(documents)


@app.get("/health")
def health() -> dict:
    count = _state["collection"].count() if _state["collection"] else 0
    valid, message = verify_local_events(_state["vault_dir"])
    return {
        "status": "ok" if valid else "blocked",
        "indexed_chunks": count,
        "local_event_store": message,
        "zone_c_connection": "not_used_by_local_server",
    }


@app.post("/reindex")
def reindex() -> dict:
    count = index_obsidian_vault(_state["vault_dir"])
    projection = write_projection(_state["vault_dir"])
    return {"status": "ok", "indexed": count, "projection": str(projection)}


def _to_ymd(date_str: str) -> int:
    digits = re.sub(r"[^0-9]", "", date_str)
    if len(digits) < 8:
        raise ValueError("date must contain YYYYMMDD")
    return int(digits[:8])


@app.get("/events")
def events(
    stream: str | None = None,
    event_type: str | None = None,
    after: str | None = None,
    before: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
) -> dict:
    return {
        "events": query_events(
            vault_path=_state["vault_dir"],
            stream=stream,
            event_type=event_type,
            after=after,
            before=before,
            limit=limit,
        )
    }


@app.get("/query")
def query_vault(
    q: str = Query(..., description="Semantic search query"),
    top_k: int = Query(3, ge=1, le=50),
    module: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
) -> dict:
    clauses = []
    if module:
        clauses.append({"module": module})
    if created_after:
        clauses.append({"created_ymd": {"$gte": _to_ymd(created_after)}})
    if created_before:
        clauses.append({"created_ymd": {"$lte": _to_ymd(created_before)}})
    where = None
    if len(clauses) == 1:
        where = clauses[0]
    elif clauses:
        where = {"$and": clauses}
    result = _state["collection"].query(query_texts=[q], n_results=top_k, where=where)
    context = []
    for i in range(len(result["ids"][0])):
        context.append({
            "content": result["documents"][0][i],
            "metadata": result["metadatas"][0][i],
            "distance": result["distances"][0][i] if "distances" in result else None,
        })
    return {"query": q, "context": context, "memory_layer": "local_obsidian_projection"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", default=os.environ.get("OBSIDIAN_VAULT_PATH", ""))
    ap.add_argument("--db", default="")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    vault = Path(args.vault).expanduser().resolve() if args.vault else (Path.home() / "Documents" / "HENRI_Research_Vault").resolve()
    if not vault.exists():
        raise SystemExit(f"OBSIDIAN_VAULT_PATH does not exist: {vault}")
    db = Path(args.db).expanduser().resolve() if args.db else vault / "_agentic" / "vector_index"
    db.mkdir(parents=True, exist_ok=True)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(db))
    _state["collection"] = client.get_or_create_collection(name=COLLECTION, embedding_function=embedding_fn)
    _state["vault_dir"] = str(vault)
    _state["db_dir"] = str(db)
    index_obsidian_vault(str(vault))
    write_projection(str(vault))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
