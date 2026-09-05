"""Carrier K5 / C3 — Zone C world-knowledge harness (retrieval + egress).

End-to-end path over the SEALED K5 schema on main (04f2f61):

  query text
    -> C2 codec (zone_c_world_knowledge_codec.proj_of)
    -> pgvector cosine retrieval on corpus_chunks.proj
       (mandatory domain filter; partial HNSW per domain)
    -> provenance-bearing chunk list (source_id, char_span, chunk_sha256,
       cosine). Raw text is NOT stored in Zone C (world-knowledge boundary);
       text for egress context is re-sliced from the hash-verified source
       file at files_dir/<source_id> and its sha256 is re-checked against
       corpus_chunks.chunk_sha256 before use.
    -> frozen backbone generate_text (henri_backbone_adapter, default-OFF
       HENRI_BACKBONE=1, zero-trainable) with the retrieved context
    -> typed answer + provenance + telemetry, or fail-closed abstain.

Writes (ingest) are default-OFF (HENRI_K5_HARNESS=1) and simply delegate to
zone_c_world_knowledge_ingest.ingest (mode=real|fixture|dry-run) so receipts,
contamination gate, and claim verification stay in ONE place. Reads are
read-only and need no harness flag.

CLI:
  python zone_c_world_knowledge_harness.py ingest --manifest M --files-dir F [--mode real|fixture|dry-run] [--commit]
  python zone_c_world_knowledge_harness.py query --domain D --query Q --k N --dsn-env PATH
  python zone_c_world_knowledge_harness.py generate --domain D --query Q --files-dir F [--k N] [--max-context-chars C] [--dsn-env PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# errors + gates
# ---------------------------------------------------------------------------


class HarnessError(RuntimeError):
    """Base class for C3 harness contract failures."""


class HarnessDisabledError(HarnessError):
    """Raised when an ingest write path is used without HENRI_K5_HARNESS=1."""


def harness_enabled() -> bool:
    return os.environ.get("HENRI_K5_HARNESS", "0").strip() in {"1", "true", "True", "yes"}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_dsn_env(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        raise HarnessError(f"dsn env file not found: {p}")
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip().endswith("_DSN") and value.strip():
            return value.strip()
    raise HarnessError(f"no *_DSN entry found in {p}")


def resolve_dsn(dsn_env: str | None) -> str:
    if dsn_env:
        return parse_dsn_env(dsn_env)
    dsn = os.environ.get("K5_TZCSM_TEST_DSN", "").strip()
    if not dsn:
        raise HarnessError("no database DSN: pass --dsn-env PATH or set K5_TZCSM_TEST_DSN")
    return dsn


# ---------------------------------------------------------------------------
# retrieval (read-only)
# ---------------------------------------------------------------------------


def query_corpus(domain: str, query: str, k: int, *, dsn_env: str | None = None,
                 dsn: str | None = None,
                 conn: Any | None = None) -> list[dict[str, Any]]:
    """Top-k provenance-bearing chunks for one domain (mandatory filter).

    Pass an open psycopg connection as conn to run inside an existing
    transaction (rollback-only tests); otherwise a new connection is opened
    from dsn/dsn_env/K5_TZCSM_TEST_DSN and closed on return.
    """
    import psycopg
    from zone_c_world_knowledge_codec import get_codec

    if k <= 0:
        raise HarnessError(f"invalid k: {k}")
    own_conn = conn is None
    if own_conn:
        conn = psycopg.connect(dsn or resolve_dsn(dsn_env), connect_timeout=10)
    qv = get_codec().proj_of(query).astype(float)
    vec = "[" + ",".join(f"{float(x):.6f}" for x in qv) + "]"
    out: list[dict[str, Any]] = []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, source_id, domain, chunk_index, char_span,
                       chunk_sha256, 1 - (proj <=> %s::vector) AS cosine
                FROM corpus_chunks
                WHERE domain = %s
                ORDER BY proj <=> %s::vector
                LIMIT %s
                """,
                (vec, domain, vec, int(k)),
            )
            for row in cur.fetchall():
                out.append({
                    "chunk_id": row[0],
                    "source_id": row[1],
                    "domain": row[2],
                    "chunk_index": int(row[3]),
                    "char_span": row[4],
                    "chunk_sha256": row[5],
                    "cosine": float(row[6]),
                })
    finally:
        if own_conn:
            conn.close()
    return out


def slice_chunk_text(source_file: Path, char_span: str, expected_sha256: str,
                     chunk_index: int) -> str:
    """Re-slice raw text from the source file and verify the chunk digest.

    Raw text is never read from Zone C; it is read from the authorized
    source bytes and verified against the stored chunk_sha256 before use.
    """
    if not source_file.is_file():
        raise HarnessError(f"source file missing: {source_file}")
    try:
        start_s, end_s = char_span.split(":")
        start, end = int(start_s), int(end_s)
    except ValueError as exc:
        raise HarnessError(f"malformed char_span {char_span!r}") from exc
    data = source_file.read_bytes()
    text = data.decode("utf-8", errors="replace")
    if end > len(text) or start < 0 or start > end:
        raise HarnessError(
            f"chunk {chunk_index}: span {char_span} outside source length {len(text)}"
        )
    chunk = text[start:end]
    actual = _sha256_bytes(chunk.encode("utf-8"))
    if actual != expected_sha256:
        raise HarnessError(
            f"chunk {chunk_index}: sha256 {actual[:16]}... != stored "
            f"{expected_sha256[:16]}... (source bytes changed or manifest drift)"
        )
    return chunk


def build_context(chunks: list[dict[str, Any]], files_dir: Path,
                  max_chars: int = 6000) -> tuple[str, list[dict[str, Any]]]:
    """Assemble verified chunk text into a context block with provenance."""
    parts: list[str] = []
    used: list[dict[str, Any]] = []
    total = 0
    for c in chunks:
        text = slice_chunk_text(files_dir / c["source_id"], c["char_span"],
                                c["chunk_sha256"], c["chunk_index"])
        if total + len(text) > max_chars:
            text = text[: max(0, max_chars - total)]
        if text:
            parts.append(f"[source {c['source_id']} span {c['char_span']} cos {c['cosine']:.3f}]\n{text}")
            used.append(c)
            total += len(text)
        if total >= max_chars:
            break
    return "\n\n".join(parts), used


# ---------------------------------------------------------------------------
# egress (frozen backbone, default-OFF)
# ---------------------------------------------------------------------------


def generate_answer(query: str, context: str, *,
                    max_new_tokens: int = 512) -> tuple[str, dict[str, Any]]:
    """Frozen-backbone generation over retrieved context (zero-trainable).

    Requires HENRI_BACKBONE=1 and HENRI_BACKBONE_MODEL_DIR (or default
    artifact dir). Fail-closed typed errors otherwise.
    """
    from henri_backbone_adapter import QwenBackboneAdapter

    prompt = (
        "Answer the question using ONLY the provided evidence.\n"
        "If the evidence is insufficient, answer: ABSTAIN.\n\n"
        f"EVIDENCE:\n{context}\n\nQUESTION: {query}\n\nANSWER:"
    )
    adapter = QwenBackboneAdapter()
    adapter.load()
    text, telemetry = adapter.generate_text(prompt)
    meta = {
        "prompt_chars": len(prompt),
        "context_chars": len(context),
        "max_new_tokens": max_new_tokens,
        "telemetry": telemetry.__dict__ if hasattr(telemetry, "__dict__") else str(telemetry),
    }
    return text, meta


# ---------------------------------------------------------------------------
# ingest delegation (default-OFF)
# ---------------------------------------------------------------------------


def ingest(manifest: str, files_dir: str, *, mode: str = "real", commit: bool = False,
           claims: str | None = None, dsn_env: str | None = None,
           dsn: str | None = None) -> dict[str, Any]:
    """Delegate to the sealed K5 ingest runner (single source of truth)."""
    if mode != "dry-run" and not harness_enabled():
        raise HarnessDisabledError(
            "HENRI_K5_HARNESS is not set; ingest write modes are disabled"
        )
    from zone_c_world_knowledge_ingest import ingest as k5_ingest

    if mode != "dry-run" and dsn is None:
        dsn = resolve_dsn(dsn_env)
    return k5_ingest(
        manifest, files_dir, claims_path=claims, mode=mode,
        dsn=dsn, commit=commit,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_ingest(args: argparse.Namespace) -> int:
    report = ingest(args.manifest, args.files_dir, mode=args.mode,
                    commit=args.commit, claims=args.claims, dsn_env=args.dsn_env)
    import json
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    hits = query_corpus(args.domain, args.query, args.k, dsn_env=args.dsn_env)
    import json
    print(json.dumps(hits, indent=2, sort_keys=True))
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    hits = query_corpus(args.domain, args.query, args.k, dsn_env=args.dsn_env)
    if not hits:
        print(json.dumps({"answer": "ABSTAIN", "reason": "no retrieved chunks"}, sort_keys=True))
        return 0
    context, used = build_context(hits, Path(args.files_dir),
                                  max_chars=args.max_context_chars)
    if not context:
        print(json.dumps({"answer": "ABSTAIN", "reason": "empty verified context"}, sort_keys=True))
        return 0
    answer, meta = generate_answer(args.query, context, max_new_tokens=args.max_new_tokens)
    import json
    print(json.dumps({"answer": answer, "provenance": used, "meta": meta}, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description="Zone C world-knowledge harness (C3)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="delegate to the sealed K5 ingest runner")
    pi.add_argument("--manifest", required=True)
    pi.add_argument("--files-dir", required=True)
    pi.add_argument("--claims", default=None)
    pi.add_argument("--mode", choices=("dry-run", "fixture", "real"), default="dry-run")
    pi.add_argument("--commit", action="store_true")
    pi.add_argument("--dsn-env", default=None)
    pi.set_defaults(fn=_cmd_ingest)

    pq = sub.add_parser("query", help="top-k provenance-bearing retrieval")
    pq.add_argument("--domain", required=True)
    pq.add_argument("--query", required=True)
    pq.add_argument("--k", type=int, default=5)
    pq.add_argument("--dsn-env", default=None)
    pq.set_defaults(fn=_cmd_query)

    pg = sub.add_parser("generate", help="retrieve + frozen-backbone answer")
    pg.add_argument("--domain", required=True)
    pg.add_argument("--query", required=True)
    pg.add_argument("--files-dir", required=True)
    pg.add_argument("--k", type=int, default=3)
    pg.add_argument("--max-context-chars", type=int, default=6000)
    pg.add_argument("--max-new-tokens", type=int, default=512)
    pg.add_argument("--dsn-env", default=None)
    pg.set_defaults(fn=_cmd_generate)

    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
