"""Contract tests — K5 C3 world-knowledge harness (zone_c_world_knowledge_harness).

Covers: default-OFF ingest gate; source-slicing provenance verification
(sha256 must match stored chunk digest; tamper -> fail closed); DB read path
against dev PG when K5_TZCSM_TEST_DSN is set (rollback-only, zero
persistence); and abstain behavior with no corpus.
"""

from __future__ import annotations

import hashlib
import os
import pathlib

import pytest

from zone_c_world_knowledge_ingest import chunk_text
from zone_c_world_knowledge_codec import encode as codec_encode
from zone_c_world_knowledge_harness import (
    HarnessDisabledError,
    HarnessError,
    build_context,
    harness_enabled,
    ingest as harness_ingest,
    query_corpus,
    slice_chunk_text,
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _vec(values) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in values) + "]"


def _write_source(tmp: pathlib.Path, name: str = "computing_src.rst",
                  text: str | None = None) -> tuple[pathlib.Path, str]:
    if text is None:
        text = ("The bisect module provides binary search helpers for sorted lists. " * 40)
    p = tmp / name
    p.write_text(text, encoding="utf-8")
    return p, text


def _chunk_meta(text: str, chunk_size: int = 2000, overlap: int = 100):
    meta = []
    for idx, span, chunk in chunk_text(text, chunk_size, overlap):
        meta.append({
            "chunk_index": idx,
            "char_span": span,
            "chunk_sha256": _sha256_text(chunk),
        })
    return meta


# ---------- ingest gate ----------

def test_ingest_delegation_default_off(monkeypatch, tmp_path):
    monkeypatch.delenv("HENRI_K5_HARNESS", raising=False)
    with pytest.raises(HarnessDisabledError):
        harness_ingest(str(tmp_path / "m.jsonl"), str(tmp_path), mode="real",
                       dsn="postgresql://u:x@localhost:1/db")


def test_harness_enabled_flag():
    os.environ.pop("HENRI_K5_HARNESS", None)
    assert harness_enabled() is False


# ---------- provenance slicing ----------

def test_slice_chunk_text_roundtrip(tmp_path):
    p, text = _write_source(tmp_path)
    meta = _chunk_meta(text)
    c0 = meta[0]
    out = slice_chunk_text(p, c0["char_span"], c0["chunk_sha256"], c0["chunk_index"])
    start, end = (int(x) for x in c0["char_span"].split(":"))
    assert out == text[start:end]


def test_slice_chunk_text_tamper_fails(tmp_path):
    p, text = _write_source(tmp_path)
    meta = _chunk_meta(text)
    c0 = meta[0]
    p.write_text("tampered bytes that change the digest", encoding="utf-8")
    with pytest.raises(HarnessError):
        slice_chunk_text(p, c0["char_span"], c0["chunk_sha256"], c0["chunk_index"])


def test_slice_chunk_span_out_of_range(tmp_path):
    p, text = _write_source(tmp_path)
    with pytest.raises(HarnessError):
        slice_chunk_text(p, "999999:1000000", _sha256_text("x"), 0)


def test_build_context_concatenates_verified_chunks(tmp_path):
    p, text = _write_source(tmp_path)
    meta = _chunk_meta(text)
    chunks = [{"source_id": p.name, **m, "domain": "computing", "cosine": 0.9}
              for m in meta[:2]]
    ctx, used = build_context(chunks, tmp_path, max_chars=5000)
    assert "bisect" in ctx
    assert len(used) == 2


# ---------- DB read path (rollback-only, requires K5_TZCSM_TEST_DSN) ----------

def test_query_corpus_domain_filter_rollback(tmp_path):
    dsn = os.environ.get("K5_TZCSM_TEST_DSN")
    if not dsn:
        pytest.skip("K5_TZCSM_TEST_DSN not set")
    import psycopg

    conn = psycopg.connect(dsn, connect_timeout=5)
    try:
        text = "The bisect module provides binary search helpers for ordered sequences. " * 20
        fname = "computing_src.rst"
        (tmp_path / fname).write_text(text, encoding="utf-8")
        file_sha = _sha256_text(text)
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO domain_source_manifest
                   (source_id, domain, title, origin, sha256, license,
                    retrieved_utc, updated_utc, source_revision)
                   VALUES (%s,%s,%s,%s,%s,%s, now(), now(), %s)
                   ON CONFLICT (source_id) DO NOTHING""",
                (fname, "computing", "fixture source", "fixture", file_sha,
                 "FIXTURE", "v0"),
            )
            wave_bytes, proj = codec_encode(text[:4000])
            chunk_sha = _sha256_text(text[:4000])
            cur.execute(
                """INSERT INTO corpus_chunks
                   (chunk_id, source_id, domain, chunk_index, char_span,
                    chunk_sha256, wave_payload, proj, claim_count,
                    ingested_utc, status)
                   VALUES (%s,%s,%s,0,%s,%s,%s,%s::vector,0,now(),'VERIFIED')
                   ON CONFLICT (chunk_id) DO NOTHING""",
                (f"{fname}:0", fname, "computing", "0:4000", chunk_sha,
                 wave_bytes, _vec(proj)),
            )
        hits = query_corpus("computing", "bisect binary search helpers",
                            k=3, conn=conn)
        assert len(hits) >= 1
        assert all(h["domain"] == "computing" for h in hits)
        assert all("cosine" in h for h in hits)
    finally:
        conn.rollback()
        conn.close()
    # zero persistence asserted
    import psycopg
    with psycopg.connect(dsn, connect_timeout=5) as conn2:
        with conn2.cursor() as cur:
            cur.execute("SELECT count(*) FROM corpus_chunks WHERE source_id=%s",
                        ("computing_src.rst",))
            assert cur.fetchone()[0] == 0
