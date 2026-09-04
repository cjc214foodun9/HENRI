"""Carrier K5 (TZCSM) contract tests: C1-C6 fixture gates + additive schema.

Fixture level only. The fixture codec is a deterministic placeholder
(FIXTURE_CODEC_VERSION); the production frozen-encoder pin is a separate
open decision and is NOT exercised here.

DB-dependent tests run against the LOCAL DEV DSN only when
K5_TZCSM_TEST_DSN is set; all writes roll back (no persistence). Without the
env var (remote read-only runs), DB tests skip and the pure gates still run.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = REPO_ROOT / "HENRI V2"

from zone_c_world_knowledge_fixtures import (  # noqa: E402
    BLOCK_DIM,
    CONTAMINATION_MARKERS,
    FIXTURE_CODEC_VERSION,
    NUM_BLOCKS,
    WAVE_BYTES,
    build_envelope,
    classify,
    contamination_gate,
    encode_text,
    encode_text_sha256,
    project,
    verify_claim,
)

DSN = os.environ.get("K5_TZCSM_TEST_DSN")
needs_db = pytest.mark.skipif(not DSN, reason="K5_TZCSM_TEST_DSN not set")

PROD_SCAN_FILES = [
    "production_arc_run.py",
    "efe_planner.py",
    "zone_c_axiom_seeder.py",
    "zone_c_epistemic_axiom_harness.py",
]


@pytest.fixture()
def db():
    if not DSN:
        pytest.skip("K5_TZCSM_TEST_DSN not set")
    import psycopg

    conn = psycopg.connect(DSN, connect_timeout=5)
    conn.autocommit = False
    yield conn
    conn.rollback()
    conn.close()


def _insert_source(cur, source_id, domain, sha256, title="T", origin="fixture://test",
                   retrieved_utc="2026-09-04T00:00:00Z", revision="r1"):
    cur.execute(
        """
        INSERT INTO domain_source_manifest
            (source_id, domain, title, origin, sha256, license,
             retrieved_utc, updated_utc, source_revision)
        VALUES (%s, %s, %s, %s, %s, 'CC0', %s, now(), %s)
        ON CONFLICT (source_id) DO NOTHING
        """,
        (source_id, domain, title, origin, sha256, retrieved_utc, revision),
    )


def _vec(vals) -> str:
    """pgvector text-literal formatting for numpy/list values."""
    return "[" + ",".join(f"{float(x):.6f}" for x in vals) + "]"


def _insert_chunk(cur, source_id, domain, text, idx=0, status="VERIFIED"):
    wb = encode_text(text)
    chunk_id = f"{source_id}:{idx}"
    cur.execute(
        """
        INSERT INTO corpus_chunks
            (chunk_id, source_id, domain, chunk_index, char_span,
             chunk_sha256, wave_payload, proj, claim_count, ingested_utc, status)
        VALUES (%s, %s, %s, %s, '0-10', %s, %s, %s::vector, 0, now(), %s)
        ON CONFLICT (chunk_id) DO NOTHING
        """,
        (chunk_id, source_id, domain, idx, hashlib.sha256(wb).hexdigest(),
         wb, _vec(project(wb)), status),
    )
    return chunk_id


# --------------------------------------------------------------------------- C1
def test_c1_cross_process_hash_identity():
    text = "The boiling point of water at one atmosphere is 100 degrees Celsius."
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from zone_c_world_knowledge_fixtures import encode_text_sha256;"
        "print(encode_text_sha256(%r))" % (str(CODE_DIR), text)
    )
    outs = []
    for _ in range(2):
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=180)
        assert r.returncode == 0, r.stderr[-800:]
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1]
    assert len(outs[0]) == 64


def test_c1_wave_geometry_payload():
    wb = encode_text("fixed fixture text 1")
    assert len(wb) == WAVE_BYTES == NUM_BLOCKS * BLOCK_DIM * 4
    w = np.frombuffer(wb, dtype=np.float32).reshape(NUM_BLOCKS, BLOCK_DIM)
    norms = np.linalg.norm(w, axis=1)
    assert float(np.abs(norms - 1.0).max()) < 1e-6
    assert FIXTURE_CODEC_VERSION.startswith("fixture-phasor")


def test_c1_default_off_no_production_consumer():
    for fn in PROD_SCAN_FILES:
        p = CODE_DIR / fn
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        assert "zone_c_world_knowledge_fixtures" not in txt, fn
        assert "zone_c_world_knowledge.sql" not in txt, fn


# --------------------------------------------------------------------------- C2
def test_c2_contamination_gate_rejects_all_markers():
    for m in CONTAMINATION_MARKERS:
        status, hit = contamination_gate(f"Sample question about {m} with details.")
        assert status == "CONTAMINATION_REJECT", m
        assert hit == m


def test_c2_benign_text_passes():
    status, hit = contamination_gate(
        "Photosynthesis converts carbon dioxide and water into glucose.")
    assert status == "PASS"
    assert hit is None


# --------------------------------------------------------------------------- C3
@needs_db
def test_c3_contradiction_never_one_sided(db):
    cur = db.cursor()
    _insert_source(cur, "s-a", "natural_science", "a" * 64, title="A")
    _insert_source(cur, "s-b", "natural_science", "b" * 64, title="B")
    cid_a = _insert_chunk(cur, "s-a", "natural_science",
                          "Water boils at 100 C at sea level.")
    cid_b = _insert_chunk(cur, "s-b", "natural_science",
                          "Water boils at 95 C at sea level.")
    cur.execute(
        """
        INSERT INTO world_claims
            (claim_id, chunk_id, domain, claim_text_hash, claim_type,
             verification_status, evidence_link)
        VALUES (gen_random_uuid(), %s, 'natural_science', %s, 'fact',
                'VERIFIED', 'fixture://ev/a')
        RETURNING claim_id
        """,
        (cid_a, hashlib.sha256(b"text a").hexdigest()),
    )
    ca = cur.fetchone()[0]
    cur.execute(
        """
        INSERT INTO world_claims
            (claim_id, chunk_id, domain, claim_text_hash, claim_type,
             verification_status, evidence_link)
        VALUES (gen_random_uuid(), %s, 'natural_science', %s, 'fact',
                'VERIFIED', 'fixture://ev/b')
        RETURNING claim_id
        """,
        (cid_b, hashlib.sha256(b"text b").hexdigest()),
    )
    cb = cur.fetchone()[0]
    cur.execute(
        "INSERT INTO contradiction_ledger (claim_id_a, claim_id_b, relation)"
        " VALUES (%s, %s, 'CONTRADICTS')",
        (ca, cb),
    )
    row = {"claim_id": str(ca), "source_sha256": "a" * 64, "char_span": "0-10",
           "domain": "natural_science", "retrieved_utc": "2026-09-04T00:00:00Z",
           "confidence": 0.9}
    assert classify(row, contradicted=True, stale=False)["envelope_type"] == "CONTRADICTED"
    assert classify(row, contradicted=False, stale=False)["envelope_type"] == "RESULT"
    cur.execute(
        "SELECT relation FROM contradiction_ledger"
        " WHERE claim_id_a = %s AND claim_id_b = %s",
        (ca, cb),
    )
    assert cur.fetchone()[0] == "CONTRADICTS"


# --------------------------------------------------------------------------- C4
def test_c4_freshness_downgrade():
    row = {"claim_id": "c", "source_sha256": "s" * 64, "char_span": "0-10",
           "domain": "computing", "retrieved_utc": "2026-09-01T00:00:00Z",
           "confidence": 0.8}
    assert classify(row, contradicted=False, stale=True)["envelope_type"] == "STALE"
    assert classify(row, contradicted=False, stale=False)["envelope_type"] == "RESULT"


# --------------------------------------------------------------------------- C5
def test_c5_envelope_completeness_fail_closed():
    complete = {"claim_id": "c1", "source_sha256": "s" * 64, "char_span": "0-10",
                "domain": "law", "retrieved_utc": "2026-09-04T00:00:00Z",
                "confidence": 0.95}
    env = build_envelope(complete)
    assert env is not None
    assert env["envelope_type"] == "RESULT"
    for field in ("claim_id", "source_sha256", "char_span", "domain",
                  "retrieved_utc", "confidence"):
        bad = dict(complete)
        bad[field] = None
        assert build_envelope(bad) is None, field
    assert classify(complete, contradicted=True, stale=True)["envelope_type"] == "CONTRADICTED"


# --------------------------------------------------------------------------- C6
@needs_db
def test_c6_retrieval_domain_filter_and_match(db):
    cur = db.cursor()
    _insert_source(cur, "src-quick", "computing", "q" * 64, title="Quick")
    _insert_source(cur, "src-med", "medicine", "m" * 64, title="Med")
    qid = _insert_chunk(cur, "src-quick", "computing",
                        "def quicksort(arr): return sorted(arr)")
    _insert_chunk(cur, "src-med", "medicine",
                  "Ibuprofen reduces fever and inflammation.")
    q = project(encode_text("def quicksort(arr): return sorted(arr)")).tolist()
    cur.execute(
        "SELECT c.chunk_id FROM corpus_chunks c"
        " WHERE c.domain = 'computing'"
        " ORDER BY c.proj <=> %s::vector LIMIT 1",
        (q,),
    )
    assert cur.fetchone()[0] == qid
    # absent query must not spuriously match at cosine ~1.0
    a = project(encode_text("A completely different fact about the economy.")).tolist()
    cur.execute(
        "SELECT 1 - (c.proj <=> %s::vector) FROM corpus_chunks c"
        " WHERE c.domain = 'computing'"
        " ORDER BY c.proj <=> %s::vector LIMIT 1",
        (a, a),
    )
    assert cur.fetchone()[0] < 0.999
    # medicine chunk must stay invisible to the computing-domain filter
    m = project(encode_text("Ibuprofen reduces fever and inflammation.")).tolist()
    cur.execute(
        "SELECT 1 - (c.proj <=> %s::vector) FROM corpus_chunks c"
        " WHERE c.domain = 'computing'"
        " ORDER BY c.proj <=> %s::vector LIMIT 1",
        (m, m),
    )
    assert cur.fetchone()[0] < 0.999


def test_c6_evidence_required_grader():
    assert verify_claim("fact", None, None, "a" * 64) == "VERIFICATION_ABSTAINED"
    assert verify_claim("fact", "fixture://ev", "a" * 64, "a" * 64) == "VERIFIED"
    assert verify_claim("fact", "fixture://ev", "a" * 64, "b" * 64) == "VERIFICATION_ABSTAINED"


# ------------------------------------------------------------------ schema
@needs_db
def test_schema_additive_receipt(db):
    cur = db.cursor()
    cur.execute("SELECT version FROM zone_c_schema_migrations ORDER BY version")
    assert [r[0] for r in cur.fetchall()] == [1, 2]
    cur.execute("SELECT count(*) FROM pg_indexes WHERE indexname LIKE 'corpus_chunks_proj_%'")
    assert cur.fetchone()[0] == 14
    for t in ("domain_source_manifest", "corpus_chunks", "world_claims",
              "contradiction_ledger"):
        cur.execute(f"SELECT count(*) FROM {t}")
        cur.fetchone()
    cur.execute("SELECT count(*) FROM boundary_axioms")
    assert cur.fetchone()[0] == 11
