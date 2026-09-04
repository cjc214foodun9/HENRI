"""Carrier K5 (TZCSM) — production ingest runner contract tests.

Cover: default-OFF gating, fail-closed encoder boundary (BLOCKED_NO_ENCODER),
source hash mismatch, contamination rejection, claim abstention vs verified,
chunk boundaries, and receipt-only DB mode (rolled back) against a dev PG
fixture when K5_TZCSM_TEST_DSN is set.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib

import pytest

from zone_c_world_knowledge_encoder_pin import EncoderDisabledError
from zone_c_world_knowledge_ingest import (
    EncoderUnavailableError,
    IngestDisabledError,
    SourceHashMismatchError,
    chunk_text,
    contamination_gate,
    fixture_encode,
    fixture_projection,
    ingest,
    verify_claim_independent,
)


def _write_manifest(dirpath: pathlib.Path, rows: list[dict]) -> pathlib.Path:
    p = dirpath / "manifest.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return p


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _source_row(source_id: str, text: str, domain: str = "computing") -> dict:
    return {
        "source_id": source_id,
        "domain": domain,
        "title": "fixture source",
        "origin": "fixture",
        "sha256": _sha256_text(text),
        "license": "FIXTURE",
        "retrieved_utc": "2026-09-04T00:00:00+00:00",
        "source_revision": "v0",
    }


def _claims_for(source_id: str, idx: int, text: str, evidence: str | None) -> list[dict]:
    return [{
        "source_id": source_id,
        "chunk_index": idx,
        "claim_text": text,
        "claim_type": "DEFINITION",
        "evidence_link": evidence,
    }]


@pytest.fixture()
def scene(tmp_path):
    source_id = "src_a.txt"
    text = "The quick brown fox jumps over the lazy dog. " * 60
    src = tmp_path / source_id
    src.write_text(text, encoding="utf-8")
    rows = [_source_row(source_id, text)]
    manifest = _write_manifest(tmp_path, rows)
    return {"tmp": tmp_path, "manifest": manifest, "source_id": source_id, "text": text}


# ---------- chunking ----------

def test_chunk_text_contiguous_and_bounded():
    text = "a" * 1000
    chunks = list(chunk_text(text, chunk_size=300, overlap=50))
    assert chunks[0][0] == 0
    indices = [idx for idx, _, _ in chunks]
    assert indices[0] == 0
    for (i, span, chunk) in chunks:
        a, b = (int(x) for x in span.split(":"))
        assert chunk == text[a:b]
        assert len(chunk) <= 300
    # contiguous coverage
    assert "".join(c for _, _, c in chunks).startswith(text[:300])


def test_chunk_text_invalid():
    with pytest.raises(Exception):
        list(chunk_text("abc", chunk_size=0, overlap=0))


# ---------- contamination ----------

def test_contamination_gate_detects_benchmark_marker():
    status, marker = contamination_gate("some text with GPQA Diamond questions inside")
    assert status == "CONTAMINATION_REJECT"
    assert marker == "GPQA Diamond"


def test_contamination_gate_clean():
    assert contamination_gate("ordinary text about foxes") == ("PASS", None)


# ---------- encoder gates ----------

def test_fixture_encode_deterministic_shape():
    w1 = fixture_encode("same text")
    w2 = fixture_encode("same text")
    assert w1 == w2
    assert len(w1) == 8192 * 8 * 4
    proj = fixture_projection(w1)
    assert proj.shape == (2000,)


def test_real_mode_blocked_no_encoder(monkeypatch, scene):
    monkeypatch.setenv("K5_INGEST_ENCODER", "1")
    with pytest.raises(EncoderUnavailableError):
        ingest(str(scene["manifest"]), str(scene["tmp"]), mode="real")


def test_fixture_mode_requires_ingest_flag(monkeypatch, scene):
    monkeypatch.delenv("K5_INGEST_ENCODER", raising=False)
    monkeypatch.setenv("K5_FIXTURE_ENCODER", "1")
    with pytest.raises(EncoderDisabledError):
        ingest(str(scene["manifest"]), str(scene["tmp"]), mode="fixture",
               dsn="postgresql://u:p@localhost:1/db")


def test_fixture_mode_requires_fixture_flag(monkeypatch, scene):
    monkeypatch.setenv("K5_INGEST_ENCODER", "1")
    monkeypatch.delenv("K5_FIXTURE_ENCODER", raising=False)
    with pytest.raises(IngestDisabledError):
        ingest(str(scene["manifest"]), str(scene["tmp"]), mode="fixture",
               dsn="postgresql://u:p@localhost:1/db")


# ---------- dry-run ----------

def test_dry_run_no_flags_no_db(monkeypatch, scene):
    report = ingest(str(scene["manifest"]), str(scene["tmp"]), mode="dry-run")
    assert report["mode"] == "dry-run"
    assert len(report["sources"]) == 1
    s = report["sources"][0]
    assert s["chunks_total"] > 0
    assert s["chunks_written"] == s["chunks_total"]
    assert s["chunks_rejected"] == 0


def test_dry_run_reports_rejected_chunk(monkeypatch, scene, tmp_path):
    text = scene["text"] + " GPQA Diamond target line"
    src = tmp_path / "src_b.txt"
    src.write_text(text, encoding="utf-8")
    rows = [_source_row("src_b.txt", text)]
    manifest = _write_manifest(tmp_path, rows)
    report = ingest(str(manifest), str(tmp_path), mode="dry-run", chunk_size=500, overlap=50)
    s = report["sources"][0]
    assert s["chunks_rejected"] >= 1


def test_dry_run_abstains_without_evidence(monkeypatch, scene):
    claims = _claims_for(scene["source_id"], 0, "a claim", None)
    claims_path = scene["tmp"] / "claims.jsonl"
    with open(claims_path, "w", encoding="utf-8") as f:
        for c in claims:
            f.write(json.dumps(c) + "\n")
    report = ingest(str(scene["manifest"]), str(scene["tmp"]),
                    claims_path=str(claims_path), mode="dry-run", chunk_size=500, overlap=50)
    s = report["sources"][0]
    assert s["abstained_claims"] >= 1


# ---------- hash mismatch ----------

def test_source_hash_mismatch_fail_closed(scene, tmp_path):
    wrong = scene["tmp"] / "src_wrong.txt"
    wrong.write_text("different bytes", encoding="utf-8")
    rows = [_source_row("src_wrong.txt", "expected content")]
    manifest = _write_manifest(tmp_path, rows)
    with pytest.raises(SourceHashMismatchError):
        ingest(str(manifest), str(tmp_path), mode="dry-run")


# ---------- claim checker ----------

def test_verify_claim_independent():
    sha = "a" * 64
    assert verify_claim_independent("c", "http://ev", sha, sha) == "VERIFIED"
    assert verify_claim_independent("c", None, sha, sha) == "VERIFICATION_ABSTAINED"
    assert verify_claim_independent("c", "http://ev", sha, "b" * 64) == "VERIFICATION_ABSTAINED"


# ---------- receipt-only DB mode (dev PG fixture; skip without DSN) ----------

@pytest.mark.skipif(not os.environ.get("K5_TZCSM_TEST_DSN"), reason="no K5_TZCSM_TEST_DSN")
def test_fixture_db_receipt_only_rolls_back(monkeypatch, scene):
    monkeypatch.setenv("K5_INGEST_ENCODER", "1")
    monkeypatch.setenv("K5_FIXTURE_ENCODER", "1")
    report = ingest(str(scene["manifest"]), str(scene["tmp"]), mode="fixture",
                    dsn=os.environ["K5_TZCSM_TEST_DSN"], commit=False, chunk_size=500, overlap=50)
    assert report["tx"] == "ROLLED_BACK_RECEIPT_ONLY"
    assert len(report["coverage_receipts"]) == 1
    rc = report["coverage_receipts"][0]
    assert rc["domain"] == "computing"
    assert rc["chunk_count"] > 0  # visible inside the transaction
    # after rollback nothing persists
    import psycopg
    with psycopg.connect(os.environ["K5_TZCSM_TEST_DSN"]) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM domain_source_manifest WHERE source_id = %s",
                        (scene["source_id"],))
            assert cur.fetchone()[0] == 0
