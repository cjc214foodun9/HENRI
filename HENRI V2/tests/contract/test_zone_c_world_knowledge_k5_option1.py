"""Carrier K5 (TZCSM) option 1 — frozen encoder pin + manifest/receipt tests.

CPU-only. Real shard hashing (C1 cross-process identity) runs on the CUDA
target where the artifact lives; these tests cover the default-OFF gate,
fail-closed mismatch behavior, pin-record shape, manifest validation, and
coverage receipts against the dev PG fixture (empty tables -> NOT_EVALUATED).
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from zone_c_world_knowledge_encoder_pin import (
    EncoderArtifactMismatchError,
    EncoderDisabledError,
    EncoderPinError,
    pin_record,
    verify_encoder_artifact,
)
from zone_c_world_knowledge_manifest import (
    DOMAINS,
    ManifestError,
    build_coverage_receipt,
    load_manifest,
    validate_manifest_row,
)


# ---------- encoder pin ----------

def test_pin_record_shape():
    rec = pin_record()
    assert rec["model_id"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert len(rec["revision"]) == 40
    assert len(rec["shards"]) == 4
    for name, digest in rec["shards"].items():
        assert name.startswith("model-0000") and name.endswith(".safetensors")
        assert len(digest) == 64


def test_encoder_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("K5_INGEST_ENCODER", raising=False)
    with pytest.raises(EncoderDisabledError):
        verify_encoder_artifact(str(tmp_path))


def test_encoder_fail_closed_on_missing_shard(monkeypatch, tmp_path):
    monkeypatch.setenv("K5_INGEST_ENCODER", "1")
    with pytest.raises((EncoderPinError, EncoderArtifactMismatchError)):
        verify_encoder_artifact(str(tmp_path))  # empty dir


def test_encoder_fail_closed_on_mismatch(monkeypatch, tmp_path):
    monkeypatch.setenv("K5_INGEST_ENCODER", "1")
    for name in pin_record()["shards"]:
        (tmp_path / name).write_bytes(b"not-a-real-shard")
    with pytest.raises(EncoderArtifactMismatchError):
        verify_encoder_artifact(str(tmp_path))


# ---------- manifest ----------

def test_manifest_domains_complete():
    assert len(DOMAINS) == 14
    for d in ("formal_science", "medicine", "governance", "safety"):
        assert d in DOMAINS


def test_validate_manifest_row_ok():
    validate_manifest_row({
        "source_id": "s1",
        "domain": "medicine",
        "title": "t",
        "origin": "https://example.org/x",
        "sha256": "a" * 64,
        "license": "CC-BY-4.0",
        "retrieved_utc": "2026-09-04T00:00:00Z",
    })


@pytest.mark.parametrize("mutator", [
    lambda r: r.update({"domain": "nope"}),
    lambda r: r.update({"sha256": "short"}),
    lambda r: r.update({"retrieved_utc": "yesterday"}),
    lambda r: r.pop("source_id"),
])
def test_validate_manifest_row_rejects(mutator):
    row = {
        "source_id": "s1",
        "domain": "medicine",
        "title": "t",
        "origin": "https://example.org/x",
        "sha256": "a" * 64,
        "license": "CC-BY-4.0",
        "retrieved_utc": "2026-09-04T00:00:00Z",
    }
    mutator(row)
    with pytest.raises(ManifestError):
        validate_manifest_row(row)


def test_load_manifest_jsonl(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(
        json.dumps({
            "source_id": "s1", "domain": "law", "title": "t",
            "origin": "https://example.org/x", "sha256": "b" * 64,
            "license": "CC0", "retrieved_utc": "2026-09-04T00:00:00+00:00",
        })
        + "\n",
        encoding="utf-8",
    )
    rows = load_manifest(p)
    assert len(rows) == 1 and rows[0]["source_id"] == "s1"


def test_load_manifest_empty(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert load_manifest(p) == []


def test_load_manifest_missing(tmp_path):
    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "absent.jsonl")


# ---------- coverage receipt (dev PG fixture; empty tables) ----------

def test_coverage_receipt_empty_tables():
    import os
    dsn = os.environ.get("K5_TZCSM_TEST_DSN")
    if not dsn:
        pytest.skip("K5_TZCSM_TEST_DSN not set; receipt DB test skipped")
    import psycopg
    with psycopg.connect(dsn, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            rec = build_coverage_receipt(cur, "medicine")
    assert rec["status"] == "NOT_EVALUATED"
    assert rec["chunk_count"] == 0
    assert rec["claim_count"] == 0
