"""Carrier K5 (TZCSM) fixture harness — deterministic gates C1-C6.

STATUS: fixture-only. No production consumer imports this module. The tables
it targets (domain_source_manifest, corpus_chunks, world_claims,
contradiction_ledger) are empty additive schema from
migrations/zone_c_world_knowledge.sql; no corpus bytes are authorized until
K5 section 8 gates pass (encoder pin + pilot domain + C1-C6 receipts).

ENCODER: the deterministic phasor codec below is a FIXTURE placeholder
(CODEC_VERSION). The production frozen encoder pin is an open user decision
and is NOT implied by this module.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np

DOMAINS = (
    "formal_science", "natural_science", "medicine", "engineering", "computing",
    "law", "economics", "humanities", "language", "arts", "education",
    "practical_skills", "safety", "governance",
)

FIXTURE_CODEC_VERSION = "fixture-phasor-v1-pending-encoder-pin"
NUM_BLOCKS = 8192
BLOCK_DIM = 8
WAVE_BYTES = NUM_BLOCKS * BLOCK_DIM * 4  # float32
PROJ_DIM = 2000

# Fixture contamination markers. Production contamination review is stronger
# (pinned benchmark bytes + prompt contracts); this list is a fixture stand-in.
CONTAMINATION_MARKERS = (
    "GPQA Diamond", "SWE-bench", "HumanEval", "MMMU", "ARC-AGI",
    "Terminal-Bench", "AA-Omniscience", "Artificial Analysis", "T^2 Telecom",
    "QuickPT", "IF Bench", "AA-LCR", "AAII",
)


def _row_angles(seed_hex: str, k: int) -> np.ndarray:
    out = np.empty(BLOCK_DIM, dtype=np.float64)
    for j in range(BLOCK_DIM):
        h = hashlib.sha256(f"{seed_hex}:{k}:{j}".encode()).digest()
        out[j] = (h[0] / 255.0) * 2.0 * np.pi
    return out


def encode_text(text: str) -> bytes:
    """Deterministic [8192,8] float32 wave payload (fixture codec).

    Row-normalized unit vectors; payload bytes are a pure function of text.
    """
    seed_hex = hashlib.sha256(text.encode("utf-8")).hexdigest()
    rows = np.empty((NUM_BLOCKS, BLOCK_DIM), dtype=np.float32)
    for k in range(NUM_BLOCKS):
        v = np.cos(_row_angles(seed_hex, k))
        n = float(np.linalg.norm(v))
        rows[k] = (v / n).astype(np.float32)
    return rows.tobytes()


def encode_text_sha256(text: str) -> str:
    return hashlib.sha256(encode_text(text)).hexdigest()


def project(wave_bytes: bytes) -> np.ndarray:
    """Deterministic 2000-d L2-normalized fixture projection."""
    w = np.frombuffer(wave_bytes, dtype=np.float32).reshape(NUM_BLOCKS, BLOCK_DIM)
    kk = np.arange(NUM_BLOCKS, dtype=np.float32)
    jj = np.arange(PROJ_DIM, dtype=np.float32)
    phase = kk[:, None] * (jj[None, :] + 1.0) * 0.01
    c = np.cos(phase).astype(np.float32)
    proj = (w[:, jj.astype(np.int64) % BLOCK_DIM] * c).sum(axis=0)
    n = float(np.linalg.norm(proj))
    if n < 1e-12:
        return np.zeros(PROJ_DIM, dtype=np.float32)
    return (proj / n).astype(np.float32)


def contamination_gate(text: str) -> tuple[str, str | None]:
    for marker in CONTAMINATION_MARKERS:
        if marker.lower() in text.lower():
            return "CONTAMINATION_REJECT", marker
    return "PASS", None


def verify_claim(claim_text: str, evidence_link: str | None,
                 authorized_sha256: str | None, source_sha256: str) -> str:
    """Fixture independent checker. Abstain unless an evidence link exists and
    the source hash is present and non-empty."""
    if not evidence_link or not source_sha256:
        return "VERIFICATION_ABSTAINED"
    if authorized_sha256 is not None and source_sha256 != authorized_sha256:
        return "VERIFICATION_ABSTAINED"
    return "VERIFIED"


def build_envelope(row: dict) -> dict | None:
    """Fail-closed envelope. Any missing field -> None (typed abstention)."""
    required = ("claim_id", "source_sha256", "char_span", "domain",
                "retrieved_utc", "confidence")
    for f in required:
        if row.get(f) in (None, ""):
            return None
    return {"envelope_type": "RESULT", **{f: row[f] for f in required}}


def classify(row: dict | None, contradicted: bool, stale: bool) -> dict | None:
    """Contradiction/freshness resolution over a candidate row.

    Precedence: missing envelope fields -> None (typed abstention);
    contradicted -> CONTRADICTED; stale -> STALE; else RESULT.
    """
    env = build_envelope(row)
    if env is None:
        return None
    if contradicted:
        env["envelope_type"] = "CONTRADICTED"
    elif stale:
        env["envelope_type"] = "STALE"
    return env


# pgvector cosine distance SQL uses the <=> operator (vector extension).
RETRIEVE_SQL = """
SELECT c.chunk_id, m.sha256 AS source_sha256, c.char_span, c.domain,
       c.ingested_utc, m.updated_utc, m.source_revision,
       1 - (c.proj <=> %s::vector) AS cosine
FROM corpus_chunks c
JOIN domain_source_manifest m ON m.source_id = c.source_id
WHERE c.domain = %s AND c.status = 'VERIFIED'
ORDER BY c.proj <=> %s::vector
LIMIT %s
"""
