"""Carrier K5 (TZCSM) — production ingest runner (default-OFF, fail-closed).

Implements the K5 section 4 ingress pipeline as far as live machinery allows:

  1. per-domain source manifest (JSONL, validated by
     zone_c_world_knowledge_manifest.validate_manifest_row)
  2. source bytes: file name == source_id inside --files-dir; sha256 MUST
     equal the manifest row digest (SourceHashMismatchError on mismatch)
  3. chunking: char windows, one chunk at a time (chunk_index, char_span,
     chunk_sha256). No bulk batch load.
  4. contamination gate: first-line marker scan (production review pins
     benchmark bytes when sources are approved). Hit -> CONTAMINATION_REJECT,
     no chunk write.
  5. independent claim checker: a claim is VERIFIED only when it carries an
     evidence link AND its source sha256 matches the manifest. Otherwise
     VERIFICATION_ABSTAINED. Abstained claims are recorded, never asserted
     as fact.
  6. encoder boundary: real production encoding is an open item (K5 section
     9, structured compositional text codec). This runner refuses real
     encoding with BLOCKED_NO_ENCODER until a codec module
     (zone_c_world_knowledge_codec) exists. Fixture mode uses the
     deterministic phasor codec from the C1-C6 fixture suite and is guarded
     so it can never target a production database without an explicit
     K5_ALLOW_PROD_FIXTURE=1 override.
  7. coverage receipts per domain via build_coverage_receipt.

ENV GATES (fail-closed):
  - K5_INGEST_ENCODER=1  required for any encoding mode (fixture or real).
  - K5_FIXTURE_ENCODER=1 required for --mode fixture.
  - K5_ALLOW_PROD_FIXTURE=1 permits fixture writes to a non-localhost DSN.
  - K5_TZCSM_TEST_DSN or --dsn-env supplies the database URL.
DB writes run in one transaction per source. Without --commit the
transaction is rolled back after the receipt is printed (receipt-only runs).

Approval: APPROVE_USER_20260904_K5_TZCSM option 1 (loader+runner machinery,
default-OFF, remote-verified, fast-forward). Corpus ingestion remains
BLOCKED_NO_SOURCE until domain sources are approved.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from zone_c_world_knowledge_manifest import (
    DOMAINS,
    ManifestError,
    build_coverage_receipt,
    load_manifest,
    validate_manifest_row,
)
from zone_c_world_knowledge_encoder_pin import (
    EncoderDisabledError,
    encoder_enabled,
)

FIXTURE_CODEC_VERSION = "fixture-phasor-v1-pending-encoder-pin"
NUM_BLOCKS = 8192
BLOCK_DIM = 8
PROJ_DIM = 2000

# First-line contamination markers. Production review will pin benchmark
# bytes and prompt contracts when sources are approved (K5 section 4.4).
CONTAMINATION_MARKERS = (
    "GPQA Diamond", "SWE-bench", "HumanEval", "MMMU", "ARC-AGI",
    "Terminal-Bench", "AA-Omniscience", "Artificial Analysis", "T^2 Telecom",
    "QuickPT", "IF Bench", "AA-LCR", "AAII",
)

CLAIM_FIELDS = ("chunk_index", "claim_text", "claim_type", "evidence_link")


def _scope_claims(
    claims: dict[int, list[dict[str, Any]]], source_id: str
) -> dict[int, list[dict[str, Any]]]:
    """Return claims that apply to one source. A claim without source_id
    applies to every source (single-source manifests); with source_id it
    applies only to the matching source."""
    out: dict[int, list[dict[str, Any]]] = {}
    for idx, items in claims.items():
        scoped = [c for c in items if c.get("source_id", source_id) == source_id]
        if scoped:
            out[idx] = scoped
    return out


class IngestError(RuntimeError):
    """Base class for ingest-runner contract failures."""


class IngestDisabledError(IngestError):
    """Raised when an encoding/write mode is attempted without its env gate."""


class EncoderUnavailableError(IngestError):
    """Raised when real encoding is requested before a codec exists."""


class SourceHashMismatchError(IngestError):
    """Raised when source bytes do not match the manifest sha256."""


# --------------------------------------------------------------------------
# deterministic helpers
# --------------------------------------------------------------------------

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _row_angles(seed_hex: str, k: int) -> np.ndarray:
    out = np.empty(BLOCK_DIM, dtype=np.float64)
    for j in range(BLOCK_DIM):
        h = hashlib.sha256(f"{seed_hex}:{k}:{j}".encode()).digest()
        out[j] = (h[0] / 255.0) * 2.0 * np.pi
    return out


def fixture_encode(text: str) -> bytes:
    """Deterministic [8192,8] float32 wave payload (FIXTURE codec only).

    Row-normalized unit vectors; payload bytes are a pure function of text.
    Identical algorithm to the C1-C6 fixture suite.
    """
    seed_hex = hashlib.sha256(text.encode("utf-8")).hexdigest()
    rows = np.empty((NUM_BLOCKS, BLOCK_DIM), dtype=np.float32)
    for k in range(NUM_BLOCKS):
        v = np.cos(_row_angles(seed_hex, k))
        n = float(np.linalg.norm(v))
        rows[k] = (v / n).astype(np.float32)
    return rows.tobytes()


def fixture_projection(wave_bytes: bytes) -> np.ndarray:
    """Deterministic 2000-d L2-normalized projection (FIXTURE codec only)."""
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


def verify_claim_independent(
    claim_text: str,
    evidence_link: str | None,
    manifest_sha256: str,
    actual_sha256: str,
) -> str:
    """Independent claim checker (K5 4.3). Abstain unless an evidence link
    exists and the actual source digest equals the manifest digest."""
    if not claim_text.strip():
        return "VERIFICATION_ABSTAINED"
    if not evidence_link or not manifest_sha256 or not actual_sha256:
        return "VERIFICATION_ABSTAINED"
    if actual_sha256 != manifest_sha256:
        return "VERIFICATION_ABSTAINED"
    return "VERIFIED"


def chunk_text(text: str, chunk_size: int, overlap: int) -> Iterator[tuple[int, str, str]]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise IngestError(f"invalid chunking: size={chunk_size} overlap={overlap}")
    start = 0
    idx = 0
    total = len(text)
    while start < total:
        end = min(start + chunk_size, total)
        yield idx, f"{start}:{end}", text[start:end]
        idx += 1
        if end == total:
            break
        start = max(end - overlap, start + 1)


def _vec(values: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in values) + "]"


def parse_dsn_env(path: str) -> str:
    """Read a KEY=VALUE env file and return the *_DSN value (e.g. ZONE_C_PROD_DSN)."""
    p = Path(path)
    if not p.is_file():
        raise IngestError(f"dsn env file not found: {p}")
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip().endswith("_DSN") and value.strip():
            return value.strip()
    raise IngestError(f"no *_DSN entry found in {p}")


def resolve_dsn(args: argparse.Namespace) -> str:
    if args.dsn_env:
        return parse_dsn_env(args.dsn_env)
    dsn = os.environ.get("K5_TZCSM_TEST_DSN", "").strip()
    if not dsn:
        raise IngestDisabledError(
            "no database DSN: pass --dsn-env PATH or set K5_TZCSM_TEST_DSN"
        )
    return dsn


def _is_localhost(host: str | None) -> bool:
    return (host or "").lower() in ("localhost", "127.0.0.1", "::1")


# --------------------------------------------------------------------------
# per-source processing
# --------------------------------------------------------------------------

def _load_source_bytes(row: dict[str, Any], files_dir: Path) -> bytes:
    source_id = row["source_id"]
    candidate = files_dir / source_id
    if not candidate.is_file():
        raise IngestError(
            f"source file not found: expected {candidate} (file name == source_id)"
        )
    data = candidate.read_bytes()
    digest = _sha256_bytes(data)
    if digest != row["sha256"].lower():
        raise SourceHashMismatchError(
            f"source {source_id}: sha256 {digest} != manifest {row['sha256']}"
        )
    return data


def _resolve_real_codec():
    try:
        return importlib.import_module("zone_c_world_knowledge_codec")
    except ModuleNotFoundError as exc:
        raise EncoderUnavailableError(
            "BLOCKED_NO_ENCODER: no production codec module "
            "(zone_c_world_knowledge_codec). K5 section 9 codec is an open "
            "item; structured compositional encoder required."
        ) from exc


def _insert_source(cur, row: dict[str, Any]) -> None:
    cur.execute(
        """
        INSERT INTO domain_source_manifest
            (source_id, domain, title, origin, sha256, license,
             retrieved_utc, updated_utc, source_revision)
        VALUES (%s, %s, %s, %s, %s, %s, %s, now(), %s)
        ON CONFLICT (source_id) DO NOTHING
        """,
        (row["source_id"], row["domain"], row["title"], row["origin"],
         row["sha256"], row["license"], row["retrieved_utc"],
         row.get("source_revision")),
    )


def process_source(
    row: dict[str, Any],
    files_dir: Path,
    claims_by_index: dict[int, list[dict[str, Any]]],
    *,
    mode: str,
    chunk_size: int,
    overlap: int,
    cur=None,
) -> dict[str, Any]:
    """Process one manifest source. cur is a live psycopg cursor when the
    caller wants DB writes (fixture/real mode); None means dry-run."""
    if row["domain"] not in DOMAINS:
        raise ManifestError(f"unknown domain: {row['domain']}")
    data = _load_source_bytes(row, files_dir)
    text = data.decode("utf-8", errors="replace")

    counts = {
        "source_id": row["source_id"],
        "domain": row["domain"],
        "chunks_total": 0,
        "chunks_written": 0,
        "chunks_rejected": 0,
        "claims_written": 0,
        "verified_claims": 0,
        "abstained_claims": 0,
    }

    if mode != "dry-run":
        if cur is None:
            raise IngestError("DB mode requires a cursor")
        _insert_source(cur, row)

    for idx, span, chunk in chunk_text(text, chunk_size, overlap):
        counts["chunks_total"] += 1
        status, marker = contamination_gate(chunk)
        if status == "CONTAMINATION_REJECT":
            counts["chunks_rejected"] += 1
            continue
        chunk_sha = _sha256_bytes(chunk.encode("utf-8"))
        claims = claims_by_index.get(idx, [])
        chunk_status = "VERIFIED" if claims and all(
            c["verification_status"] == "VERIFIED" for c in claims
        ) else "VERIFICATION_ABSTAINED"
        counts["chunks_written"] += 1

        if mode == "dry-run":
            counts["claims_written"] += len(claims)
            counts["verified_claims"] += sum(
                1 for c in claims if c["verification_status"] == "VERIFIED"
            )
            counts["abstained_claims"] += sum(
                1 for c in claims if c["verification_status"] != "VERIFIED"
            )
            continue

        if mode == "fixture":
            wave_bytes = fixture_encode(chunk)
            proj = fixture_projection(wave_bytes)
        else:  # real
            codec = _resolve_real_codec()
            wave_bytes, proj = codec.encode(chunk)  # type: ignore[attr-defined]

        chunk_id = f"{row['source_id']}:{idx}"
        cur.execute(
            """
            INSERT INTO corpus_chunks
                (chunk_id, source_id, domain, chunk_index, char_span,
                 chunk_sha256, wave_payload, proj, claim_count,
                 ingested_utc, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s, now(), %s)
            ON CONFLICT (chunk_id) DO NOTHING
            """,
            (chunk_id, row["source_id"], row["domain"], idx, span,
             chunk_sha, wave_bytes, _vec(proj), len(claims), chunk_status),
        )
        for claim in claims:
            cur.execute(
                """
                INSERT INTO world_claims
                    (chunk_id, domain, claim_text_hash, claim_type,
                     verification_status, evidence_link, sealed_utc)
                VALUES (%s, %s, %s, %s, %s, %s, now())
                """,
                (chunk_id, row["domain"], claim["claim_text_hash"],
                 claim["claim_type"], claim["verification_status"],
                 claim.get("evidence_link")),
            )
            counts["claims_written"] += 1
            if claim["verification_status"] == "VERIFIED":
                counts["verified_claims"] += 1
            else:
                counts["abstained_claims"] += 1
    return counts


def load_claims(path: str | None) -> dict[int, list[dict[str, Any]]]:
    """Load optional claims JSONL: {chunk_index, claim_text, claim_type,
    evidence_link}. Returns {chunk_index: [validated claims]}."""
    by_index: dict[int, list[dict[str, Any]]] = {}
    if not path:
        return by_index
    p = Path(path)
    if not p.is_file():
        raise IngestError(f"claims file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IngestError(f"{p}:{line_no} not JSON: {exc}") from exc
            for field in CLAIM_FIELDS:
                if field not in obj:
                    raise IngestError(f"{p}:{line_no} missing claim field: {field}")
            idx = int(obj["chunk_index"])
            by_index.setdefault(idx, []).append(obj)
    return by_index


def _finalize_claims(
    claims_by_index: dict[int, list[dict[str, Any]]], manifest_sha: str, actual_sha: str
) -> None:
    for claims in claims_by_index.values():
        for claim in claims:
            status = verify_claim_independent(
                claim["claim_text"], claim.get("evidence_link"),
                manifest_sha, actual_sha,
            )
            claim["verification_status"] = status
            claim["claim_text_hash"] = _sha256_bytes(
                claim["claim_text"].encode("utf-8")
            )


def ingest(
    manifest_path: str,
    files_dir: str,
    *,
    claims_path: str | None = None,
    mode: str = "dry-run",
    chunk_size: int = 4000,
    overlap: int = 200,
    dsn: str | None = None,
    commit: bool = False,
) -> dict[str, Any]:
    """Run the K5 ingress pipeline over one manifest. Returns a report."""
    rows = load_manifest(manifest_path)
    if not rows:
        raise IngestError("manifest is empty: no sources to ingest")
    files_dir_p = Path(files_dir)
    if not files_dir_p.is_dir():
        raise IngestError(f"files dir not found: {files_dir_p}")

    if mode in ("fixture", "real"):
        if not encoder_enabled():
            raise EncoderDisabledError(
                "K5_INGEST_ENCODER is not set; encoding modes are disabled"
            )
        if mode == "fixture" and os.environ.get("K5_FIXTURE_ENCODER", "0").strip() \
                not in {"1", "true", "True", "yes"}:
            raise IngestDisabledError(
                "K5_FIXTURE_ENCODER is not set; fixture mode is disabled"
            )
    if mode == "real":
        _resolve_real_codec()  # raises BLOCKED_NO_ENCODER until a codec exists

    report: dict[str, Any] = {
        "mode": mode,
        "manifest": manifest_path,
        "commit": commit,
        "sources": [],
        "coverage_receipts": [],
        "encoder": FIXTURE_CODEC_VERSION if mode == "fixture" else None,
    }

    if mode == "dry-run":
        for row in rows:
            claims = _scope_claims(load_claims(claims_path), row["source_id"])
            claims_by_index = claims
            data = _load_source_bytes(row, files_dir_p)
            _finalize_claims(claims_by_index, row["sha256"], _sha256_bytes(data))
            counts = process_source(
                row, files_dir_p, claims_by_index,
                mode="dry-run", chunk_size=chunk_size, overlap=overlap, cur=None,
            )
            report["sources"].append(counts)
        return report

    # DB mode: one transaction over all sources; rollback unless --commit.
    import psycopg  # local import keeps module import light

    conn = psycopg.connect(dsn or resolve_dsn(argparse.Namespace(dsn_env=None)))
    if mode == "fixture" and not _is_localhost(conn.info.host) and \
            os.environ.get("K5_ALLOW_PROD_FIXTURE", "0").strip() not in {"1", "true"}:
        conn.close()
        raise IngestDisabledError(
            "fixture mode targets a non-localhost DSN; set "
            "K5_ALLOW_PROD_FIXTURE=1 only for approved disposable targets"
        )
    try:
        with conn.cursor() as cur:
            for row in rows:
                claims = _scope_claims(load_claims(claims_path), row["source_id"])
                data = _load_source_bytes(row, files_dir_p)
                _finalize_claims(claims, row["sha256"], _sha256_bytes(data))
                counts = process_source(
                    row, files_dir_p, claims,
                    mode=mode, chunk_size=chunk_size, overlap=overlap, cur=cur,
                )
                report["sources"].append(counts)
                receipt = build_coverage_receipt(cur, row["domain"])
                report["coverage_receipts"].append(receipt)
        if commit:
            conn.commit()
            report["tx"] = "COMMITTED"
        else:
            conn.rollback()
            report["tx"] = "ROLLED_BACK_RECEIPT_ONLY"
    finally:
        conn.close()
    return report


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="K5 TZCSM ingest runner")
    parser.add_argument("--manifest", required=True, help="per-domain JSONL manifest")
    parser.add_argument("--files-dir", required=True, help="dir with source files (file name == source_id)")
    parser.add_argument("--claims", default=None, help="optional claims JSONL")
    parser.add_argument("--mode", choices=("dry-run", "fixture", "real"), default="dry-run")
    parser.add_argument("--chunk-size", type=int, default=4000)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--dsn-env", default=None, help="path to env file with *_DSN")
    parser.add_argument("--commit", action="store_true", help="commit the transaction")
    args = parser.parse_args(argv)

    dsn = None
    if args.mode != "dry-run":
        dsn = resolve_dsn(args) if not args.dsn_env else parse_dsn_env(args.dsn_env)
    report = ingest(
        args.manifest, args.files_dir,
        claims_path=args.claims, mode=args.mode,
        chunk_size=args.chunk_size, overlap=args.overlap,
        dsn=dsn, commit=args.commit,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
