"""Carrier K5 (TZCSM) — per-domain manifest and coverage-receipt reader.

Schema (K5 section 3/7):
  manifest row: source_id, domain, title, origin, sha256, license, retrieved_utc
  receipt: manifest_count, chunk_count, claim_count, verified_count,
           abstained_count, rejected_count, last_ingest_utc

A domain without a receipt row is NOT_EVALUATED. This module never writes
corpus rows; it only builds receipts from the K5 tables and validates
manifest files. Read-only by construction.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone
from typing import Any

DOMAINS = [
    "formal_science", "natural_science", "medicine", "engineering",
    "computing", "law", "economics", "humanities", "language", "arts",
    "education", "practical_skills", "safety", "governance",
]

MANIFEST_FIELDS = ("source_id", "domain", "title", "origin", "sha256",
                   "license", "retrieved_utc")


class ManifestError(ValueError):
    """Invalid manifest row."""


def validate_manifest_row(row: dict[str, Any]) -> None:
    for field in MANIFEST_FIELDS:
        if field not in row or not str(row[field]).strip():
            raise ManifestError(f"manifest row missing field: {field}")
    if row["domain"] not in DOMAINS:
        raise ManifestError(f"unknown domain: {row['domain']}")
    digest = str(row["sha256"]).lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ManifestError("sha256 must be a 64-character hex digest")
    try:
        datetime.fromisoformat(str(row["retrieved_utc"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestError(f"retrieved_utc not ISO-8601: {row['retrieved_utc']}") from exc


def load_manifest(path: str | pathlib.Path) -> list[dict[str, Any]]:
    """Load and validate a per-domain JSONL manifest. Empty file = []."""
    rows = []
    p = pathlib.Path(path)
    if not p.is_file():
        raise ManifestError(f"manifest not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{p}:{line_no} not JSON: {exc}") from exc
            validate_manifest_row(row)
            rows.append(row)
    return rows


def build_coverage_receipt(cur, domain: str) -> dict[str, Any]:
    """Coverage receipt for one domain from the live K5 tables.

    Any domain without a receipt row is NOT_EVALUATED. cur is a psycopg
    cursor on a Zone C database with the K5 additive schema applied.
    """
    cur.execute(
        """
        SELECT
          (SELECT count(*) FROM domain_source_manifest WHERE domain = %s),
          (SELECT count(*) FROM corpus_chunks WHERE domain = %s),
          (SELECT count(*) FROM world_claims WHERE domain = %s),
          (SELECT count(*) FROM world_claims WHERE domain = %s
             AND verification_status = 'VERIFIED'),
          (SELECT count(*) FROM world_claims WHERE domain = %s
             AND strpos(verification_status, 'ABSTAIN') = 1),
          (SELECT count(*) FROM world_claims WHERE domain = %s
             AND strpos(verification_status, 'REJECT') = 1),
          (SELECT max(ingested_utc) FROM corpus_chunks WHERE domain = %s)
        """,
        (domain,) * 7,
    )
    row = cur.fetchone()
    return {
        "domain": domain,
        "manifest_count": int(row[0] or 0),
        "chunk_count": int(row[1] or 0),
        "claim_count": int(row[2] or 0),
        "verified_count": int(row[3] or 0),
        "abstained_count": int(row[4] or 0),
        "rejected_count": int(row[5] or 0),
        "last_ingest_utc": row[6].isoformat() if row[6] else None,
        "status": "RECEIPTED" if int(row[1] or 0) > 0 else "NOT_EVALUATED",
    }


if __name__ == "__main__":
    import sys

    # Deterministic receipt for a manifest file (no DB required):
    #   python zone_c_world_knowledge_manifest.py <path-to-manifest.jsonl>
    rows = load_manifest(sys.argv[1])
    by_domain: dict[str, int] = {}
    for r in rows:
        by_domain[r["domain"]] = by_domain.get(r["domain"], 0) + 1
    print(json.dumps({"rows": len(rows), "by_domain": by_domain}, indent=2, sort_keys=True))
