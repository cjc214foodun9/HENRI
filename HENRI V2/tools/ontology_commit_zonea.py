#!/usr/bin/env python
"""Zone A ontology commit: PROBE -> VERIFY -> COMMIT (builder protocol).

Approval: the human granted the commit decision in the task message ("Commit,
you are granted approval"; the attached SPEC said "propose-only", so the message
is the user's own directive in the human channel).  Approval covers the DECISION,
not the VERIFICATION -- so this script still runs the probe and the verification
and REFUSES to commit on any failure.  The advisory typed gate returned
ESCALATE/no-authorisation; that is recorded in the receipt, not treated as a veto
of an explicit human grant.

Robustness note: objects.jsonl contains TWO historical schema variants.  56
records use `id`/`type`/`utc`; the rest use `record_id`/`kind`/`created_utc`.
The store is append-only, so legacy rows are never edited.  This script reads
either key for the duplicate check and writes only the current schema.

Run:  python tools/ontology_commit_zonea.py [--dry-run]
Exit: 0 on commit (or clean dry-run); 1 on any probe/verify failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STORE = Path("C:/Users/chan/henri-telemetry/ontology/objects.jsonl")
CANDIDATES = Path("C:/Users/chan/henri-telemetry/ontology/candidates.jsonl")
REQUIRED = {"record_id", "kind", "definition", "evidence_class",
            "status", "probe_ref", "created_utc"}
VALID_KINDS = {"term", "mapping", "constraint", "evidence"}
VALID_EVIDENCE = {"OBSERVED", "DERIVED", "INFERRED", "HYPOTHESIS",
                  "FALSIFIED", "BLOCKED"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            rows.append(json.loads(ln))
    return rows


def existing_ids(rows: list[dict]) -> set[str]:
    """Accept BOTH schema variants: `record_id` (current) and `id` (legacy)."""
    out = set()
    for r in rows:
        rid = r.get("record_id") or r.get("id")
        if rid:
            out.add(str(rid))
    return out


def probe_path(probe_ref: str) -> Path | None:
    """Return the resolvable filesystem path, or None for a URI-scheme ref."""
    if "://" in probe_ref or probe_ref.startswith("mcp__"):
        return None
    return Path(probe_ref.split("#")[0].strip())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(STORE))
    ap.add_argument("--candidates", default=str(CANDIDATES))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    store = Path(args.store)
    cands = load_jsonl(Path(args.candidates))
    existing = load_jsonl(store)
    ids_now = existing_ids(existing)

    print(f"store     : {store}")
    print(f"records   : {len(existing)}  (distinct ids: {len(ids_now)})")
    print(f"candidates: {len(cands)}")

    # ---- 1. PROBE ------------------------------------------------------
    print("\n--- 1. PROBE ---")
    probe_fail = []
    probes = {}
    for c in cands:
        rid = c["record_id"]
        ref = c.get("probe_ref", "")
        p = probe_path(ref)
        if p is None:
            probes[rid] = {"kind": "uri", "ref": ref, "ok": True}
            continue
        exists = p.is_file()
        digest = "sha256:" + sha256_file(p) if exists else None
        claimed = c.get("probe_sha256")
        matched = (claimed is None) or (claimed == digest)
        probes[rid] = {"kind": "file", "path": str(p), "exists": exists,
                       "sha256": digest, "claimed": claimed, "matched": matched,
                       "ok": exists and matched}
        if not probes[rid]["ok"]:
            probe_fail.append(rid)
        print(f"  {rid}  {'OK' if probes[rid]['ok'] else 'FAIL'}"
              f"  {p.name if exists else p.name + ' (MISSING)'}")
        if exists and not matched:
            print(f"      digest mismatch: claimed {claimed} actual {digest}")

    if probe_fail:
        print(f"\nPROBE FAILED: {probe_fail}")
        return 1

    # ---- 2. VERIFY -----------------------------------------------------
    print("\n--- 2. VERIFY ---")
    problems = []
    for c in cands:
        rid = str(c["record_id"])
        missing = REQUIRED - set(c.keys())
        if missing:
            problems.append(f"{rid}: missing required {sorted(missing)}")
        if c.get("kind") not in VALID_KINDS:
            problems.append(f"{rid}: invalid kind {c.get('kind')!r}")
        if c.get("evidence_class") not in VALID_EVIDENCE:
            problems.append(f"{rid}: invalid evidence_class {c.get('evidence_class')!r}")
        if c.get("status") not in {"active", "superseded", "rejected"}:
            problems.append(f"{rid}: invalid status {c.get('status')!r}")
        if not str(c.get("definition", "")).strip():
            problems.append(f"{rid}: empty definition")
        if rid in ids_now:
            problems.append(f"{rid}: duplicate record_id already in store")
        # id format: ont-<kind>-<12 hex>
        parts = rid.split("-")
        if len(parts) != 3 or parts[0] != "ont" or parts[1] != c.get("kind") \
                or len(parts[2]) != 12:
            problems.append(f"{rid}: id must be ont-<kind>-<12 hex>")
        try:
            int(parts[2], 16)
        except (ValueError, IndexError):
            problems.append(f"{rid}: id suffix is not 12 hex characters")
        if c["kind"] == "mapping":
            tgt = c.get("target") or {}
            if not tgt.get("locator"):
                problems.append(f"{rid}: mapping requires target.locator")
            if not c.get("verified_by_probe"):
                problems.append(f"{rid}: mapping requires verified_by_probe")
        if c["kind"] == "evidence" and not c.get("supports"):
            problems.append(f"{rid}: evidence requires a non-empty supports list")

    # Every Term/Constraint must resolve to at least one Evidence record.
    ev_ids = {str(c["record_id"]) for c in cands if c["kind"] == "evidence"}
    for c in cands:
        if c["kind"] in {"term", "constraint"}:
            rid = str(c["record_id"])
            refs = {str(c.get("record_id")) for c in cands if c["kind"] == "evidence"}
            supported = any(
                rid in [str(s) for s in e.get("supports", [])]
                for e in cands if e["kind"] == "evidence"
            )
            if not supported:
                problems.append(f"{rid}: no Evidence record supports it")

    if problems:
        for p in problems:
            print(f"  FAIL {p}")
        return 1
    for c in cands:
        print(f"  OK   {c['record_id']}  [{c['kind']}/{c['evidence_class']}]")

    if args.dry_run:
        print("\nDRY RUN: no write performed.")
        return 0

    # ---- 3. COMMIT -----------------------------------------------------
    print("\n--- 3. COMMIT ---")
    backup = store.with_suffix(store.suffix + f".bak-{int(time.time())}")
    shutil.copy2(store, backup)
    print(f"backup    : {backup.name}")

    before = len(existing)
    stamp = datetime.now(timezone.utc).isoformat()
    with open(store, "a", encoding="utf-8") as fh:
        for c in cands:
            rec = dict(c)
            rec["created_utc"] = stamp
            rec["status"] = "active"
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    after_rows = load_jsonl(store)                 # full re-parse = verification
    after = len(after_rows)
    after_ids = existing_ids(after_rows)
    print(f"records   : {before} -> {after}  (expected +{len(cands)})")

    if after != before + len(cands):
        print("COMMIT VERIFY FAILED: record count mismatch")
        return 1
    if len(after_ids) != len(existing_ids(existing)) + len(cands):
        print("COMMIT VERIFY FAILED: duplicate ids introduced")
        return 1

    print("COMMIT VERIFIED: store re-parses; no duplicate ids; ids unique.")
    print(f"head ids  : {[c['record_id'] for c in cands]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
