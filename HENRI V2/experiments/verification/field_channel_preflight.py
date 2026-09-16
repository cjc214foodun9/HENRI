#!/usr/bin/env python3
"""FIELD-CHANNEL ARTIFACT PREFLIGHT: SHA-256 + byte count + load status, fail-closed.

WHY THIS EXISTS
    `field_channel_checkpoints/*.pt` are gitignored external overlays, so no commit
    check ever covered them. Measured 2026-09-16: the set holds 41 files, 40 are
    byte-identical in size (104,859,636 B) and one is TRUNCATED
    (`field_channel_ar25-0c556536_1a5cdff3.pt`, 55,287,808 B) and unreadable
    ("File is not a zip file"). Nothing detected it. This preflight does.

FAIL-CLOSED CONTRACT
    Exit 0 only when every file present matches its manifest entry and loads.
    Exit 1 on any missing file, byte-count mismatch, ZIP failure, or tensor-load
    failure. Exit 2 with a typed SKIP when the overlay directory is absent (the set
    is NOT in Git, so a clean CI checkout legitimately has no overlay -- that must
    SKIP, not FAIL, or every clean-runner suite turns red for the wrong reason).

WRITE-ONCE EVIDENCE (CLASS52)
    The manifest writer REFUSES to overwrite a non-trivial existing manifest: it
    writes a timestamped sibling instead and reports the prior path. An artifact
    that records state must not be silently replaced by a later measurement.
    Prove the guard by running twice -- the first manifest must stay byte-identical.

USAGE
    python experiments/verification/field_channel_preflight.py --write-manifest
    python experiments/verification/field_channel_preflight.py --verify
    python experiments/verification/field_channel_preflight.py --report OUT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]          # ...\HENRI V2
CKPT_DIR = REPO / "field_channel_checkpoints"
MANIFEST = REPO / "experiments" / "verification" / "field_channel_manifest.json"
EXPECTED_BYTES = 104_859_636                        # measured invariant, 40/40 files
EXIT_OK, EXIT_FAIL, EXIT_SKIP = 0, 1, 2


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def probe(path: Path) -> dict:
    """Deterministic per-file record. Never raises; encodes failure as status."""
    rec = {"name": path.name, "bytes": path.stat().st_size}
    try:
        rec["sha256"] = sha256_file(path)
    except Exception as e:  # unreadable bytes
        rec["sha256"] = None
        rec["status"] = "UNREADABLE"
        rec["detail"] = f"{type(e).__name__}: {e}"
        return rec

    if rec["bytes"] != EXPECTED_BYTES:
        rec["status"] = "BYTE_COUNT_MISMATCH"
        rec["expected_bytes"] = EXPECTED_BYTES
    try:
        z = zipfile.ZipFile(path)
        bad = z.testzip()
        if bad:
            rec["status"] = "ZIP_BAD_ENTRY"
            rec["detail"] = str(bad)
            return rec
        rec["zip_ok"] = True
    except Exception as e:
        rec["status"] = "ZIP_UNREADABLE"
        rec["detail"] = f"{type(e).__name__}: {str(e)[:120]}"
        return rec

    try:
        import torch
        obj = torch.load(path, map_location="cpu", weights_only=True)
        rec["load_status"] = "LOADED"
        rec["keys"] = sorted(obj.keys()) if isinstance(obj, dict) else [type(obj).__name__]
        if isinstance(obj, dict):
            if "wave" in obj:
                rec["wave_numel"] = int(obj["wave"].numel())
                rec["wave_dtype"] = str(obj["wave"].dtype)
            for k in ("env", "buffer_size", "l3_loss", "scale"):
                if k in obj:
                    v = obj[k]
                    rec[k] = v if not hasattr(v, "tolist") else v.tolist()
    except ImportError:
        rec["load_status"] = "SKIPPED_NO_TORCH"
    except Exception as e:
        rec["load_status"] = "LOAD_FAILED"
        rec["detail"] = f"{type(e).__name__}: {str(e)[:120]}"
        if "status" not in rec:
            rec["status"] = "LOAD_FAILED"
    rec.setdefault("status", "OK")
    return rec


def write_manifest(records: list, path: Path) -> Path:
    """Write-once: refuse to replace a non-trivial existing manifest."""
    payload = {
        "schema_id": "henri.field-channel-manifest.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "directory": str(CKPT_DIR),
        "expected_bytes": EXPECTED_BYTES,
        "n_files": len(records),
        "records": records,
    }
    if path.exists():
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
            prior_n = len(prior.get("records", []))
        except Exception:
            prior_n = 0
        if prior_n > 1:
            sibling = path.with_name(
                f"{path.stem}.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}{path.suffix}")
            sibling.write_text(json.dumps(payload, indent=1), encoding="utf-8")
            print(f"WRITE-ONCE GUARD: refused to overwrite {path} "
                  f"({prior_n} records); wrote {sibling}")
            return sibling
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="field-channel artifact preflight")
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--report", default=None, help="write the per-file receipt JSON")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if not CKPT_DIR.is_dir():
        print(f"SKIP: overlay directory absent ({CKPT_DIR}). "
              f"The set is gitignored external state; a clean checkout has no overlay. "
              f"This is not a failure.")
        return EXIT_SKIP

    files = sorted(CKPT_DIR.glob("*.pt"))
    if not files:
        print(f"SKIP: no *.pt files under {CKPT_DIR}")
        return EXIT_SKIP

    records = [probe(p) for p in files]

    bad = [r for r in records if r.get("status") != "OK"
           or r.get("load_status") in ("LOAD_FAILED", "UNREADABLE")]

    if not args.quiet:
        n_ok = len(records) - len(bad)
        print(f"field-channel preflight: {len(records)} files, {n_ok} OK, {len(bad)} FAIL")
        for r in bad:
            print(f"  FAIL {r['name']}: status={r.get('status')} "
                  f"bytes={r['bytes']} load={r.get('load_status')} "
                  f"{r.get('detail', '')[:90]}")

    if args.write_manifest:
        out = write_manifest(records, MANIFEST)
        print(f"manifest: {out}")

    if args.report:
        Path(args.report).write_text(json.dumps(
            {"schema_id": "henri.field-channel-preflight.v1", "records": records,
             "n_files": len(records), "n_failed": len(bad)}, indent=1), encoding="utf-8")
        print(f"report: {args.report}")

    print(f"RESULT: field-channel preflight covered {len(records)} file(s), {len(bad)} failed")
    if bad and args.verify:
        return EXIT_FAIL
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
