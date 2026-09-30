"""Generate a SYNTHETIC lock receipt -- for proving gate LOGIC, never the contract.

WHY THIS IS MARKED
  A hand-written receipt that satisfies the amended contract is a fabricated-pass
  artifact: if it were passed to `--receipt` and printed LOCKED, a later reader
  could mistake it for a measurement. It therefore carries
  "synthetic_control": true, and `contract_lock_check.py` routes any such receipt
  to LOCK_VERDICT: FIXTURE_CHECKED (exit 2) -- a verdict that can never be the
  lock. The only LOCKED that counts comes from `--live` on the target.

  Do NOT commit the generated file. It exists to prove the gate fails closed and
  passes when it should, and nothing else.

Usage:
    python make_synthetic_lock_fixture.py /path/to/out.json
"""
from __future__ import annotations

import json
import sys
import time

SYNTHETIC = {
    "synthetic_control": True,
    "_fixture_note": (
        "SYNTHETIC. Proves the gate's logic only. NOT evidence of the contract. "
        "Never report a FIXTURE_CHECKED as a lock."
    ),
    "gpu": "NVIDIA GeForce RTX 5090",
    "torch": "2.12.0+cu130",
    "D": 65536,
    # clause (a): the shape a REAL receipt has -- marker PARSED from child stdout
    "smoke_exit_code": 0,
    "smoke_seconds": 41.2,
    "smoke_output_bytes": 3300,
    "smoke_stdout_sha256":
        "9f2b7c4e1a6d8035fb21e0c4a7d3965b8c1f4e2a9d6b0378c5a1e8f2b4d70936",
    "smoke_marker_seen_in_output": True,
    "smoke_marker_count": 1,
    "checkpoint_status": "LOADED",
    "trained_decoder_active": True,
    "fail_closed_generic": True,
    # clause (b)
    "perceive_norm": 1.0,
    "perceive_norm_err": 0.0,
    "perceive_finite": True,
    "perceive_shape": [8192, 8],
    # clause (c) -- the measured values the amended thresholds derive from
    "perceive_1step_us": 465.6, "perceive_1step_std_us": 3.1,
    "act_step_us": 3445.5,      "act_step_std_us": 21.0,
    "encode_1step_us": 251.0,   "encode_1step_std_us": 2.4,
}


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_lock_fixture.json"
    rec = dict(SYNTHETIC)
    rec["measured_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(f"synthetic fixture -> {out}")
    print("  expected gate verdict: FIXTURE_CHECKED (exit 2) -- NEVER 'LOCKED'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
