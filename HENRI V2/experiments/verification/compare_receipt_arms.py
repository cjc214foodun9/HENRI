#!/usr/bin/env python3
"""Compare two evaluate_60_task_koopman_gap receipts arm-by-arm.

Used to decide whether a committed receipt is EXECUTION-DERIVED: re-run the
harness, then compare the fresh receipt against the committed one. Identical arm
tables mean the committed numbers came from running the committed code.

    python compare_receipt_arms.py A.json B.json
Exit 0 if arm tables match, 1 otherwise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TOL = 1e-9


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    b = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

    ka, kb = set(a["arms"]), set(b["arms"])
    if ka - kb:
        print("  arms only in A:", sorted(ka - kb))
    if kb - ka:
        print("  arms only in B:", sorted(kb - ka))

    ndiff = 0
    for k in sorted(ka & kb):
        ha = a["arms"][k]["held_out_mean"]
        hb = b["arms"][k]["held_out_mean"]
        ca = a["arms"][k]["in_sample_ceiling_mean"]
        cb = b["arms"][k]["in_sample_ceiling_mean"]
        dh, dc = abs(ha - hb), abs(ca - cb)
        bad = dh > TOL or dc > TOL
        ndiff += bad
        print(f"  {k:26s} A held={ha:+.9f}  B held={hb:+.9f}  d={dh:.2e}"
              + ("   <-- DIFFERS" if bad else ""))

    same_set = not (ka ^ kb)
    print(f"  arms_differing={ndiff}")
    print("  STRICT_IDENTICAL:", ndiff == 0 and same_set)
    return 0 if (ndiff == 0 and same_set) else 1


if __name__ == "__main__":
    sys.exit(main())
