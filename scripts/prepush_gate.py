"""D6: gate-first promotion. Runs the fast gate set and refuses on any failure.

WHY
    A prior pipeline pushed to main while its own regressions returned rc=1,
    because it spawned a bare `python` that had no torch. This runner uses
    sys.executable (the interpreter that is actually running), records every
    return code to a manifest, and exits nonzero if any gate fails.

INSTALL (local, unversioned -- NOT a substitute for CI)
    git config core.hooksPath scripts/hooks
    # or copy to .git/hooks/pre-push
    The hook is local to this clone. A local hook is NOT a CI gate and does not
    protect a merge performed elsewhere. State that plainly.

USAGE
    python scripts/prepush_gate.py            # run, print, write manifest
    python scripts/prepush_gate.py --quiet     # rc only
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
V = os.path.join(REPO, "HENRI V2")
MANIFEST = os.path.join(REPO, "design", "zone_a", "evidence",
                        "prepush_gate_manifest.json")

# (name, argv relative to V) -- fast gates only. The full pytest suite is too
# slow for a pre-push hook and would be bypassed in practice.
GATES = [
    ("verify_tri_model", ["henri_core/verify_tri_model.py"]),
    ("tri_model_19", ["tests/unit/test_henri_tri_model.py"]),
    ("gate_fix_selfcheck", ["henri_core/test_gate_fixes_selfcheck.py"]),
    ("harness_hygiene", ["henri_core/test_harness_hygiene.py"]),
    ("q4_wiring", ["henri_core/test_q4_wiring.py"]),
    ("veto_coupling", ["henri_core/test_veto_coupling.py"]),
    ("q4_discriminative", ["henri_core/exp_q4_discriminative.py"]),
    ("q4_margin", ["henri_core/exp_q4_margin.py"]),
    ("q4_seed_transfer", ["henri_core/exp_q4_seed_transfer.py"]),
    ("phase1_veto", ["henri_core/exp_novelty_veto.py"]),
]


def run_gate(name, argv):
    r = subprocess.run([sys.executable, *argv], cwd=V,
                       capture_output=True, text=True, errors="replace")
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    return {"name": name, "rc": int(r.returncode), "tail": tail[:200],
            "interpreter": sys.executable}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    results = [run_gate(n, argv) for n, argv in GATES]
    failed = [r for r in results if r["rc"] != 0]
    man = {
        "schema": "henri.prepush.gate.manifest.v1",
        "when": _dt.datetime.now().isoformat(timespec="seconds"),
        "interpreter": sys.executable,
        "cwd": V,
        "gates": results,
        "n_gates": len(results), "n_failed": len(failed),
        "all_green": not failed,
        "head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                               capture_output=True, text=True).stdout.strip(),
    }
    man["manifest_sha256"] = hashlib.sha256(
        json.dumps(man, sort_keys=True).encode()).hexdigest()
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(man, indent=1))

    if not a.quiet:
        print(f"D6 gate-first promotion: {man['n_gates']} gates, "
              f"{man['n_failed']} failed  ({man['head'][:9]})")
        for r in results:
            print(f"  rc={r['rc']:<2} {r['name']:<20} {r['tail'][:80]}")
        print(f"  manifest: {MANIFEST}")
        print(f"  {'ALL GREEN -- push permitted' if man['all_green'] else 'BLOCKED -- do not push'}")
    return 0 if man["all_green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
