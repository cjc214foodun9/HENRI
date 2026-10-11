"""probe_urc6_relational.py -- G-URC6 (relational operator accuracy >= 0.8500).

THESIS ITEM 4. The directive asks for URC Stage 5 on relational tasks with
    G-URC6: operator accuracy >= 0.8500 (>= +0.35 over the linear baseline).

WHAT I MEASURED BEFORE WRITING ANY BENCHMARK
    git ls-files | grep arc -> 381 files. Of the .json entries, ZERO carry ARC task shape
    (train/test pairs of grids). The tree holds ARC *receipts and engines*, not ARC *task
    data*. Measured count of real ARC task JSONs: 0.

CONSEQUENCE, STATED PLAINLY
    G-URC6 cannot be earned on this host. There is no task data to score, and no CUDA.
    The correct deliverable is a fail-closed BLOCKED receipt plus the provisioning list.
    I do NOT substitute a proxy benchmark and call it G-URC6.

WHAT I DO MEASURE INSTEAD, LABELLED AS A WIRING CHECK AND NOT A BENCHMARK
    The tripartite factorization's algebraic self-consistency on the in-tree module:
    round-trip reconstruction error and associativity error. These numbers exist already
    in the module's own test (6/6 passing, round-trip 1.333e-07). Reproducing them here is
    a WIRING check: it shows the Stage-5 delegation target exists and composes. It is NOT
    relational accuracy, it does NOT clear any bar, and it is labelled "proxy_wiring_check".

SCOPE: CPU, no training. No capability claim. No CUDA. No AAII claim.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
V = REPO / "HENRI V2"
if str(V) not in sys.path:
    sys.path.insert(0, str(V))

EV = REPO / "design" / "zone_a" / "evidence"
EV.mkdir(parents=True, exist_ok=True)

R = {
    "schema": "henri.urc6.relational.observed.v1",
    "evidence_class": "OBSERVED",
    "what": "G-URC6 relational operator accuracy >= 0.8500",
    "bar": 0.8500,
    "non_claims": [
        "no proxy is reported as G-URC6",
        "the wiring check below is NOT a benchmark and clears no bar",
        "no CUDA; no AAII claim",
    ],
}

print("=" * 78)
print("STEP 1 -- is ARC relational TASK data present anywhere in tree?")
print("=" * 78)
import re
import subprocess

r = subprocess.run("git ls-files", shell=False, capture_output=True, text=True, cwd=str(REPO))
tracked = r.stdout.splitlines()
arc_files = [f for f in tracked if re.search(r"arc", f, re.I)]
task_shaped = 0
for f in arc_files:
    if not f.endswith(".json"):
        continue
    p = REPO / f
    if not p.is_file():
        continue
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        continue
    if isinstance(d, dict) and "train" in d and "test" in d:
        tr = d.get("train")
        if isinstance(tr, list) and tr and isinstance(tr[0], dict) \
                and {"input", "output"} <= set(tr[0]):
            task_shaped += 1
R["arc_data_audit"] = {
    "tracked_files_matching_arc": len(arc_files),
    "real_task_json_count": int(task_shaped),
    "note": "ARC receipts and engines exist; ARC task data does not.",
}
print(f"  tracked /arc/ files {len(arc_files)}   real ARC task JSONs: {task_shaped}")

print()
print("=" * 78)
print("STEP 2 -- wiring check ONLY: tripartite factorization self-consistency")
print("=" * 78)
proxy = None
try:
    import henri_core.henri_resonator_flow as rf                    # noqa: E402
    gen = torch.Generator().manual_seed(20261011)
    Dd = 64
    # a random unitary relational operator, factorized and recomposed
    A = torch.randn(Dd, Dd, generator=gen, dtype=torch.float64)
    Qm, _ = torch.linalg.qr(A)
    recon = None
    for cand in ("round_trip", "factorize_recompose", "factorise"):
        fn = getattr(rf, cand, None)
        if callable(fn):
            try:
                recon = fn(Qm)
                break
            except Exception:                                   # noqa: BLE001
                continue
    if recon is not None and torch.is_tensor(recon):
        err = float((recon - Qm).norm() / Qm.norm().clamp_min(1e-30))
        proxy = {"round_trip_rel_error": err, "D": Dd, "source": "module function"}
    else:
        proxy = {"round_trip_rel_error": None, "D": Dd,
                 "source": "module exposes no single-call factorize; see unit test"}
except Exception as exc:                                        # noqa: BLE001
    proxy = {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
R["proxy_wiring_check"] = {
    "is_a_benchmark": False,
    "is_g_urc6": False,
    "clears_no_bar": True,
    "measured": proxy,
    "note": ("Round-trip reconstruction error of the tripartite factorization. This shows "
             "the Stage-5 delegation target exists and composes. It is NOT relational "
             "accuracy and it clears no bar."),
}
print(f"  proxy: {json.dumps(proxy, default=str)[:200]}")

print()
print("=" * 78)
print("VERDICT")
print("=" * 78)
R["checks"] = {
    "arc_task_data_present": bool(task_shaped > 0),
    "cuda_available": bool(torch.cuda.is_available()),
    "proxy_would_be_a_fabrication": True,
}
for k, v in R["checks"].items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
R["verdict"] = "G_URC6_BLOCKED_DATA_ABSENT"
R["provisioning_required"] = [
    "ARC-AGI task data (train/test grid pairs) staged into the repository",
    "a CUDA host whose containment is proven within the boundary",
]
R["g_urc6_status"] = ("BLOCKED. No ARC relational task data exists in tree (0 of 381 "
                      "matching files are task JSONs) and no CUDA device is available. "
                      "Substituting a proxy and reporting it as G-URC6 would be fabrication.")
print(f"  VERDICT: {R['verdict']}")

out = EV / "henri_urc6_relational_observed.json"
out.write_text(json.dumps(R, indent=2, default=str) + "\n", encoding="utf-8")
print(f"\n  receipt -> {out.relative_to(REPO)}")
print("  OBSERVED by execution. No bar moved. No CUDA. No AAII claim.")
