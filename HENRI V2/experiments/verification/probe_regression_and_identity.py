#!/usr/bin/env python3
"""REGRESSION PROBE v2: extract the ACTUAL error, and settle file identity.

THE REGRESSION (measured, payload_keys census)
    pre-fix  bridge OFF : 60 payloads, keys = {gate_status, detail}
                          (the width mismatch, correctly reported as UNAVAILABLE)
    pre-fix  bridge ON  : 60 payloads, keys = {delta_axiom, delta_epistemic, hard_vetoed}
    post-fix bridge OFF :  0 payloads under key `sagnac_veto`
                          64 dicts somewhere carrying an `error` key
    The run itself completed (steps to 63, GAME_OVER, scorecard written) and
    `[opine]` printed 64 times, so the OPINE block executed.

    v1 collected only KEY NAMES for the error dicts, not the messages, so the cause
    was not captured. This version prints the messages verbatim, which is the only
    way to name the cause.

ALSO SETTLES: which file are the advisor renderings reading?
    Advisor renderings consistently report production_arc_run.py as 3426 lines /
    136161 bytes with a `_name == "SagnacGateUnavailable"` handler and extra fields.
    My two measurement passes reported 3486/197175 (worktree) and 3266/182483 (main).
    Neither is 3426. There is a FOURTH copy I never measured:
    .worktrees/basal-syncytium/HENRI V2/production_arc_run.py
    All four are hashed here. A match at 3426 would resolve the discrepancy as a
    DIFFERENT TREE rather than fabrication.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

TOP = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM")
EXP = Path(r"C:\Users\chan\HENRI_telemetry_exports")

print("=" * 78)
print("PART 1: the ACTUAL error message in the post-fix telemetry")
print("=" * 78)
files = sorted(EXP.glob("production_run_*.jsonl"),
               key=lambda p: p.stat().st_mtime, reverse=True)[:3]
for f in files:
    msgs, keysets, arms = collections.Counter(), collections.Counter(), collections.Counter()
    opine_keys = collections.Counter()
    n = 0

    def walk(o):
        if isinstance(o, dict):
            e = o.get("error")
            if isinstance(e, str):
                msgs[e[:260]] += 1
            if "sagnac_veto" in o and isinstance(o["sagnac_veto"], dict):
                keysets[tuple(sorted(o["sagnac_veto"].keys()))] += 1
            if "opine_info" in o and isinstance(o["opine_info"], dict):
                for k in o["opine_info"]:
                    opine_keys[k] += 1
                b = o["opine_info"].get("macro_option_branch")
                arms[str(b)] += 1
            if o.get("veto_error") is not None:
                msgs["VETO_ERROR: " + str(o["veto_error"])[:220]] += 1
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        n += 1
        try:
            walk(json.loads(line))
        except Exception:
            pass

    print(f"\n{f.name}  records={n}")
    print(f"  sagnac_veto keysets : {dict(keysets) or '{}'}")
    print(f"  opine_info keys     : {dict(opine_keys) or '{}'}")
    print(f"  macro_option_branch : {dict(arms) or '{}'}")
    if msgs:
        print("  ERROR MESSAGES:")
        for k, v in msgs.most_common(4):
            print(f"    {v:>3}x  {k}")
    else:
        print("  no error strings found")

print()
print("=" * 78)
print("PART 2: file identity across ALL copies")
print("=" * 78)
copies = [
    TOP / ".worktrees" / "semantic-backbone" / "HENRI V2" / "production_arc_run.py",
    TOP / ".worktrees" / "basal-syncytium" / "HENRI V2" / "production_arc_run.py",
    TOP / ".worktrees" / "_baseline_ctrl" / "HENRI V2" / "production_arc_run.py",
    TOP / "HENRI V2" / "production_arc_run.py",
]
for p in copies:
    if not p.exists():
        print(f"  ABSENT  ...{str(p)[len(str(TOP)):]}")
        continue
    b = p.read_bytes()
    t = b.decode("utf-8", errors="replace")
    try:
        rel = p.relative_to(TOP)
    except Exception:
        rel = p
    print(f"  lines={len(t.splitlines()):>5} bytes={len(b):>7} "
          f"sha16={hashlib.sha256(b).hexdigest()[:16]} "
          f"isinst={t.count('isinstance(_veto_exc')} "
          f"namecmp={t.count('SagnacGateUnavailable\"')} "
          f"cand_elems={t.count('candidate_elems')} "
          f"nc8192={t.count('num_channels=8192')}")
    print(f"          {rel}")
print()
print("  ADVISOR RENDERINGS claim: lines=3426 bytes=136161 "
      "(with namecmp=1 and cand_elems>0)")
print("  A copy matching 3426/136161 would confirm 'different tree' rather than "
      "'fabricated'.")

print()
print("=" * 78)
print("PART 3: contested module in every tree, and efe_planner widths")
print("=" * 78)
for base in (TOP / ".worktrees" / "semantic-backbone" / "HENRI V2",
             TOP / ".worktrees" / "basal-syncytium" / "HENRI V2",
             TOP / "HENRI V2"):
    cm = base / "synthesize_sagnac_guarded_macro_options.py"
    ep = base / "efe_planner.py"
    etxt = ep.read_text(encoding="utf-8", errors="ignore") if ep.exists() else ""
    try:
        rel = base.relative_to(TOP)
    except Exception:
        rel = base
    print(f"  {rel}")
    print(f"    contested_module exists = {cm.exists()}")
    print(f"    efe_planner lines={len(etxt.splitlines())} "
          f"def_field_to_wave={etxt.count('def field_to_wave')} "
          f"mentions={etxt.count('field_to_wave')}")
