#!/usr/bin/env python3
"""SETTLE BOTH OPEN QUESTIONS IN ONE PASS.

Q1 (regression): after the num_channels fix, veto payloads went 60 -> 0, replaced by
    64 dicts carrying an `error` key. What is the ACTUAL message? v1 of the probe
    collected only key NAMES, so the cause was never captured.

Q2 (identity): advisor renderings show production_arc_run.py at 3426 lines /
    136161 bytes with a name-comparison handler and fields I did not write
    (candidate_elems, veto_stage, shape_candidate, _branch_used, macro_option_log).
    My two measured trees are 3486/197175 (worktree) and 3266/182483 (main).
    NEITHER is 3426. A FOURTH copy exists that I have never measured:
        .worktrees/basal-syncytium/HENRI V2/production_arc_run.py
    If that copy IS 3426/136161, my earlier "fabricated" label is WRONG and the
    correct reading is "different tree" -- which I must correct openly. If it is not,
    the rendering matches no file on disk.

    This matters beyond bookkeeping: if basal-syncytium already contains a MORE
    DEVELOPED version of this block (with shape diagnostics and a branch log), then
    work I am doing now may already exist there, and that is a fact the user needs.

Method for Q2: hash EVERY copy found by an explicit walk (not a shell glob), and
report the distinguishing strings, so the comparison is byte-level and complete.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
from pathlib import Path

TOP = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM")
EXP = Path(r"C:\Users\chan\HENRI_telemetry_exports")

# ==================================================================== Q1
print("=" * 78)
print("Q1: the ACTUAL post-fix error message")
print("=" * 78)
files = sorted(EXP.glob("production_run_*.jsonl"),
               key=lambda p: p.stat().st_mtime, reverse=True)[:3]
for f in files:
    msgs = collections.Counter()
    sv_keys = collections.Counter()
    at = collections.Counter()
    n = 0

    def walk(o, path="r"):
        if isinstance(o, dict):
            if isinstance(o.get("error"), str):
                msgs[o["error"][:210]] += 1
                at[path.split(".")[-1] if "." in path else path] += 1
            if isinstance(o.get("sagnac_veto"), dict):
                sv_keys[tuple(sorted(o["sagnac_veto"].keys()))] += 1
            if isinstance(o.get("veto_error"), str):
                msgs["VETO_ERROR " + o["veto_error"][:200]] += 1
            for k, v in o.items():
                walk(v, path + "." + str(k))
        elif isinstance(o, list):
            for v in o[:5]:
                walk(v, path + "[]")

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
    print(f"  sagnac_veto keysets: {dict(sv_keys) or '{}'}")
    if msgs:
        print("  ERROR MESSAGES (this names the cause):")
        for k, v in msgs.most_common(6):
            print(f"    {v:>3}x  {k}")
    else:
        print("  (no error strings)")

# ==================================================================== Q2
print()
print("=" * 78)
print("Q2: EVERY production_arc_run.py copy, hashed")
print("=" * 78)
copies = []
for root, dirs, names in os.walk(TOP):
    dirs[:] = [d for d in dirs if d != ".git"]
    for nm in names:
        if nm == "production_arc_run.py":
            copies.append(Path(root) / nm)
copies.sort()

TARGET_LINES, TARGET_BYTES = 3426, 136161
matched = None
for p in copies:
    try:
        b = p.read_bytes()
    except Exception as e:  # noqa: BLE001
        print(f"  UNREADABLE {p}: {type(e).__name__}")
        continue
    t = b.decode("utf-8", errors="replace")
    rel = str(p)[len(str(TOP)):].lstrip("\\/")
    is_match = (len(t.splitlines()) == TARGET_LINES and len(b) == TARGET_BYTES)
    if is_match:
        matched = rel
    print(f"  lines={len(t.splitlines()):>5} bytes={len(b):>7} "
          f"sha16={hashlib.sha256(b).hexdigest()[:16]} "
          f"isinst={t.count('isinstance(_veto_exc')} "
          f"namecmp={t.count('name == \"SagnacGateUnavailable\"')} "
          f"cand_elems={t.count('candidate_elems')} "
          f"branch_used={t.count('_branch_used')} "
          f"nc8192={t.count('num_channels=8192')}")
    print(f"        {'<<< MATCHES THE RENDERING' if is_match else ''} {rel}")

print()
print(f"  copies found: {len(copies)}")
print(f"  rendering target: lines={TARGET_LINES} bytes={TARGET_BYTES}")
print(f"  MATCHED FILE: {matched if matched else 'NONE -- the rendering matches no file on disk'}")

# contested module
print()
print("=" * 78)
print("Q2b: contested module and efe_planner def counts, all trees")
print("=" * 78)
for base in sorted({p.parent for p in copies}):
    cm = base / "synthesize_sagnac_guarded_macro_options.py"
    ep = base / "efe_planner.py"
    rel = str(base)[len(str(TOP)):].lstrip("\\/")
    etxt = ep.read_text(encoding="utf-8", errors="ignore") if ep.exists() else ""
    print(f"  {rel}")
    print(f"    contested_module_exists={cm.exists()}")
    if etxt:
        print(f"    efe_planner lines={len(etxt.splitlines())} "
              f"def_field_to_wave={etxt.count('def field_to_wave')}")

print()
print("=" * 78)
print("VERDICT")
print("=" * 78)
if matched:
    print(f"  The advisor rendering reads a REAL file: {matched}")
    print("  => my earlier 'fabricated' label was WRONG; the correct label is")
    print("     'a different worktree'. CORRECT THIS OPENLY.")
else:
    print("  No file on disk matches the rendering's line/byte counts.")
    print("  => the rendering does not correspond to any HENRI tree here.")
