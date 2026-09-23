#!/usr/bin/env python3
"""SETTLE THE IDENTITY DISPUTE PROPERLY, AND VERIFY THE FLAG-GATED CHANGE.

MY LIKELY ERROR, STATED UP FRONT
    I measured `production_arc_run.py` in four copies, all under
        C:/Users/chan/Desktop/HENRI 7B SWARM
    and found NONE at 3426 lines / 136161 bytes. I labelled the advisor rendering
    "fabricated" on that basis.

    But this project keeps worktrees OUTSIDE that root. An earlier session in this
    conversation ran `git -C "C:/Users/chan/henri-worktrees/aaii-v43" ...`, and my
    `os.walk` never visited `C:/Users/chan/henri-worktrees/`. If a copy there is
    3426/136161 with `_branch_used` / `candidate_elems` / `macro_option_log`, then my
    label was WRONG and the correct reading is "a different worktree outside the
    project root". That is a correction I owe, not a detail.

    This script searches the WHOLE user home for the three contested files and hashes
    every copy, so the answer is byte-level and complete.

ALSO VERIFIED HERE
    The flag-gated macro-resolution change in the semantic-backbone tree:
        _num_channels = int(SCALE["num_blocks"]) if HENRI_MACRO_NUM_CHANNELS==1 else 8192
    Checks: flag present, default arm preserves 8192, SCALE arm present, the store
    consumes the variable, and no bare call-site literal remains.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

HOME = Path(r"C:\Users\chan")
SB = HOME / "Desktop" / "HENRI 7B SWARM" / ".worktrees" / "semantic-backbone" / "HENRI V2"

CONTESTED = [
    "production_arc_run.py",
    "efe_planner.py",
    "synthesize_sagnac_guarded_macro_options.py",
]
MARKERS = ["_branch_used", "macro_option_log", "option_conditioned_gain",
           "candidate_elems", "veto_stage", "shape_candidate", "_name == ",
           "isinstance(_veto_exc", "num_channels=8192", "HENRI_MACRO_NUM_CHANNELS"]

# Roots to search: the whole home, but skip heavy/irrelevant trees.
SKIP = {".git", "node_modules", "__pycache__", ".venv", "venv", "site-packages",
        "hf_cache", "AppData", "miniconda3", "anaconda3", ".cache", "Temp"}
MAX_DEPTH = 7


def walk_roots():
    for root, dirs, files in os.walk(HOME):
        depth = str(root).count(os.sep) - str(HOME).count(os.sep)
        if depth > MAX_DEPTH:
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in SKIP]
        yield root, dirs, files


print("=" * 78)
print("PART A: every copy of the three contested files, anywhere under the home")
print("=" * 78)
hits: dict = {name: [] for name in CONTESTED}
for root, dirs, files in walk_roots():
    for name in CONTESTED:
        if name in files:
            hits[name].append(Path(root) / name)

for name in CONTESTED:
    print(f"\n--- {name}: {len(hits[name])} copy/copies ---")
    if not hits[name]:
        print("    NONE FOUND anywhere under C:/Users/chan")
        continue
    for p in sorted(hits[name]):
        try:
            b = p.read_bytes()
        except Exception as e:  # noqa: BLE001
            print(f"    UNREADABLE {p}: {type(e).__name__}")
            continue
        t = b.decode("utf-8", errors="replace")
        nl, nb = len(t.splitlines()), len(b)
        tag = "   <<<< 3426/136161 MATCH" if (nl == 3426 or nb == 136161) else ""
        rel = str(p)[len(str(HOME)):].lstrip("\\/")
        print(f"    lines={nl:<6} bytes={nb:<8} sha16={hashlib.sha256(b).hexdigest()[:16]}{tag}")
        print(f"        {rel}")
        if name == "production_arc_run.py":
            ms = {k: t.count(k) for k in MARKERS if t.count(k)}
            print(f"        markers={ms}")

print()
print("=" * 78)
print("PART B: the flag-gated macro-resolution change in semantic-backbone")
print("=" * 78)
rp = SB / "production_arc_run.py"
if not rp.exists():
    print("  RUNNER MISSING at", rp)
else:
    t = rp.read_text(encoding="utf-8", errors="ignore")
    checks = {
        "flag_present": "HENRI_MACRO_NUM_CHANNELS" in t,
        "default_arm_8192": bool(re.search(r"else\s+8192\)", t)),
        "scale_arm": bool(re.search(
            r'_num_channels\s*=\s*\(int\(SCALE\["num_blocks"\]\)', t)),
        "store_uses_var": bool(re.search(
            r"ActionOutcomeGeneratorStore\([^)]*num_channels=_num_channels", t, re.S)),
        "flag_defaults_off": '"HENRI_MACRO_NUM_CHANNELS", "0") == "1"' in t,
        "no_call_site_8192": not re.search(
            r"(?<![_\w])num_channels\s*=\s*8192", t),
        "isinstance_handler": "isinstance(_veto_exc, SagnacGateUnavailable)" in t,
        "gate_unavailable_imported": "SagnacMCTSPlanner, SagnacGateUnavailable" in t,
    }
    for k, v in checks.items():
        print(f"  {k:<28} {v}")
    bad = [k for k, v in checks.items() if not v]
    print("  ALL_OK=" + str(not bad) + ("  failures=" + str(bad) if bad else ""))
    print(f"  runner lines={len(t.splitlines())} bytes={len(rp.read_bytes())}")

print()
print("=" * 78)
print("VERDICT")
print("=" * 78)
m3426 = [str(p) for p in hits["production_arc_run.py"]
         if p.exists() and (len(p.read_bytes()) == 136161
                            or len(p.read_text(encoding="utf-8", errors="replace")
                                   .splitlines()) == 3426)]
if m3426:
    print("  The rendering DOES correspond to a real file:")
    for m in m3426:
        print("    " + m)
    print("  => my earlier 'fabricated' label was WRONG. Correct label: a different")
    print("     worktree outside the project root. CORRECT THIS OPENLY.")
else:
    print("  No copy anywhere under the home matches 3426/136161.")
sint = hits["synthesize_sagnac_guarded_macro_options.py"]
print(f"  synthesize_sagnac_guarded_macro_options.py copies found: {len(sint)}")
efe = hits["efe_planner.py"]
efe_defs = []
for p in efe:
    try:
        tt = p.read_text(encoding="utf-8", errors="ignore")
        efe_defs.append((str(p)[len(str(HOME)):], len(tt.splitlines()),
                         tt.count("def field_to_wave")))
    except Exception:
        pass
print(f"  efe_planner.py copies: {len(efe)}")
for rel, nl, nd in efe_defs:
    print(f"    lines={nl} def_field_to_wave={nd}  {rel}")
