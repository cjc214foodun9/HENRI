#!/usr/bin/env python3
"""TWO FINAL QUESTIONS, ONE PASS.

Q1 REGRESSION (load-bearing). After binding num_channels to SCALE, the macro field
   became 64 channels at CPU scale and the veto's width problem disappeared -- but a
   DIFFERENT error surfaced, 64x:
       RuntimeError: einsum(): subscript n has size 8192 for operand 1 which does not
                     broadcast with previously seen size 64
   So a SECOND hardcoded 8192 lives downstream in the macro-option path. Name it.
   NOTE: at GPU scale SCALE["num_blocks"] == 8192, so my change is a NO-OP there and
   the second site is also correct there. This chain is REDUCED-SCALE-ONLY.

Q2 IDENTITY. Advisor renderings report production_arc_run.py at 3426 lines / 136161
   bytes with fields I did not author. My measurements report a different size and
   those fields at count 0. A nested `HENRI V2/HENRI V2/` copy would reconcile this,
   so every candidate path is checked explicitly, including nested ones.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

TOP = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM")
WT = TOP / ".worktrees" / "semantic-backbone" / "HENRI V2"

REL = str(TOP)
SEP = os.sep


def rel(p: Path) -> str:
    s = str(p)
    return s[len(REL):].lstrip(SEP) if s.startswith(REL) else s


# ============================================================ Q2: identity
print("=" * 78)
print("Q2  IDENTITY: every production_arc_run.py under the project tree")
print("=" * 78)
found = []
for root, dirs, names in os.walk(TOP):
    dirs[:] = [d for d in dirs if d != ".git"]
    for nm in names:
        if nm == "production_arc_run.py":
            found.append(Path(root) / nm)
found.sort()
for p in found:
    try:
        b = p.read_bytes()
    except Exception as e:  # noqa: BLE001
        print("  UNREADABLE", rel(p), type(e).__name__)
        continue
    t = b.decode("utf-8", errors="replace")
    lines = len(t.splitlines())
    tag = ""
    if lines == 3426 or len(b) == 136161:
        tag = "   <<<< MATCHES THE ADVISOR RENDERING"
    print("  lines=%-5d bytes=%-7d sha16=%s isinst=%d namecmp=%d candel=%d bUsed=%d%s"
          % (lines, len(b), hashlib.sha256(b).hexdigest()[:16],
             t.count("isinstance(_veto_exc"),
             t.count('name == "SagnacGateUnavailable"'),
             t.count("candidate_elems"),
             t.count("_branch_used"), tag))
    print("        " + rel(p))
print("  copies=%d   rendering target: lines=3426 bytes=136161" % len(found))

print()
print("  -- contested module, per tree --")
for base in sorted({p.parent for p in found}):
    cm = base / "synthesize_sagnac_guarded_macro_options.py"
    print("  exists=%-5s  %s" % (cm.exists(), rel(cm)))
    # also probe for a NESTED copy
    nested = base / "HENRI V2" / "synthesize_sagnac_guarded_macro_options.py"
    if nested.exists():
        print("  nested=%-5s  %s" % (nested.exists(), rel(nested)))

print()
print("  -- efe_planner.py sizes (the rendering claims 2185/2346 exist) --")
for base in sorted({p.parent for p in found}):
    ep = base / "efe_planner.py"
    if ep.exists():
        t = ep.read_text(encoding="utf-8", errors="ignore")
        print("  lines=%-5d def_field_to_wave=%d  %s"
              % (len(t.splitlines()), t.count("def field_to_wave"), rel(ep)))

# ============================================================ Q1: second 8192
print()
print("=" * 78)
print("Q1  the SECOND hardcoded 8192 (the einsum operand)")
print("=" * 78)
CANDIDATES = [
    "opine_object_mcts.py",
    "henri_external_outcome_refactor_module.py",
    "universal_data_transducer.py",
    "chromodynamic_grounding.py",
]
for name in CANDIDATES:
    p = WT / name
    if not p.exists():
        print("\n%s: ABSENT" % name)
        continue
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    eight = [(i, l.strip()) for i, l in enumerate(lines, 1) if re.search(r"\b8192\b", l)]
    eins = [(i, l.strip()) for i, l in enumerate(lines, 1) if "einsum" in l]
    print("\n%s  lines=%d  8192hits=%d einsum=%d" % (name, len(lines), len(eight), len(eins)))
    for i, l in eight[:10]:
        print("   8192 %5d: %s" % (i, l[:116]))
    for i, l in eins[:12]:
        # flag einsums that mention n in more than one operand
        subs = re.findall(r"['\"]([a-z,]+->[a-z]+)['\"]", l)
        mark = ""
        for s in subs:
            lhs, rhs = s.split("->")
            if lhs.count("n") >= 2 or (lhs.count("n") == 1 and "n" in rhs):
                mark = "   <<<< uses n across operands"
        print("   eins %5d: %s%s" % (i, l[:100], mark))
