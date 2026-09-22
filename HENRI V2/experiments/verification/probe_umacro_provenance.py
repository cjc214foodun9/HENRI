#!/usr/bin/env python3
"""Where does `_u_macro`'s BLOCK COUNT come from, and does it track `num_blocks`?

WHY THIS IS THE ROOT QUESTION
    MEASURED (action5_su3_width_rule.json, own receipt):
        SU3FieldWaveTransducer.field_to_wave: [B, N, 3, 3] -> [B, N*8]
        N=64  -> 512      N=512 -> 4096      N=8192 -> 65536   (all match N*8)
        encoder reference: num_blocks=64 -> 512 ; num_blocks=8192 -> 65536
    So the two sides agree EXACTLY WHEN the SU(3) field's N equals the encoder's
    `num_blocks`. At the gauntlet's reduced scale the observed call was
        ((65536,), (512,), (512,))   =>  N_field = 8192, num_blocks = 64
    i.e. the macro field was built with 8192 blocks inside a 64-block run.

    This probe asks whether that is a CONFIG LEAK (the field ignores the run's
    num_blocks) or INTENTIONAL (the field has its own resolution). The answer
    determines the correct repair:
      * config leak  -> plumb num_blocks into the field construction (a bounded fix,
        and then the veto runs locally with NO bridge)
      * intentional  -> the two sides are different resolutions by design, and a
        declared cross-resolution comparison is required (REQUIRES_APPROVAL)

    Reported, not assumed: the assignment site and its arguments are read verbatim.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

WT = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\semantic-backbone\HENRI V2")
RUNNER = WT / "production_arc_run.py"

out: dict = {"checks": {}, "errors": [], "notes": []}
chk = out["checks"]
rl = RUNNER.read_text(encoding="utf-8", errors="ignore").splitlines()

# ---------------------------------------------------------------- _u_macro sites
u_sites = []
for i, l in enumerate(rl, 1):
    if re.search(r"\b_u_macro\b", l):
        u_sites.append((i, l.strip()[:150]))
chk["_u_macro_mentions"] = u_sites
assign_sites = [(i, t) for i, t in u_sites if re.match(r"_u_macro\s*=", t)]
chk["_u_macro_assignments"] = assign_sites
chk["_n_assignments"] = len(assign_sites)

# ---------------------------------------------------------------- num_blocks plumbing
nb_sites = []
for i, l in enumerate(rl, 1):
    if re.search(r"num_blocks", l):
        nb_sites.append((i, l.strip()[:150]))
chk["num_blocks_mentions"] = nb_sites[:40]
chk["n_num_blocks_mentions"] = len(nb_sites)

# ---------------------------------------------------------------- OPINE construction
opine_ctor = []
for i, l in enumerate(rl, 1):
    if re.search(r"OPINE|_opine\s*=|MacroOption|macro_option_engine", l):
        opine_ctor.append((i, l.strip()[:150]))
chk["opine_mentions"] = opine_ctor[:30]

# Read the region around each _u_macro assignment for its arguments.
ctx = {}
for i, _ in assign_sites:
    lo, hi = max(1, i - 4), min(len(rl), i + 12)
    ctx[str(i)] = [f"{k}: {rl[k-1]}" for k in range(lo, hi + 1)]
chk["assignment_context"] = ctx

# ---------------------------------------------------------------- verdict
if not assign_sites:
    out["notes"].append(
        "_u_macro is never assigned in production_arc_run.py; it is produced by the "
        "OPINE/macro-option component. The block count therefore originates in that "
        "component's construction, which must be inspected next.")
else:
    joined = "\n".join(t for _, t in assign_sites)
    chk["assignment_exprs"] = joined
    if "num_blocks" in joined:
        chk["verdict"] = "FIELD_TAKES_num_blocks -> config leak is FIXABLE by plumbing"
    else:
        chk["verdict"] = ("FIELD_IS_BUILT_WITHOUT_num_blocks -> either a different "
                          "resolution by design, or a fixed 8192 leak; inspect the "
                          "component named in assignment_context")

# Where is the OPINE engine constructed? Its ctor arguments settle the field's N.
opine_engine_defs = []
for p in sorted(WT.rglob("*.py")):
    if "_archive" in p.parts or ".git" in p.parts:
        continue
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    if re.search(r"class\s+\w*OPINE\w*|class\s+\w*MacroOption\w*", txt):
        for i, l in enumerate(txt.splitlines(), 1):
            if re.match(r"class\s+(?:\w*OPINE\w*|\w*MacroOption\w*)", l.strip()):
                opine_engine_defs.append({"file": p.name, "line": i, "text": l.strip()[:110]})
chk["opine_engine_classes"] = opine_engine_defs

# the ctor signature of that class, if in its own file
for d in opine_engine_defs:
    p = WT / d["file"]
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    body = [f"{i}: {lines[i-1]}" for i in range(d["line"], min(len(lines), d["line"] + 40) + 1)]
    chk[f"ctor_body::{d['file']}"] = body
    j = "\n".join(body)
    chk[f"ctor_uses_num_blocks::{d['file']}"] = "num_blocks" in j
    chk[f"ctor_uses_d_model::{d['file']}"] = "d_model" in j

out["verdict"] = "PASS" if not out["errors"] else "FAIL"
receipt = WT / "experiments" / "verification" / "umacro_provenance.json"
receipt.write_text(json.dumps(out, indent=1), encoding="utf-8")

print("UMA=" + out["verdict"])
print("n_assign=" + str(chk.get("_n_assignments"))
      + " n_nb_mentions=" + str(chk.get("n_num_blocks_mentions")))
print("verdict=" + str(chk.get("verdict")))
print("assign_exprs=" + str(chk.get("assignment_exprs"))[:220])
print("opine_classes=" + json.dumps(chk.get("opine_engine_classes")))
for k, v in chk.items():
    if k.startswith("ctor_uses_"):
        print("  " + k + "=" + str(v))
first = list(ctx.items())[:1]
if first:
    k, v = first[0]
    print("ctx@" + k + ":")
    for line in v[:8]:
        print("   " + line[:150])
print("RECEIPT=" + str(receipt))
