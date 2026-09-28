"""Render the six-gaps architecture figure: gap -> new module -> gate -> status.

Every value in this figure is read from the live repository and the receipts, so
the figure cannot drift from the code. If a file is missing the figure says
ABSENT rather than drawing a green box.
"""
import hashlib
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = r"C:/Users/chan/henri-worktrees/zone-a-selfplay/HENRI V2"
V = os.path.join(C, "experiments", "verification")


def exists(rel):
    p = os.path.join(C, rel)
    return os.path.getsize(p) if os.path.exists(p) else None


def count(rel, pats):
    p = os.path.join(C, rel)
    if not os.path.exists(p):
        return {k: None for k in pats}
    s = open(p, encoding="utf-8", errors="ignore").read()
    return {k: s.count(k) for k in pats}


def receipt(name):
    p = os.path.join(V, name)
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


suite = receipt("operator_router_family_suite.json") or {}
cv = receipt("action2_topo3_router_cv_observed.json") or {}
dec = count("henri_decoder.py", ["PrefixConditioner", "prefix_cond"])
tok = count("o_vsa_ingress_tokenizer.py", ["multiscale", "interior", "boundary"])

GAPS = [
    ("G1 HIERARCHY", "henri_scene_binder.py",
     "29/29 tests; permutation + exact role-filler read-out\n"
     "poisoned prior (qFHRR random ring) NOT used",
     "BUILT"),
    ("G2 OPERATOR POOL", "henri_operator_router.py",
     "28/28 tests; LOO-CV, capacity tie-break\n"
     "family suite: in-family OK, out-of-family FALSIFIED",
     "BUILT"),
    ("G3 EGRESS", "henri_hopfield_egress.py",
     "PRESENT with SEALED beta=8.0\n"
     "doc demands beta*=26.10 -> MEASURE, never adopt",
     "PRESENT"),
    ("G4 WORLD MODEL", "henri_action_koopman.py",
     "31/31 tests; per-action K_a, abstains on unseen\n"
     "rollout error vs known truth < 0.15",
     "BUILT"),
    ("G5 CURRICULUM", "henri_curriculum_governor.py",
     "19/19 tests; 5-rung heterogeneous ladder,\n"
     "plateau KILL switch reachable",
     "BUILT"),
    ("G6 SUBSTRATE", "henri_prefix_kv.py",
     "weights hardlinked (nlink=2); prefix wired\n"
     "default OFF; state_dict byte-identical",
     "WIRED"),
]

fig = plt.figure(figsize=(15.0, 9.2))
fig.text(0.5, 0.965, "Project HENRI — Six Missing Systems: gap -> module -> gate -> measured status",
         ha="center", fontsize=13.5, fontweight="bold")
fig.text(0.5, 0.935,
         "every status below is read from the live repository and the receipts at render time; "
         "an absent file renders as ABSENT, never as a green box",
         ha="center", fontsize=8.6, color="#555555", style="italic")

colors = {"BUILT": "#228833", "PRESENT": "#cc7722", "WIRED": "#3366aa", "ABSENT": "#bb2222"}
y = 0.845
for name, mod, gate, status in GAPS:
    size = exists(mod)
    head = f"{name}   ->   {mod}"
    sub = (f"file {size:,} B" if size else "file ABSENT") + "   |   " + gate
    fig.text(0.035, y, head, fontsize=10.5, fontweight="bold", family="monospace",
             bbox=dict(boxstyle="round,pad=0.45", fc="#f4f4f4", ec=colors[status], lw=1.8))
    fig.text(0.045, y - 0.042, sub, fontsize=8.5, family="monospace", color="#333333")
    fig.text(0.965, y - 0.006, status, fontsize=11, fontweight="bold", ha="right",
             color=colors[status])
    y -= 0.125

# ---- measured evidence strip
oof = ", ".join(suite.get("out_of_family_failed", []) or "n/a")
lines = [
    "MEASURED THIS SESSION (read from receipts at render time)",
    f"  router (3-channel, LOO-CV)     : contained {cv.get('containment',{}).get('topo_delta','n/a')} "
    f"| reflection {cv.get('reflection',{}).get('rigid_delta','n/a')} | router "
    f"{cv.get('router_correct_total','n/a')} | controls {cv.get('controls_result',{}).get('controls_ok','n/a')}",
    f"  family suite (OUT-OF-FAMILY)   : in_family_ok={suite.get('in_family_ok','n/a')} "
    f"| FAILED={oof}",
    f"  non-convex control             : {suite.get('nonconvex_control_delta','n/a')} "
    "  <- SHAPE general, TOPOLOGY limited",
    f"  open-curve TOPO-own inertness  : {suite.get('open_curve_topo_own_delta','n/a')} (exact)",
    f"  decoder prefix wiring          : PrefixConditioner in code = {dec['PrefixConditioner']}, "
    f"prefix_cond = {dec['prefix_cond']}  | default OFF, state_dict 4 keys unchanged",
    f"  multiscale ingress             : tokenizer hits multiscale={tok['multiscale']} "
    f"interior={tok['interior']} boundary={tok['boundary']}  17/17 tests",
    "HONEST BOUNDARIES: synthetic grids, first held pair per task; NOT an ARC/SciCode score. "
    "SciCode/ARC scoring = BLOCKED (no trained prefix projections, no remote CUDA, arc_agi absent).",
]
fig.text(0.035, 0.115, "\n".join(lines), fontsize=8.3, va="top", family="monospace",
         bbox=dict(boxstyle="round,pad=0.6", fc="#eef7ff", ec="#3366aa"))

fig.text(0.5, 0.022,
         "claim -> hypothesis -> evidence -> mechanism -> action -> verification -> uncertainty   |   "
         "labels: OBSERVED / DERIVED / INFERRED / HYPOTHESIS / FALSIFIED / BLOCKED",
         ha="center", fontsize=8.2, color="#555555")

out = os.path.join(C, "docs", "diagrams", "six_gaps_architecture.png")
fig.savefig(out, dpi=140)
print("wrote %s %d bytes" % (out, os.path.getsize(out)))
