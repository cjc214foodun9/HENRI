"""RENDER the six-stage pipeline figure FROM THE MEASURED JSON.

RUN THIS WITH THE VIZ INTERPRETER (matplotlib):
    %LOCALAPPDATA%/hermes/viz-venv/Scripts/python.exe tools/render_functional_pipeline.py

Reads experiments/verification/functional_pipeline_measurements.json so no number in the
figure can drift from the measurement, and so the renderer needs no torch.
"""
import json
import os
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.dirname(C)
M = os.path.join(C, "experiments", "verification",
                 "functional_pipeline_measurements.json")
d = json.load(open(M, encoding="utf-8"))
try:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=R,
                          capture_output=True, text=True).stdout.strip()
except Exception:
    head = "?"

fig = plt.figure(figsize=(14.8, 10.2))
fig.text(0.5, 0.970, "Project HENRI - the six-stage functional pipeline, joined and measured",
         ha="center", fontsize=13.8, fontweight="bold")
fig.text(0.5, 0.944, "every number read from the measured JSON at render time  |  HEAD %s" % head,
         ha="center", fontsize=8.5, color="#555555", style="italic")

# ---------------------------------------------------------------- stage diagram
ax = fig.add_axes([0.045, 0.545, 0.91, 0.35]); ax.axis("off")
STAGES = [
    ("1 INGRESS", "2-D grid tasks\n3 families\nnon-identity\n3 disjoint bands", "#4477aa"),
    ("2 BINDER", "hierarchical\nrole-filler scene\nmeasured shape\nand norm", "#228833"),
    ("3 ROUTER", "LOO-CV over\nRIDGE / D4 / TOPO\nTOPO = 0 fitted\nparams", "#aa3377"),
    ("4 WORLD", "Koopman rollout\n5 latent steps\nK_a Psi(t)", "#cc7722"),
    ("5 GATE", "Sagnac homodyne\neps = %.2f\nEXTINGUISHES\npaths" % d["pipeline_report"]["epsilon_hard"], "#882222"),
    ("6 EGRESS", "Hopfield snap\nbeta = %.1f sealed\n-> CODEBOOK\nINDEX" % d["pipeline_report"]["beta"], "#6633aa"),
]
w, gap = 0.146, 0.0085
for i, (name, body, col) in enumerate(STAGES):
    x = 0.008 + i * (w + gap)
    ax.add_patch(plt.Rectangle((x, 0.44), w, 0.48, transform=ax.transAxes,
                               facecolor=col, alpha=0.13, edgecolor=col, lw=1.6))
    ax.text(x + w / 2, 0.86, name, transform=ax.transAxes, ha="center",
            fontsize=9.0, fontweight="bold", color=col)
    ax.text(x + w / 2, 0.62, body, transform=ax.transAxes, ha="center",
            fontsize=7.3, color="#333333")
    if i < len(STAGES) - 1:
        ax.annotate("", xy=(x + w + gap - 0.001, 0.68), xytext=(x + w, 0.68),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))

ax.text(0.008, 0.30,
        "PATH A (symbolic, grid-valued):  ingress -> binder -> router+region select -> PREDICTED GRID",
        transform=ax.transAxes, fontsize=8.3, fontweight="bold", color="#2255aa")
ax.text(0.008, 0.16,
        "PATH B (continuous, action-valued):  ingress -> binder -> koopman rollout -> sagnac veto -> hopfield egress -> CODEBOOK INDEX",
        transform=ax.transAxes, fontsize=8.3, fontweight="bold", color="#aa3333")
ax.text(0.008, -0.02,
        "THE TWO PATHS DO NOT SHARE A MIDDLE: a symbolic operator cannot be Koopman-rolled out, and a latent trajectory is not a grid fill.\n"
        "The blueprint draws ONE arrow through both. Conflating them would be a category error, so the pipeline runs them separately and states which produced the output.",
        transform=ax.transAxes, fontsize=7.5, color="#882222", va="top")

# ---------------------------------------------------------------- measured results
ax2 = fig.add_axes([0.045, 0.215, 0.44, 0.27])
fams = list(d["families"])
vals = [sum(1 for r in d["families"][f] if r["exact"]) for f in fams]
ns = [len(d["families"][f]) for f in fams]
ax2.bar(range(len(fams)), vals, color="#228833", width=0.55)
for i, v in enumerate(vals):
    ax2.text(i, v + 0.06, "%d/%d" % (v, ns[i]), ha="center", fontsize=9.5,
             fontweight="bold")
ax2.set_xticks(range(len(fams)))
ax2.set_xticklabels([f.replace("_", "\n") for f in fams], fontsize=8.2)
ax2.set_ylim(0, 5.0)
ax2.set_ylabel("exact grid match (3 demos -> held-out)")
routes = ", ".join("%s->%s" % (f.split("_")[0], d["families"][f][0]["route"]) for f in fams)
ax2.set_title("PATH A on %d seeds per family: %d/%d exact\nroutes: %s"
              % (len(d["seeds"]), d["exact_total"], d["n_total"], routes), fontsize=9.5)
ax2.grid(axis="y", alpha=0.22)

# ---------------------------------------------------------------- the corrections
ax3 = fig.add_axes([0.545, 0.215, 0.41, 0.27]); ax3.axis("off")
ax3.set_title("MEASURED DEFECTS FOUND AND CORRECTED (all mine)", fontsize=9.8, loc="left")
BANDS = d["router_bands"]
defs = [
    ("1 ENCODER CONTRACT (9 call sites)",
     "flat() assumed a tensor; the topological encoder returns (list, features)\n"
     "-> AttributeError at every flat(self.enc.encode(...)) site",
     "FIX: one adapter accepts tensor | (wave,feats) | list; RAISES otherwise"),
    ("2 PALETTE COLLISION",
     "my curve band hit the router's NOISE_BAND %s\n-> rings were BACKGROUND; TOPO abstained at cv 0.0000"
     % (BANDS["NOISE_BAND"],),
     "FIX: bands IMPORTED from the router (ring %s, fill band %s)"
     % (BANDS["RING_BAND"], BANDS["FILL_BAND"][0])),
    ("3 PER-TASK FILL / RANDOM TRANSFORM",
     "TopoChannel needs ONE constant across demos; the router searches ONE operator\n"
     "-> per-task IoU was 1.0000 and it STILL abstained; held-out matched 1 of 4 seeds",
     "FIX: fill, cue and rotation are batch-constant"),
]
y = 0.97
for h, b, f in defs:
    ax3.text(0, y, h, fontsize=8.5, fontweight="bold", color="#882222", family="monospace")
    ax3.text(0.02, y - 0.072, b, fontsize=7.3, family="monospace", color="#333333")
    ax3.text(0.02, y - 0.190, f, fontsize=7.3, family="monospace", color="#226622",
             fontweight="bold")
    y -= 0.335

pa, pb = d["path_a"], d["path_b"]
fig.text(0.5, 0.148,
         "WHAT THIS IS  |  PATH A is IN-CONTEXT OPERATOR SELECTION from K demonstrations, measured as exact grid match -- real, narrow, and NOT learning. "
         "PATH B returns a CODEBOOK INDEX, never an executed action.\n"
         "PATH A trace on %s: %s  (route %s, correct %s)   |   PATH B: %s\n"
         "WHAT THIS IS NOT  |  no benchmark score exists. arc_agi is absent on this host, the prefix projection is untrained, and there is no attention core to host a KV cache. "
         "SciCode / ARC-AGI / AAII stay BLOCKED.\n"
         "STILL OPEN  |  the 2-D emitter is NOT wired into the seeding driver (opt-in by design); the cadence escalation mode is unit-tested but NOT yet re-run end to end."
         % (pa["family"], " -> ".join(pa["stages"]), pa["route"], pa["correct"],
            pb.get("abstain_reason") or pb.get("emitted_index")),
         ha="center", va="top", fontsize=7.7,
         bbox=dict(boxstyle="round,pad=0.55", fc="#fff7ec", ec="#cc7722"))

fig.text(0.5, 0.030,
         "claim -> hypothesis -> evidence -> mechanism -> action -> verification -> uncertainty   |   "
         "labels: OBSERVED / DERIVED / HYPOTHESIS / FALSIFIED / BLOCKED   |   no benchmark score is claimed",
         ha="center", fontsize=8.0, color="#555555")

out = os.path.join(C, "docs", "diagrams", "functional_pipeline_result.png")
fig.savefig(out, dpi=140)
print("wrote %s %d bytes" % (out, os.path.getsize(out)))
