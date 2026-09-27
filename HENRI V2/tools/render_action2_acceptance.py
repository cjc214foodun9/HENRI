#!/usr/bin/env python
"""Render the ACTION 2 acceptance split + the ACTION 3 transducer pipeline.

Deterministic matplotlib (fixed seed/DPI/palette). Evidence labels are printed
on the figure. It never overrides a numeric receipt.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(CODE, "experiments", "verification",
                 "action2_acceptance_reflection_containment_observed.json")
OUT = os.path.join(CODE, "docs", "diagrams", "action2_acceptance_and_transducer.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
d = json.load(open(P, encoding="utf-8"))

fig = plt.figure(figsize=(14.5, 6.2), dpi=110)
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28)

# ---------------- panel 1: the acceptance split ----------------
ax = fig.add_subplot(gs[0, 0])
fams  = ["REFLECTION", "CONTAINMENT", "TOTAL"]
ident = [d["reflection"]["identity_mean"], d["containment"]["identity_mean"], None]
ctrl  = [d["reflection"]["control_mean"],  d["containment"]["control_mean"],  None]
treat = [d["reflection"]["treatment_mean"],d["containment"]["treatment_mean"],None]
x = [0, 1, 2]; w = 0.26
ax.bar([i - w for i in x], [v if v is not None else 0 for v in ident], w, label="identity",      color="#888888")
ax.bar([i     for i in x], [v if v is not None else 0 for v in ctrl ], w, label="CONTROL (diag ridge)", color="#1f77b4")
ax.bar([i + w for i in x], [v if v is not None else 0 for v in treat], w, label="TREATMENT (class)",     color="#d62728")
for i, (a, b, c) in enumerate(zip(ident, ctrl, treat)):
    if a is not None:
        ax.text(i - w, a + 0.02, "%.4f" % a, ha="center", fontsize=8.5, weight="bold")
        ax.text(i,     b + 0.02, "%.4f" % b, ha="center", fontsize=8.5, weight="bold")
        ax.text(i + w, c + 0.02, "%.4f" % c, ha="center", fontsize=8.5, weight="bold")
# total panel = recovered counts, on a twin axis
axt = ax.twinx()
axt.bar([2.30], [d["reflection"]["recovered"]], 0.34, color="#2ca02c", alpha=0.85, label="recovered (count)")
axt.bar([2.62], [d["containment"]["recovered"]], 0.34, color="#d62728", alpha=0.85)
axt.set_ylim(0, 16); axt.set_ylabel("recovered / 8", color="#2ca02c")
axt.text(2.30, d["reflection"]["recovered"] + 0.4, "8/8", ha="center", fontsize=9, weight="bold", color="#2ca02c")
axt.text(2.62, 0.4, "0/8", ha="center", fontsize=9, weight="bold", color="#d62728")
ax.set_xticks([0, 1]); ax.set_xticklabels(["REFLECTION\n(rigid -> D4)", "CONTAINMENT\n(position-dependent)"], fontsize=9)
ax.set_ylabel("held-out cosine"); ax.set_ylim(0, 1.15); ax.set_xlim(-0.55, 1.45)
ax.set_title("ACTION 2 acceptance: %s vs target %s -> FALSIFIED\n"
             "reflections exact (cos 1.000000); containment inexpressible by a global-operator class"
             % (d["recovery_total"], d["preregistration"]["target"]), fontsize=10)
ax.legend(fontsize=8, loc="upper left"); ax.grid(axis="y", alpha=0.25)
ax.axhline(0.0, color="k", lw=0.6)

# ---------------- panel 2: transducer pipeline ----------------
ax2 = fig.add_subplot(gs[0, 1]); ax2.axis("off")
ax2.set_title("ACTION 3 — FUWT transducer (299 lines, 23/23 tests): mechanisms present",
              fontsize=10)
steps = [
    ("unitary wave", "psi  [B, 65536] complex64\n||psi||_2 = 1  (fail-closed)",
     "#e8f0fe"),
    ("polar_decompose", "-> PolarFeatures [B, 32, 3]\n(r_k, cos theta_k, sin theta_k)", "#e6f4ea"),
    ("lexical_snap", "softmax(beta * W_codebook . Re(psi))\nbeta = 8.0 fixed by spec", "#fce8e6"),
    ("prefix_embeddings", "-> [B, 32, d_kv] one-way\nseeded fixed rotation, not learned", "#f3e8fd"),
]
y = 0.88
for i, (t, sub, col) in enumerate(steps):
    ax2.add_patch(plt.Rectangle((0.05, y - 0.155), 0.9, 0.15,
                                transform=ax2.transAxes, facecolor=col, edgecolor="#444", lw=1.0))
    ax2.text(0.08, y - 0.045, t, transform=ax2.transAxes, fontsize=10.5, weight="bold")
    ax2.text(0.08, y - 0.115, sub, transform=ax2.transAxes, fontsize=8.6, family="monospace")
    if i < len(steps) - 1:
        ax2.annotate("", xy=(0.5, y - 0.215), xytext=(0.5, y - 0.155),
                     xycoords="axes fraction", textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.4))
    y -= 0.235
ax2.text(0.05, 0.055, "fail-closed: non-unitary -> TransducerNormViolation | bad shape -> TransducerShapeError\n"
                      "HONEST LIMIT: no KV wiring into a backbone (models/ ABSENT); no benchmark score claimed.",
         transform=ax2.transAxes, fontsize=8.4, color="#a00")

fig.suptitle("HENRI — ACTION 2 acceptance (measured 8/16) + ACTION 3 transducer surface   "
             "[OBSERVED: experiments/verification/action2_acceptance_reflection_containment_observed.json]",
             fontsize=11.5, weight="bold")
fig.savefig(OUT, bbox_inches="tight")
print("wrote %s  (%d bytes)" % (OUT, os.path.getsize(OUT)))
