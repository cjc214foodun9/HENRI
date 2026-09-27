#!/usr/bin/env python
"""ACTION 2 result figure: the paired held-out A/B that falsified the resonator class.

Deterministic matplotlib (fixed seed/DPI/palette). Evidence labels are printed on
the figure. It never overrides the numeric receipt.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CODE = r"C:\Users\chan\henri-worktrees\zone-a-selfplay\HENRI V2"
P = os.path.join(CODE, "experiments", "verification", "action2_resonator_paired_ab_observed.json")
OUT = os.path.join(CODE, "docs", "diagrams", "action2_paired_ab.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
d = json.load(open(P, encoding="utf-8"))
h2, h1 = d["h2_decision"], d["h1_mechanism"]
rows = d["per_task"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=110,
                               gridspec_kw={"width_ratios": [1.0, 1.35]})

# ---- panel 1: the three arms
names = ["identity", "CONTROL\ndiagonal ridge", "TREATMENT\nresonator class"]
vals = [h2["identity_mean_cos"], h2["control_mean_cos"], h2["treatment_mean_cos"]]
cols = ["#888888", "#1f77b4", "#d62728"]
b = ax1.bar(names, vals, color=cols, width=0.6)
ax1.set_ylabel("mean held-out cosine")
ax1.set_title("ACTION 2 — paired held-out A/B\nn=60 real ARC-AGI-2 training tasks", fontsize=11)
ax1.set_ylim(0, 0.62)
for r, v in zip(b, vals):
    ax1.text(r.get_x() + r.get_width() / 2, v + 0.008, f"{v:.6f}", ha="center", fontsize=10, weight="bold")
# Annotation placed ABOVE the tallest bar label in axes coordinates.
# A prior revision anchored it at data (1.5, 0.46), which drew the callout box over the
# CONTROL bar's numeric label and HID it (measured 2026-09-27 by inspecting the PNG).
# A figure that obscures a measured number violates the diagram mandate.
ax1.text(0.5, 0.90, f"$\\Delta$ = {h2['delta']:+.6f}  (vs $\\tau$={h2['tau']})\n$\\rightarrow$ FALSIFIED",
         transform=ax1.transAxes, ha="center", va="center", fontsize=11, weight="bold",
         color="#d62728", bbox=dict(boxstyle="round,pad=0.4", fc="#ffe6e6", ec="#d62728"))
ax1.grid(axis="y", alpha=0.25)

# ---- panel 2: per-task scatter, treatment vs control
tc = [r["treatment_cos"] for r in rows]
cc = [r["control_cos"] for r in rows]
ii = [r["identity_cos"] for r in rows]
ax2.scatter(cc, tc, s=34, c=["#d62728" if r.get("identity_triple") else "#2ca02c" for r in rows],
            alpha=0.85, edgecolors="k", linewidths=0.4,
            label="identity triple chosen (n=%d)" % sum(1 for r in rows if r.get("identity_triple")))
ax2.scatter(cc, ii, s=14, marker="x", color="#888888", alpha=0.7, label="identity baseline")
lo, hi = 0.0, 1.0
ax2.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="treatment = control")
ax2.set_xlabel("CONTROL held-out cosine (diagonal ridge)")
ax2.set_ylabel("TREATMENT / identity held-out cosine")
ax2.set_title("Per-task: treatment rarely exceeds control\n"
              f"H1 mechanism {h1['exact']}/{h1['trials']} = {h1['rate']:.3f} PASS "
              f"(solvable-by-construction only)", fontsize=11)
ax2.legend(fontsize=8, loc="upper left")
ax2.grid(alpha=0.25)

fig.suptitle("HENRI ACTION 2 (re-scoped) — resonator class FALSIFIED on real ARC   "
             "[OBSERVED: experiments/verification/action2_resonator_paired_ab_observed.json]",
             fontsize=11.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT)
print(f"wrote {OUT}  ({os.path.getsize(OUT):,} bytes)")
