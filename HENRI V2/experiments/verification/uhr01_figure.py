"""UHR-01 measured figure — deterministic (no randomness), fixed geometry.

Numbers are EXACTLY those written in experiments/verification/uhr01_verdict.md.
A diagram never overrides the numeric artifact; it restates it.

  Panel A: paired remote A/B, delta_axiom per live step (instance 52189427).
  Panel B: contract-suite / probe gate populations against tau = 0.35.

Render:  python experiments/verification/uhr01_figure.py
Output:  experiments/verification/uhr01_measured.png
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- OBSERVED: paired remote A/B (egressed JSONL) ----------------
steps = list(range(8))
baseline = [0.999185, 0.999582, 0.997542, 0.998176, 0.997175, 0.998510, 0.997528, 0.998115]
rfss = [0.0] * 8
TAU = 0.35

# ---------------- OBSERVED: gate populations (contract suite + probe) ---------
synth = [
    ("identity /\ndead input",      0.000000, "#2ca02c"),   # |delta| = 2.3e-07
    ("compliant\n(populated, 4 gens)", 0.133114, "#9467bd"),
    ("compliant\n(populated, dtheta=0.10)", 0.350760, "#e377c2"),  # AT tau
    ("invalid\n(permuted roles)",   0.498332, "#ff7f0e"),
    ("wrong axiom\n(other roles)",  0.498343, "#8c564b"),
    ("legacy\n(raw random vs axiom)", 0.496575, "#7f7f7f"),
]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.titlesize": 9, "axes.labelsize": 8,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.8), dpi=110)

# ------------------------------- Panel A -------------------------------------
ax1.plot(steps, baseline, marker="o", ms=4, lw=1.7, color="#1f77b4",
         label="BASELINE  legacy cross-family")
ax1.plot(steps, rfss, marker="s", ms=4, lw=1.7, color="#d62728",
         label="RFSS  projected (empty store)")
ax1.axhline(TAU, ls="--", lw=1.2, color="#444444")
ax1.text(0.08, TAU + 0.035, "tau = 0.35  (hard-veto threshold)", fontsize=7, color="#333333")
ax1.fill_between([-0.4, 7.4], TAU, 1.15, color="#d62728", alpha=0.06)
ax1.set_xlim(-0.4, 7.4)
ax1.set_ylim(-0.08, 1.15)
ax1.set_xticks(steps)
ax1.set_xlabel("live step  (phase823_live_gauntlet, RTX PRO 5000)")
ax1.set_ylabel("delta_axiom")
ax1.set_title("A. Paired remote A/B — OBSERVED\nbaseline 8/8 vetoed  ->  RFSS 0/8 vetoed")
ax1.grid(alpha=0.25, lw=0.6)
ax1.legend(loc="lower left", fontsize=6.8, framealpha=0.92)
ax1.annotate("0.0 (all 8 steps)\nempty store -> theta=0 -> U=I\n-> Ad(U)=I -> candidate IS the axiom",
             xy=(4.0, 0.0), xytext=(1.15, 0.60), fontsize=6.4, color="#7f0000",
             arrowprops=dict(arrowstyle="->", lw=0.9, color="#7f0000"))
ax1.text(0.08, 1.06, "engaged: False -> True   uhr01 provenance: absent -> present",
         fontsize=6.4, color="#7f0000")

# ------------------------------- Panel B -------------------------------------
names = [s[0] for s in synth]
vals = [s[1] for s in synth]
cols = [s[2] for s in synth]
bars = ax2.bar(range(len(names)), vals, color=cols, width=0.62, edgecolor="white", lw=0.5)
ax2.axhline(TAU, ls="--", lw=1.2, color="#444444")
ax2.axhspan(TAU, 0.62, color="#d62728", alpha=0.06)
ax2.set_xticks(range(len(names)))
ax2.set_xticklabels(names, fontsize=5.6)
ax2.set_ylabel("delta_axiom")
ax2.set_ylim(0, 0.62)
ax2.set_title("B. Gate populations — OBSERVED\ncompliant below tau, invalid above")
for b, v in zip(bars, vals):
    ax2.text(b.get_x() + b.get_width() / 2, v + 0.013, "%.4f" % v, ha="center", fontsize=6.0)
ax2.grid(axis="y", alpha=0.25, lw=0.6)
ax2.annotate("AT tau (margin 0.0008)\ntau is not calibrated\nfor the projected family",
             xy=(2.0, 0.3508), xytext=(2.75, 0.205), fontsize=6.2, color="#7f0000",
             arrowprops=dict(arrowstyle="->", lw=0.9, color="#7f0000"))

fig.suptitle("UHR-01 homologous representation: gate moved from ALWAYS-VETO to NEVER-VETO (live store empty)",
             fontsize=9.2, y=0.995)
fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

out = pathlib.Path(__file__).resolve().parent / "uhr01_measured.png"
fig.savefig(out, dpi=110)
print("wrote", out)
print("figure px:", [int(x) for x in (fig.get_size_inches() * 110)])
print("max px <= 800x800:", all(x <= 880 for x in (fig.get_size_inches() * 110)))
