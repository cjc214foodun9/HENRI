"""UHR-02 measured figure — deterministic, fixed geometry, no randomness at render.

Restates the numbers in experiments/verification/uhr02_*.py outputs. A diagram
never overrides the numeric artifact; the numbers plotted are the measured ones.

Panel A: the pre-registered falsification, answered. Separation of an EXACT
         equal-|Tr U| option pair, per comparison domain and baseplate, against
         the 4-sigma sampling gate.
Panel B: tau calibration on the populated store (FORM B), showing 0.35 lies in
         the measured compliant/invalid band.

Render: python experiments/verification/uhr02_figure.py
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- OBSERVED: uhr02_store_population_probe.py (populated production store) ----
K = 2048
BAND = 0.5 * (2.0 / 10.0) ** 0.5 / (K ** 0.5)
SEP_GATE = 4.0 * BAND

rows = [
    ("FORM A (vs state)\nISOTROPIC",      5.172e-03, "NO-SEP"),
    ("FORM A (vs state)\nSTRUCTURED c=6", 2.102e-01, "SEP"),
    ("FORM B (vs transition)\nISOTROPIC", 3.968e-01, "SEP"),
    ("FORM B (vs transition)\nSTRUCTURED c=6", 5.662e-01, "SEP"),
]
tau_comp, tau_inval = 0.000000, 0.649520

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.titlesize": 9, "axes.labelsize": 8,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.6), dpi=110)

labels = [r[0] for r in rows]
seps = [r[1] for r in rows]
cols = ["#d62728" if r[2] == "NO-SEP" else "#2ca02c" for r in rows]

bars = ax1.barh(range(len(rows)), seps, color=cols, height=0.58,
                edgecolor="white", lw=0.5)
ax1.axvline(SEP_GATE, ls="--", lw=1.3, color="#444444")
ax1.text(SEP_GATE * 1.08, -0.62, "4-sigma gate %.3f" % SEP_GATE,
         fontsize=6.4, color="#333333")
ax1.set_yticks(range(len(rows)))
ax1.set_yticklabels(labels, fontsize=6.4)
ax1.set_xlabel("separation of an EXACT equal-|Tr U| option pair")
ax1.set_xlim(0, 0.66)
ax1.set_title("A. The pre-registered falsification, answered\n"
              "populated store: FORM A stays magnitude-blind when isotropic")
for b, v, r in zip(bars, seps, rows):
    ax1.text(v + 0.012, b.get_y() + b.get_height() / 2, "%.3e  %s" % (v, r[2]),
             va="center", fontsize=6.0)
ax1.grid(axis="x", alpha=0.25, lw=0.6)
ax1.annotate("equal |Tr U| to 9.5e-07:\nthe pair differs ONLY in content",
             xy=(5.172e-03, 0), xytext=(0.14, 0.55), fontsize=6.2, color="#7f0000",
             arrowprops=dict(arrowstyle="->", lw=0.9, color="#7f0000"))

# ---- Panel B: tau calibration ----
ax2.bar([0], [tau_comp], width=0.5, color="#2ca02c", label="compliant (store's own D_a)")
ax2.bar([1], [tau_inval], width=0.5, color="#d62728", label="invalid (equal-|Tr U|)")
ax2.axhline(0.35, ls="--", lw=1.3, color="#444444")
ax2.text(1.32, 0.365, "blueprint tau = 0.35\nLIES IN BAND", fontsize=6.4, color="#333333",
         ha="right")
ax2.set_xticks([0, 1])
ax2.set_xticklabels(["compliant", "invalid"], fontsize=7)
ax2.set_ylabel("delta_axiom (FORM B)")
ax2.set_ylim(0, 0.78)
ax2.set_title("B. tau calibration on the POPULATED store\n"
              "band (0.000000, 0.649520) contains 0.35")
for x, v in ((0, tau_comp), (1, tau_inval)):
    ax2.text(x, v + 0.02, "%.6f" % v, ha="center", fontsize=6.4)
ax2.legend(fontsize=6.0, loc="upper left", framealpha=0.9)
ax2.grid(axis="y", alpha=0.25, lw=0.6)

fig.suptitle("UHR-02: fix the comparison's DOMAIN, not the threshold "
             "(store population cannot repair a magnitude-only channel)",
             fontsize=8.8, y=0.995)
fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

out = pathlib.Path(__file__).resolve().parent / "uhr02_measured.png"
fig.savefig(out, dpi=110)
print("wrote", out)
print("figure px:", [int(x) for x in (fig.get_size_inches() * 110)])
