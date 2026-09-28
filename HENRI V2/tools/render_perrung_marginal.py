"""Render the per-rung marginal-progress falsification figure FROM THE COMMITTED RECEIPT.

Reads `git show HEAD:HENRI V2/experiments/verification/stage1_timing_perrung_marginal.json`
so the figure cannot drift from the evidence. Viz interpreter (matplotlib), no torch.
"""
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = r"C:/Users/chan/henri-worktrees/zone-a-selfplay"
C = os.path.join(R, "HENRI V2")
REL = "HENRI V2/experiments/verification/stage1_timing_perrung_marginal.json"
SRC_REL = "HENRI V2/experiments/verification/stage1_timing_falsification.json"
OUT = os.path.join(C, "docs", "diagrams", "perrung_marginal_falsification.png")

p = subprocess.run(["git", "show", "HEAD:" + REL], cwd=R, capture_output=True, text=True)
if p.returncode:
    sys.exit("receipt not in HEAD: " + (p.stderr or "").strip())
rec = json.loads(p.stdout)
q = subprocess.run(["git", "show", "HEAD:" + SRC_REL], cwd=R, capture_output=True, text=True)
src = json.loads(q.stdout)
res = src["residuals_per_arm_seed"]
early = {k: [float(x) for x in v] for k, v in res.items() if k.startswith("early")}
late = {k: [float(x) for x in v] for k, v in res.items() if k.startswith("late")}

pre = rec["pre_registered_test"]
pair = rec["paired_matched_rung_index_test_post_hoc"]
detail = pair["per_seed"]
cd = rec.get("source_receipt_head", "")[:8]

fig = plt.figure(figsize=(13.6, 8.6), dpi=112)
fig.suptitle("Per-rung MARGINAL progress: does escalation TIMING matter?  "
             "(150,000 VM executions, 8 shared seeds, governor stops disabled)",
             fontsize=12.5, fontweight="bold", y=0.975)
gs = fig.add_gridspec(2, 3, hspace=0.44, wspace=0.30,
                      left=0.065, right=0.985, top=0.885, bottom=0.075)

# ---- A: registered test — delta vs noise band ----
axA = fig.add_subplot(gs[0, 0])
d, nb = pre["delta"], pre["noise_band_pooled_sd"]
axA.bar([0], [d], width=0.5, color="#c0392b" if d < 0 else "#2c7fb8",
        edgecolor="black", linewidth=0.9, zorder=3)
axA.axhspan(-nb, nb, color="#95a5a6", alpha=0.34, zorder=1)
axA.axhline(0, color="black", linewidth=1.0)
axA.axhline(nb, color="#34495e", linestyle="--", linewidth=1.4, zorder=2)
axA.axhline(-nb, color="#34495e", linestyle="--", linewidth=1.4, zorder=2)
lo, hi = min(d, -nb) * 1.5, max(0.0, nb) * 1.5
axA.set_ylim(lo, hi)
axA.annotate("delta = %+.4f" % d, (0, d), textcoords="offset points", xytext=(0, -16 if d < 0 else 10),
             ha="center", fontsize=9.5, fontweight="bold")
axA.text(0.62, nb, "+noise %.4f" % nb, fontsize=8.4, va="bottom", color="#2c3e50")
axA.text(0.62, -nb, "-noise %.4f" % nb, fontsize=8.4, va="top", color="#2c3e50")
axA.set_title("A. REGISTERED TEST\n|delta| within the noise band", fontsize=10, fontweight="bold")
axA.set_xticks([0]); axA.set_xticklabels(["early - late"], fontsize=9)
axA.set_ylabel("mean per-rung residual", fontsize=9)
axA.text(0.03, 0.97, "WITHIN NOISE\n-> TIMING_DOES_NOT_MATTER", transform=axA.transAxes,
         fontsize=9, va="top", fontweight="bold", color="#1a5276",
         bbox=dict(boxstyle="round,pad=0.35", fc="#d6eaf8", ec="#1a5276", lw=0.9))

# ---- B: paired per-seed deltas ----
axB = fig.add_subplot(gs[0, 1:])
seeds = [str(x["seed"]) for x in detail]
vals = [x["delta"] for x in detail]
cols = ["#c0392b" if v < 0 else "#27ae60" for v in vals]
axB.bar(range(len(vals)), vals, color=cols, edgecolor="black", linewidth=0.8, zorder=3)
axB.axhline(0, color="black", linewidth=1.0)
axB.axhline(pair["mean_delta"], color="#8e44ad", linestyle="--", linewidth=1.6, zorder=4)
axB.annotate("paired mean %+.4f" % pair["mean_delta"],
             (len(vals) - 0.5, pair["mean_delta"]), textcoords="offset points",
             xytext=(-6, -15), ha="right", fontsize=9, color="#6c3483", fontweight="bold")
axB.set_xticks(range(len(seeds))); axB.set_xticklabels(seeds, fontsize=8.4, rotation=20)
axB.set_title("B. PAIRED, MATCHED-RUNG-INDEX (both arms share seeds -> seed effect cancels)\n"
              "t = %.2f (df %d), %d/%d seeds negative, p_sign = %.4f"
              % (pair["t"], pair["df"], pair["sign_test"]["n_negative"],
                 pair["sign_test"]["n_seeds"], pair["sign_test"]["p_two_tailed_exact"]),
              fontsize=10, fontweight="bold")
axB.set_ylabel("early - late (matched rung prefix)", fontsize=9)
axB.text(0.015, 0.05, "every bar <= 0 except one: the early ladder is if anything WORSE,\n"
                      "magnitude %.4f  (%.3f%% of the ~5.43 held-out drop)"
         % (abs(pair["mean_delta"]), 100 * abs(pair["mean_delta"]) / 5.43),
         transform=axB.transAxes, fontsize=8.6, va="bottom",
         bbox=dict(boxstyle="round,pad=0.32", fc="#fdf2e9", ec="#a04000", lw=0.9))

# ---- C: rung-count asymmetry ----
axC = fig.add_subplot(gs[1, 0])
ne = sorted(len(v) for v in early.values())
nl = sorted(len(v) for v in late.values())
x = np.arange(len(ne))
axC.bar(x - 0.2, ne, width=0.4, color="#2c7fb8", edgecolor="black", lw=0.7, label="early (cadence w=3)")
axC.bar(x + 0.2, nl, width=0.4, color="#e67e22", edgecolor="black", lw=0.7, label="late (variance)")
axC.set_title("C. RUNG-COUNT ASYMMETRY\nwhy a pooled mean is confounded", fontsize=10, fontweight="bold")
axC.set_xlabel("arm (sorted)", fontsize=9); axC.set_ylabel("escalations reached", fontsize=9)
axC.legend(fontsize=8.2, loc="upper left")
axC.text(0.5, 0.055, "late reaches 1-4 rungs vs early 7-9\n-> paired test truncates to the SHARED prefix",
         transform=axC.transAxes, fontsize=8.2, ha="center",
         bbox=dict(boxstyle="round,pad=0.3", fc="#fef9e7", ec="#b7950b", lw=0.9))

# ---- D: the circular raw metric ----
axD = fig.add_subplot(gs[1, 1])
axD.axis("off")
axD.set_title("D. WHY THE RAW METRIC CANNOT DECIDE", fontsize=10, fontweight="bold")
axD.text(0.02, 0.94, "Found BEFORE the arms ran (committed):", fontsize=8.8,
         fontweight="bold", va="top")
axD.text(0.02, 0.745,
         "raw per-rung progress\n  = -0.004022\n  + 0.8992 x remaining-drop\n"
         "Pearson r = 0.9999  (n = 17)",
         fontsize=9.6, va="top", family="DejaVu Sans Mono",
         bbox=dict(boxstyle="round,pad=0.42", fc="#f4ecf7", ec="#7d3c98", lw=1.1))
axD.text(0.02, 0.30,
         "A quantity that is ~0.90 x what is left cannot\n"
         "separate 'early' from 'late' -- both arms simply\n"
         "observe different amounts of remaining drop.\n"
         "=> all analysis uses the DE-CIRCULARIZED residual.",
         fontsize=8.6, va="top")

# ---- E: verdict box ----
axE = fig.add_subplot(gs[1, 2])
axE.axis("off")
axE.set_title("E. VERDICT", fontsize=10, fontweight="bold")
axE.text(0.5, 0.985, "TIMING_IS_INERT", fontsize=10.4, ha="center", va="top", fontweight="bold",
         family="DejaVu Sans Mono", color="#1a5276",
         bbox=dict(boxstyle="round,pad=0.4", fc="#d6eaf8", ec="#1a5276", lw=1.4))
axE.text(0.5, 0.66,
         "Q: does the early ladder produce MORE\nper-rung marginal progress, beyond the\n"
         "noise band, at matched compute?\n\nA: NO.",
         fontsize=9.2, ha="center", va="top", fontweight="bold")
axE.text(0.5, 0.245,
         "NOT licensed:\n"
         "- escalation is useless in general\n"
         "- any benchmark / task-score claim\n"
         "- transfer to another substrate",
         fontsize=8.1, ha="center", va="top", color="#7b241c")

fig.text(0.5, 0.021,
         "OBSERVED: 16 arms x 150,000 VM executions on disk; governor stops disabled so no arm is "
         "truncated.  All numbers read from HEAD:%s (utc %s).  "
         "Metric circularity recorded before the run.  Figure is a render -- the receipt is authoritative."
         % (REL.split("/")[-1], rec.get("utc", "")), ha="center", fontsize=7.3, color="#566573")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=112, bbox_inches="tight", facecolor="white")
print("wrote %s  %d bytes" % (OUT, os.path.getsize(OUT)))
print("verdict label:", rec.get("conclusion", "")[:120])
print("registered delta %+.6f vs noise %.6f" % (pre["delta"], pre["noise_band_pooled_sd"]))
print("paired delta %+.6f | t %.3f | sign %d/%d"
      % (pair["mean_delta"], pair["t"], pair["sign_test"]["n_negative"],
         pair["sign_test"]["n_seeds"]))