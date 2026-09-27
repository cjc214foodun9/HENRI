"""Figure: hybrid router CV A/B (K=3, held-out-fold selection) + alignment control.

Layout history (both defects found by vision inspection, 2026-09-27):
  v1  the alignment-control table was drawn INSIDE the right panel, where it
      covered the ridge bar.
  v2  fixed that, but the green "exact recovery" annotation sat at axes
      (0.05, 0.60) -- directly on the ridge bar's 0.575953 value label -- and the
      single-line provenance footer was wider than the figure, so it clipped to
      "eceipt ...".  Both fixed here.

This module renders only; the PNG stays untracked because .gitignore:19 is
`*.png`.  The repo convention (measured: 0 tracked files under docs/diagrams)
is to track the RENDERER, exactly as tools/render_action2_acceptance.py is.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = r"C:/Users/chan/henri-worktrees/zone-a-selfplay/HENRI V2"
REC = os.path.join(C, "experiments", "verification", "action2_hybrid_router_cv_observed.json")
CTL = os.path.join(C, "experiments", "verification", "action2_hybrid_router_alignment_control.json")
rec = json.load(open(REC, encoding="utf-8"))
ctl = json.load(open(CTL, encoding="utf-8"))
con, ref, a = rec["containment"], rec["reflection"], ctl["containment"]

arms = ["identity\n(no-op)", "ridge\n(fitted K=3)", "rigid D4\n(searched)", "HYBRID\n(CV-routed)"]
cols = ["#9e9e9e", "#4477aa", "#aa3377", "#228833"]

fig = plt.figure(figsize=(13.6, 7.8))
ax1 = fig.add_axes([0.055, 0.375, 0.395, 0.400])
ax2 = fig.add_axes([0.545, 0.375, 0.395, 0.400])
YM = (-0.04, 1.22)


def draw(ax, d, title):
    vals = [d["identity_delta"], d["ridge_delta"], d["rigid_delta"], d["hybrid_delta"]]
    bars = ax.bar(arms, vals, color=cols)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.025, "%.6f" % v,
                ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0.01, color="red", ls="--", lw=1.2)
    ax.text(3.42, 0.014, r"$\tau$=0.01", color="red", fontsize=8, ha="right", va="bottom")
    ax.axhline(1.0, color="#666666", ls=":", lw=1.0)
    ax.set_ylim(*YM)
    ax.set_ylabel("cos_delta on held pair (primary metric)")
    ax.set_title(title, fontsize=10)


draw(ax1, con,
     "CONTAINMENT (n=8)  —  router picks D4 8/8, correct 8/8\n"
     "pre-registered rule fires RECOVERS (0.121468 $\\geq\\tau$);\n"
     "alignment control: 12.1% of a solve")
draw(ax2, ref,
     "REFLECTION (n=8)  —  router picks D4 8/8, correct 8/8\n"
     "hybrid = rigid = 1.000000 exact\n"
     "(the class CAN reach the ceiling)")

ax1.text(0.5, 0.30, "same y-scale as the right panel", transform=ax1.transAxes,
         fontsize=8, ha="center", color="#555555",
         bbox=dict(boxstyle="round", fc="#f0f0f0", ec="#bbbbbb"))
ax1.annotate("12.1% of the class ceiling", xy=(3.0, 0.1215), xytext=(0.95, 0.72),
             fontsize=9, color="#aa3377",
             arrowprops=dict(arrowstyle="->", color="#aa3377", lw=1.2),
             bbox=dict(boxstyle="round", fc="#fff0f6", ec="#aa3377"))

# v2 DEFECT FIX: this annotation previously sat at (0.05, 0.60), on top of the
# ridge bar's 0.575953 label.  Moved to the empty upper-left region.
ax2.annotate("exact recovery:\nthe transform is IN the class",
             xy=(2.0, 1.0), xytext=(0.08, 0.80),
             fontsize=9, color="#228833",
             arrowprops=dict(arrowstyle="->", color="#228833", lw=1.2),
             bbox=dict(boxstyle="round", fc="#f0fff4", ec="#228833"))

sp = a["specificity_ratio"]
table = (
    "ALIGNMENT CONTROL (containment, n=8) — adjudicates the 0.121468 'RECOVERS' the pre-registered rule fired\n"
    "  mask IoU vs true interior %.4f   precision %.4f   recall %.4f   |  delta_own %.6f   delta_cross(other task) %.6f\n"
    "  ops touching interior %d/%d     ops changing nothing %d/%d     specificity own/cross %s     share of class ceiling %.1f%%\n"
    "  VERDICT  %s  ->  the routed operator changes ~100 cells to cover 4 (precision 0.033) and is only %s better on its OWN task than on another"
) % (a["mean_iou"], a["mean_precision"], a["mean_recall"], a["mean_delta_own"], a["mean_delta_cross"],
     a["n_overlap"], a["n"], a["n_identity_op"], a["n"], ("%.2fx" % sp) if sp else "inf",
     100.0 * a["mean_delta_own"], ctl["alignment_verdict"], ("%.2fx" % sp) if sp else "inf")

fig.text(0.5, 0.255, table, ha="center", va="top", family="monospace", fontsize=8.6,
         bbox=dict(boxstyle="round,pad=0.6", fc="#eef7ff", ec="#3366aa"))

# v2 DEFECT FIX: one long footer line clipped to "eceipt ...".  Split in two and
# start flush left so the box cannot exceed the figure width.
fig.text(0.045, 0.100,
         "receipt %s  |  verdict %s  |  router correct %s  |  runtime %.1f s  |  encodes %d  |  HEAD f320a29"
         % (os.path.basename(REC), rec["verdict"], rec["router_correct_total"],
            rec["runtime_s"], rec["candidate_encodes"]),
         ha="left", va="top", fontsize=7.8, color="#333333",
         bbox=dict(boxstyle="round,pad=0.4", fc="#fafafa", ec="#cccccc"))
fig.text(0.045, 0.056,
         "primary metric cos_delta = cos(pred-X, Y-X); a no-op scores exactly 0.0   |   "
         "alignment control status %s (thresholds chosen after the main run — the main receipt is byte-unchanged)"
         % ctl["status"],
         ha="left", va="top", fontsize=7.8, color="#333333",
         bbox=dict(boxstyle="round,pad=0.4", fc="#fafafa", ec="#cccccc"))

fig.text(0.5, 0.945,
         "Hybrid router CV A/B  |  K=3 demos  |  held-out-fold (leave-one-out) selection  |  tau=0.01  |  "
         "non-overlapping ring positions",
         ha="center", fontsize=11.5, fontweight="bold")

out = os.path.join(C, "docs", "diagrams", "hybrid_router_cv_result.png")
fig.savefig(out, dpi=140)
print("wrote %s %d bytes" % (out, os.path.getsize(out)))
