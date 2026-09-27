"""Figure: 3-channel router CV A/B + the pre-registered control table.

Layout discipline (three defect classes caught by vision inspection earlier in
this session): annotations go in EMPTY plot space, never on a bar or its value
label; the control table is drawn OUTSIDE the axes; footer lines are split and
flushed left so they cannot exceed the figure width.

The PNG stays untracked -- .gitignore:19 is `*.png`.  Repo convention (measured:
0 tracked files under docs/diagrams) is to track the RENDERER.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C = r"C:/Users/chan/henri-worktrees/zone-a-selfplay/HENRI V2"
REC = os.path.join(C, "experiments", "verification", "action2_topo3_router_cv_observed.json")
d = json.load(open(REC, encoding="utf-8"))
con, ref, ctl = d["containment"], d["reflection"], d["controls_result"]
pr = d["preregistration"]

arms = ["identity\n(no-op)", "ridge\n(fitted K=3)", "rigid D4\n(searched)",
        "TOPO\n(local, 0 params)", "HYBRID\n(CV-routed)"]
cols = ["#9e9e9e", "#4477aa", "#aa3377", "#cc7722", "#228833"]

fig = plt.figure(figsize=(14.2, 8.4))
ax1 = fig.add_axes([0.045, 0.425, 0.42, 0.365])
ax2 = fig.add_axes([0.545, 0.425, 0.42, 0.365])
YM = (-0.06, 1.30)


def draw(ax, fam, title):
    vals = [0.0, fam["ridge_delta"], fam["rigid_delta"], fam["topo_delta"], fam["hybrid_delta"]]
    bars = ax.bar(arms, vals, color=cols, edgecolor="#333333", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.028, "%.6f" % v,
                ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0.01, color="red", ls="--", lw=1.2)
    ax.text(4.45, 0.016, r"$\tau$=0.01", color="red", fontsize=8, ha="right", va="bottom")
    ax.axhline(1.0, color="#666666", ls=":", lw=1.0)
    ax.set_ylim(*YM)
    ax.set_ylabel("cos_delta on held pair (primary metric)")
    ax.set_title(title, fontsize=10)
    return vals


draw(ax1, con,
     "CONTAINMENT (n=8)  —  router picks TOPO 8/8\n"
     "TOPO recovers the transform EXACTLY;\n"
     "the 2-channel pool could not (D4 0.1158, ridge 0.0829)")
draw(ax2, ref,
     "REFLECTION (n=8)  —  router picks D4 8/8\n"
     "TOPO ABSTAINS 8/8 on a transform whose changed set\n"
     "is the whole grid, so the global channel is selected")

# annotations in empty space only
ax1.annotate("TOPO = 1.000000 exact\n(mask 0 params, fill from demos)",
             xy=(3.0, 1.0), xytext=(0.30, 0.62), fontsize=9, color="#8a4b00",
             arrowprops=dict(arrowstyle="->", color="#cc7722", lw=1.4),
             bbox=dict(boxstyle="round", fc="#fff7ec", ec="#cc7722"))
ax1.text(4.45, 0.70, "class ceiling\n(transform contained)", fontsize=7.5, ha="right",
         va="center", color="#555555")
ax2.annotate("exact recovery:\ntransform in class",
             xy=(2.0, 0.99), xytext=(2.95, 0.46),
             fontsize=8.5, color="#228833",
             arrowprops=dict(arrowstyle="->", color="#228833", lw=1.4),
             bbox=dict(boxstyle="round", fc="#f0fff4", ec="#228833"))
ax2.text(4.45, 0.16, "TOPO delta 0.000000\n(abstained, not routed)", fontsize=7.5,
         ha="right", va="center", color="#8a4b00")

table = (
    "PRE-REGISTERED CONTROLS   (bounds MEASURED before the run; fixed so they cannot move after)\n"
    "  K1  held mask IoU > 0.5 .................. %8.4f   %s     (colour-agnostic: does the support match?)\n"
    "  K2  colour specificity >= %-4s .......... %8.3f   %s     (supplied own/cross > 3x is UNREACHABLE: measured cap 1.699x)\n"
    "  K3  mask specificity >= %-4s ............ %8.3f   %s     (was EXACTLY 1.000 on 3/8 tasks before the K3 fix)\n"
    "  K4  off-family inertness == 0.0 exactly ... %8.6f   %s     (open curve + no curve, max over 6 grids)\n"
    "  reflection ceiling D4 == 1.000000 exactly .. %s     abstention on reflection: %d/%d\n"
    "\n"
    "FIXTURE DEFECTS FOUND AND FIXED BEFORE THE RUN (disclosed, pre-registered)\n"
    "  D1 fill colour was UNOBSERVABLE (held fill absent from all demos; 5/5 tasks) -> now a TASK-LEVEL CONSTANT, band {10..17}\n"
    "  D2 ring colours intersected the noise band ({1,2} of {0,1,2}) -> disjoint bands: noise {0,1,2}, ring {4..9}, fill {10..17}\n"
) % (con["held_iou_mean"], ctl["K1_iou"],
     pr["controls"]["K2_colspec_min"], con["colspec_mean"], ctl["K2_colspec"],
     pr["controls"]["K3_maskspec_min"], con["maskspec_mean"], ctl["K3_maskspec"],
     ctl["off_family_max_delta"], ctl["K4_inertness"],
     ctl["reflection_ceiling_exact"], ref["topo_abstained"], ref["n"])
fig.text(0.5, 0.345, table, ha="center", va="top", family="monospace", fontsize=8.4,
         bbox=dict(boxstyle="round,pad=0.6", fc="#eef7ff", ec="#3366aa"))

fig.text(0.045, 0.115,
         "receipt %s  |  verdict %s  |  runtime %.1f s  |  encodes %d  |  K=3 demos, tau=0.01, "
         "leave-one-out CV, capacities TOPO<D4<RIDGE"
         % (os.path.basename(REC), d["verdict"], d["runtime_s"], d["candidate_encodes"]),
         ha="left", va="top", fontsize=7.8, color="#333333",
         bbox=dict(boxstyle="round,pad=0.4", fc="#fafafa", ec="#cccccc"))
fig.text(0.045, 0.070,
         "primary metric cos_delta = cos(pred-X, Y-X); a no-op scores exactly 0.0   |   synthetic grids, "
         "internal-representation recovery on the first held pair per task -- NOT an ARC or SciCode score",
         ha="left", va="top", fontsize=7.8, color="#333333",
         bbox=dict(boxstyle="round,pad=0.4", fc="#fafafa", ec="#cccccc"))

fig.text(0.5, 0.955,
         "3-channel router CV A/B: rigid D4 + per-slot diagonal ridge + LOCAL topological interior fill",
         ha="center", fontsize=12, fontweight="bold")

out = os.path.join(C, "docs", "diagrams", "topo3_router_cv_result.png")
fig.savefig(out, dpi=140)
print("wrote %s %d bytes" % (out, os.path.getsize(out)))
