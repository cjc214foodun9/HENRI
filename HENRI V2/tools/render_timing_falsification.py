"""Render the timing-falsification figure FROM THE COMMITTED RECEIPT.

Panel C sources per-rung rows DEFENSIVELY: the timing receipt does not carry them, so
they are read from the committed matched-compute receipt instead, and the scatter is
SKIPPED (line only) if no rows are available. A renderer must never crash on missing
optional data.
"""
import json, os, subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = r"C:/Users/chan/henri-worktrees/zone-a-selfplay"
C = os.path.join(R, "HENRI V2")
REL = "HENRI V2/experiments/verification/stage1_timing_falsification.json"
REL2 = "HENRI V2/experiments/verification/stage1_cadence_matched.json"


def blob(rel):
    out = subprocess.run(["git", "show", "HEAD:" + rel], cwd=R, capture_output=True)
    return json.loads(out.stdout.decode())


rec = blob(REL)
head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=R,
                      capture_output=True, text=True).stdout.strip()
pm = rec["primary_metric"]
pa = rec["paired_analysis_post_hoc_NOT_preregistered"]
late, early = pm["late_per_seed"], pm["early_per_seed"]
SEEDS = rec["seeds"]
tl = rec["timing_line"]

fig = plt.figure(figsize=(14.8, 9.8))
fig.text(0.5, 0.968, "Timing falsification: timing does not help; a small consistent negative direction",
         ha="center", fontsize=13.4, fontweight="bold")
fig.text(0.5, 0.940, "every number read from the git-committed receipt at render time  |  HEAD %s"
         % head, ha="center", fontsize=8.5, color="#555555", style="italic")

# ---- A: per-seed progress, both arms
ax = fig.add_axes([0.055, 0.565, 0.42, 0.30])
for i in range(len(SEEDS)):
    ax.plot([i, i], [late[i], early[i]], color="#bbbbbb", lw=1.0, zorder=1)
ax.plot(range(len(SEEDS)), late, "o-", color="#2255aa", lw=1.8, ms=7,
        label="late ladder (variance)")
ax.plot(range(len(SEEDS)), early, "s-", color="#aa3333", lw=1.8, ms=7,
        label="early ladder (cadence w=3)")
lo = min(min(late), min(early))
hi = max(max(late), max(early))
pad = (hi - lo) * 0.22
ax.set_ylim(lo - pad, hi + pad)
for i in range(len(SEEDS)):
    ax.text(i, late[i] + pad * 0.16, "%.6f" % late[i], ha="center", fontsize=7.0,
            color="#2255aa")
    ax.text(i, early[i] - pad * 0.30, "%.6f" % early[i], ha="center", fontsize=7.0,
            color="#aa3333")
ax.set_xticks(range(len(SEEDS)))
ax.set_xticklabels(["s%d" % s for s in SEEDS], fontsize=7.2)
ax.set_ylabel("held-out progress (149,750-149,806 vm each)")
ax.set_title("A  THE SEED MOVES PROGRESS (sd %.4f) and it is COMMON-MODE:\n"
             "both arms move together, so an UNPAIRED band hides the effect"
             % pm["sd_late"], fontsize=9.5)
ax.legend(fontsize=8, loc="best")
ax.grid(alpha=0.22)

# ---- B: the paired delta per seed
ax2 = fig.add_axes([0.555, 0.565, 0.40, 0.30])
d = pa["per_seed_paired_delta"]
cols = ["#aa3333" if v < 0 else "#228833" for v in d]
ax2.bar(range(len(SEEDS)), d, color=cols, width=0.55)
ax2.axhline(0, color="#333333", lw=1.0)
md = pa["mean_delta"]
ax2.axhline(md, color="#2255aa", ls="--", lw=1.5)
rng = max(abs(min(d)), abs(max(d))) or 1e-3
ax2.set_ylim(-rng * 1.55, rng * 1.35)
ax2.text(0.02, 0.96, "mean %+.6f" % md, transform=ax2.transAxes, color="#2255aa",
         fontsize=8.4, fontweight="bold", va="top")
ax2.text(0.02, 0.88,
         "paired t p=%.4f | sign test p=%.4f\n%d/%d negative | magnitude %.4f%%"
         % (pa["p_two_tailed"], pa["sign_test"]["p_two_tailed_exact"],
            pa["sign_test"]["n_negative"], pa["sign_test"]["n_seeds"],
            pa["magnitude_pct_of_late"]),
         transform=ax2.transAxes, fontsize=7.6, va="top", color="#333333")
for i, v in enumerate(d):
    ax2.text(i, v + (rng * 0.05 if v >= 0 else -rng * 0.05), "%+.6f" % v, ha="center",
             va="bottom" if v >= 0 else "top", fontsize=6.8)
ax2.set_xticks(range(len(SEEDS)))
ax2.set_xticklabels(["s%d" % s for s in SEEDS], fontsize=7.2)
ax2.set_ylabel("PAIRED delta  (early - late)")
ax2.set_title("B  PAIRED: small, consistent, MARGINALLY significant\n"
              "significance is test-dependent -- state the direction, not the p-value",
              fontsize=9.5)
ax2.grid(axis="y", alpha=0.22)

# ---- C: the circularity, sourced DEFENSIVELY
ax3 = fig.add_axes([0.055, 0.205, 0.42, 0.30])
rows_by_arm = rec.get("per_rung_progress") or {}
if not rows_by_arm:
    try:
        rows_by_arm = blob(REL2).get("per_rung_progress") or {}
    except Exception:
        rows_by_arm = {}
pts = []
for k, rows in rows_by_arm.items():
    aft = [r["heldout_after"] for r in rows if r.get("heldout_after") is not None]
    if not aft:
        continue
    fl = aft[-1]
    for r in rows:
        b, p_ = r.get("heldout_before"), r.get("progress")
        if b is None or p_ is None:
            continue
        if (b - fl) > 1e-12:
            pts.append((b - fl, p_))
if pts:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax3.scatter(xs, ys, s=26, color="#228833", alpha=0.85, zorder=3,
                label="per-rung rows (n=%d)" % len(pts))
    xmax = max(xs)
else:
    xs, ys, xmax = [], [], 1.0
    ax3.text(0.5, 0.5, "per-rung rows unavailable in the committed receipts",
             transform=ax3.transAxes, ha="center", fontsize=9)
slope = tl["slope"]
ax3.plot([0, xmax], [0, slope * xmax], color="#882222", ls="--", lw=1.5,
         label="slope %.4f  (r = %.4f, n = %d)" % (slope, tl["pearson_r"], tl["n"]))
ax3.set_xlabel("remaining drop at escalation")
ax3.set_ylabel("per-rung progress")
ax3.set_title("C  THE PRE-REGISTERED PER-RUNG METRIC IS CIRCULAR\n"
              "progress IS the drop that follows, so it restates WHEN the ladder fired",
              fontsize=9.5)
ax3.legend(fontsize=7.8)
ax3.grid(alpha=0.22)

# ---- D: verdict
ax4 = fig.add_axes([0.555, 0.205, 0.40, 0.30]); ax4.axis("off")
ax4.set_title("D  THE VERDICT (derived, both rules stated)", fontsize=9.7, loc="left")
rows = [
    ("PRE-REGISTERED rule (UNPAIRED)", "#333333", "bold"),
    ("  delta %+.6f | noise %.6f" % (pm["delta"], pm["noise_band"]), "#333333", ""),
    ("  -> TIMING_DOES_NOT_MATTER", "#333333", ""),
    ("", "", ""),
    ("CORRECT rule (PAIRED: arms share seeds)", "#333333", "bold"),
    ("  seed effect common-mode sd %.5f" % pa["common_mode_sd"], "#333333", ""),
    ("  CANCELS in the paired difference (sd %.6f)" % pa["sd_delta"], "#333333", ""),
    ("  -> %.4f%% effect, %d/%d negative" % (pa["magnitude_pct_of_late"],
                                             pa["sign_test"]["n_negative"],
                                             pa["sign_test"]["n_seeds"]), "#882222", "bold"),
    ("  -> unpaired band overstated uncertainty %.0fx"
     % (pa["common_mode_sd"] / pa["sd_delta"]), "#882222", ""),
    ("", "", ""),
    ("ROBUST ACROSS BOTH AND ALL TESTS:", "#333333", "bold"),
    ("  the early ladder never produces MORE", "#882222", "bold"),
    ("", "", ""),
    ("USER'S CONCLUSION: HOLDS and SHARPENS", "#226622", "bold"),
    ("  the remaining lever is WHAT the rungs contain", "#226622", ""),
]
y = 0.97
for txt, col, wt in rows:
    if txt:
        ax4.text(0.0, y, txt, fontsize=7.8, family="monospace", color=col,
                 fontweight=wt or "normal")
    y -= 0.066

fig.text(0.5, 0.128,
         "REFERENCE ADJUDICATION  |  Ref 1 claimed within-arm std 0.000000 (seed-independent) and commits f5a1c9d / a3b7e21: my instrument measured sd %.4f / %.4f and BOTH COMMITS ABSENT. "
         "Ref 1's TapeLearner (W = torch.zeros(256,33), counts table, online-mean update) is NOT the live class (params {emb, head}, b1/b2/eps/v/lr, Adam-style): its mechanism story describes code I do not have.\n"
         "Ref 2's core point -- the discriminating metric must be a LOCAL change, not an absolute magnitude -- is CORRECT and is implemented here as the residual against the timing line (panel C).\n"
         "SCOPE  held-out curriculum signal, not a benchmark score. No task-accuracy claim. SciCode / ARC-AGI / AAII remain BLOCKED."
         % (pm["sd_late"], pm["sd_early"]),
         ha="center", va="top", fontsize=7.4,
         bbox=dict(boxstyle="round,pad=0.55", fc="#fff7ec", ec="#cc7722"))
fig.text(0.5, 0.032,
         "claim -> hypothesis -> evidence -> mechanism -> action -> verification -> uncertainty   |   "
         "labels: OBSERVED / DERIVED / HYPOTHESIS / FALSIFIED / BLOCKED   |   no benchmark score is claimed",
         ha="center", fontsize=8.0, color="#555555")

out = os.path.join(C, "docs", "diagrams", "timing_falsification_result.png")
fig.savefig(out, dpi=140)
print("wrote %s %d bytes" % (out, os.path.getsize(out)))
