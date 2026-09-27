#!/usr/bin/env python
"""Stage-0 result figure: measured separation + the three budgets + the cost model.

Deterministic matplotlib (fixed seed/DPI). Evidence labels are printed on the
figure; it never overrides a numeric artifact.

Sources (all own measurements):
  telemetry/vast_52826640/telemetry/stage0_seeding/summary.json   (instance 52826640)
  telemetry/vast_52826640/final/telemetry/reward_norm/summary.json
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "diagrams")
SEED = os.path.join(ROOT, "telemetry", "vast_52826640", "telemetry", "stage0_seeding",
                    "summary.json")
NORM = os.path.join(ROOT, "telemetry", "vast_52826640", "final", "telemetry",
                    "reward_norm", "summary.json")
os.makedirs(OUT, exist_ok=True)

C_OK, C_BAD, C_NEU, C_ACC = "#2e7d32", "#b71c1c", "#37474f", "#1565c0"

seed = json.load(open(SEED, encoding="utf-8")) if os.path.exists(SEED) else {}
norm = json.load(open(NORM, encoding="utf-8")) if os.path.exists(NORM) else {}
r0 = (norm.get("runs") or {}).get("0", {})

fig = plt.figure(figsize=(14, 8.5))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)

# ------------------------------------------------ panel 1: reward variant ordering
ax = fig.add_subplot(gs[0, :2])
fams = ["MASTERED", "FRONTIER", "NOISE"]
raw = [r0.get(f, {}).get("reward", {}).get("RAW", 0) for f in fams]
cosf = [r0.get(f, {}).get("reward", {}).get("COS", 0) for f in fams]
x = range(len(fams))
w = 0.38
ax.bar([i - w / 2 for i in x], raw, w, label="RAW  |<g, P·dtheta>|  (SHIPPED)",
       color=C_OK, edgecolor="black", linewidth=0.6)
ax.bar([i + w / 2 for i in x], cosf, w, label="COS  normalised (REJECTED)",
       color=C_BAD, edgecolor="black", linewidth=0.6)
ax.set_xticks(list(x))
ax.set_xticklabels(fams)
ax.set_ylabel("reward (seed 0, CUDA)")
ax.set_title("Decisive probe on CUDA: RAW SEPARATES, normalising destroys it\n"
             "heldout_progress_axis_exists = %s" % norm.get("heldout_progress_axis_exists"),
             fontsize=11, fontweight="bold", color=C_NEU)
ax.legend(fontsize=8.5)
for i, v in enumerate(raw):
    ax.annotate(f"{v:.2e}", (i - w / 2, v), ha="center", va="bottom", fontsize=8, color=C_OK)
ax.grid(axis="y", alpha=0.25)

# ------------------------------------------------ panel 2: verdict matrix
ax = fig.add_subplot(gs[0, 2])
ax.axis("off")
ax.set_title("Variant verdicts", fontsize=11, fontweight="bold", color=C_NEU)
vb = norm.get("verdict_by_variant", {})
ax.text(0.02, 0.80, "variant", fontsize=9, fontweight="bold")
ax.text(0.42, 0.80, "f>mast", fontsize=9, fontweight="bold")
ax.text(0.66, 0.80, "f>noise", fontsize=9, fontweight="bold")
ax.text(0.90, 0.80, "SEP", fontsize=9, fontweight="bold")
y = 0.68
for var in ("RAW", "COS", "COSV"):
    d = vb.get(var, {})
    col = C_OK if d.get("SEPARATES") else C_BAD
    ax.text(0.02, y, var, fontsize=9.5, color=col, fontweight="bold")
    for i, k in enumerate(("frontier_gt_mastered", "frontier_gt_noise", "SEPARATES")):
        ax.text(0.42 + i * 0.24, y, str(d.get(k)), fontsize=8.5,
                color=C_OK if d.get(k) else C_BAD)
    y -= 0.13
ax.text(0.02, 0.24, "RAW = as-implemented M1 form.\nNormalising is a REGRESSION.",
        fontsize=8.4, style="italic", color=C_NEU)
ax.text(0.02, 0.06, "verdict is CUDA-OBSERVED (instance 52826640)", fontsize=8, color=C_ACC)

# ------------------------------------------------ panel 3-4-5: the three budgets
for idx, (title, key, unit, per_sec) in enumerate([
    ("VM executions", "budget_vm_executions", "program runs", "exec_per_sec"),
    ("Reward evaluations", "budget_reward_evaluations", "JVP/grad evals", "reward_evals_per_sec"),
    ("Learner tokens", "budget_learner_tokens", "next-byte tokens", "learner_tokens_per_sec"),
]):
    ax = fig.add_subplot(gs[1, idx])
    v = seed.get(key, 0)
    rate = seed.get(per_sec, 0)
    ax.bar(["measured"], [v], color=C_ACC, edgecolor="black", linewidth=0.6)
    ax.set_title(f"{title}", fontsize=10, fontweight="bold", color=C_NEU)
    ax.set_ylabel(unit)
    ax.annotate(f"{v:,}\n@ {rate:,.1f}/s", (0, v), ha="center", va="bottom", fontsize=8.6)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(labelsize=8)

fig.suptitle("HENRI Stage-0 bounded seeding — CUDA-observed (instance 52826640, HEAD d056e22)\n"
             "loss %.6f -> %.6f   timeout_rate %.4f   bank %d   distinct %d   wall %.1f s"
             % (seed.get("first_loss", 0), seed.get("final_loss", 0),
                seed.get("timeout_rate", 0), seed.get("bank_size", 0),
                seed.get("distinct_outputs", 0), seed.get("wall_seconds", 0)),
             fontsize=11.5, fontweight="bold", color=C_NEU)

p = os.path.join(OUT, "stage0_measured_result.png")
fig.savefig(p, dpi=155, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("WROTE", p)
