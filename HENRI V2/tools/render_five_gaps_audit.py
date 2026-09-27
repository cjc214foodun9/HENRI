#!/usr/bin/env python
"""Render the Five-Gaps premise audit + ACTION 1 live telemetry figure.

Deterministic matplotlib. Evidence labels are printed on the figure; it never
overrides a numeric artifact. Sources (all own measurements):
  HENRI V2/telemetry/stage0_10b/telemetry.jsonl   (the live local burn)
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(CODE, "docs", "diagrams")
TEL = os.path.join(CODE, "telemetry", "stage0_10b", "telemetry.jsonl")
os.makedirs(OUT, exist_ok=True)

C_OK, C_BAD, C_NEU, C_ACC, C_WARN = "#2e7d32", "#b71c1c", "#37474f", "#1565c0", "#ef6c00"

rows = []
if os.path.exists(TEL):
    for ln in open(TEL, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.22)

# ------------------------------------------------ panel 1: ACTION 1 progress
ax = fig.add_subplot(gs[0, 0])
if rows:
    r = [d["round"] for d in rows]
    tok = [d["learner_tokens"] for d in rows]
    ax.plot(r, tok, color=C_ACC, lw=1.6)
    ax.set_xlabel("round")
    ax.set_ylabel("learner tokens")
    tgt = 10_000_000_000
    ax.axhline(tgt, color=C_BAD, ls="--", lw=1.0)
    ax.text(r[0], tgt, "  10^10 target", color=C_BAD, fontsize=8, va="bottom")
    ax.set_title("ACTION 1 — local Stage-0 burn (OBSERVED)\n"
                 f"{rows[-1]['learner_tokens']:,} tokens  "
                 f"@ {rows[-1]['exec_per_sec']:,.0f} exec/s",
                 fontsize=10.5, fontweight="bold", color=C_NEU)
else:
    ax.text(0.5, 0.5, "no telemetry yet", ha="center", transform=ax.transAxes)
ax.grid(alpha=0.25)

# ------------------------------------------------ panel 2: premise audit
ax = fig.add_subplot(gs[0, 1])
ax.axis("off")
ax.set_title("Premise audit — 4 document claims vs live code", fontsize=10.5,
             fontweight="bold", color=C_NEU)
claims = [
    ("arc_task_functor: W=(X'X)^-1 X'Y", "FALSIFIED", "diagonal ridge; held-out 0.4368/60 tasks"),
    ("arc_sagnac_veto: cosine > 0.95", "FALSIFIED", "DEFAULT_EPSILON_HARD = 0.35; no '0.95'"),
    ("henri_wave_transducer.py missing", "OBSERVED", "confirmed ABSENT"),
    ("benchmark --benchmark scicode --window 48", "FALSIFIED", "0 argparse flags; HumanEval only"),
]
y = 0.86
for claim, verdict, detail in claims:
    col = C_BAD if verdict == "FALSIFIED" else C_WARN
    ax.text(0.01, y, verdict, fontsize=8.6, fontweight="bold", color=col)
    ax.text(0.20, y, claim, fontsize=8.4, color=C_NEU)
    ax.text(0.20, y - 0.055, detail, fontsize=7.6, style="italic", color=C_NEU)
    y -= 0.155
ax.text(0.01, 0.24, "3 of 4 structural claims were misattributed.\n"
                    "Refactoring toward a misdescribed target destroys working code.",
        fontsize=8.2, color=C_BAD, style="italic")
ax.text(0.01, 0.06, "every verdict is an own-tool read at HEAD 3ff5cb5", fontsize=7.8, color=C_ACC)

# ------------------------------------------------ panel 3: prior art resonator
ax = fig.add_subplot(gs[1, 0])
scen = ["solvable", "identity\n(truth)", "impossible", "REAL ARC"]
treat = [1.0000001, 1.0000000, -0.00146, 0.26858]
ctrl = [0.3672, 1.0000000, 0.0, 0.26858]
x = range(len(scen))
w = 0.38
ax.bar([i - w / 2 for i in x], treat, w, label="resonator (treatment)",
       color=C_ACC, edgecolor="black", lw=0.6)
ax.bar([i + w / 2 for i in x], ctrl, w, label="closest control",
       color=C_WARN, edgecolor="black", lw=0.6)
ax.set_xticks(list(x))
ax.set_xticklabels(scen, fontsize=8.5)
ax.set_ylabel("score")
ax.set_title("DECISIVE PRIOR ART — the resonator already exists and returned VOID\n"
             "carrier/aaii-v43 @ 16d573c  (instrument 28/28, mutation 4/4)", fontsize=10.5,
             fontweight="bold", color=C_NEU)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.25)
ax.annotate("treatment LOSES\nto identity", xy=(3 + w / 2, 0.26858), xytext=(2.35, 0.78),
            fontsize=8, color=C_BAD,
            arrowprops=dict(arrowstyle="->", color=C_BAD, lw=1.4))

# ------------------------------------------------ panel 4: re-scoped status
ax = fig.add_subplot(gs[1, 1])
ax.axis("off")
ax.set_title("Three-directive disposition (evidence-labelled)", fontsize=10.5,
             fontweight="bold", color=C_NEU)
disp = [
    ("ACTION 1", "RUNNING", C_OK, "held-out gate + shards built, smoke EXACT,\nlocal burn live at 0 GPU cost"),
    ("ACTION 2", "RE-SCOPED", C_WARN, "prior art VOID; default-OFF third arm\nunder pre-registered A/B vs diag_ls"),
    ("ACTION 3", "BLOCKED", C_BAD, "FUWT absent; local code backbone ABSENT\nin worktree; cited CLI cannot run"),
]
y = 0.80
for name, verdict, col, detail in disp:
    ax.text(0.02, y, name, fontsize=9.4, fontweight="bold", color=C_NEU)
    ax.text(0.24, y, verdict, fontsize=9.4, fontweight="bold", color=col)
    ax.text(0.02, y - 0.075, detail, fontsize=8.0, style="italic", color=C_NEU)
    y -= 0.235
ax.text(0.02, 0.10,
        "Promotion gate = HELD-OUT progress ONLY.\n"
        "final_loss / reward_mean are recorded but are NOT gates.",
        fontsize=8.4, color=C_BAD, fontweight="bold")
ax.text(0.02, 0.95, "no benchmark score, no ICL claim is made", fontsize=8,
        style="italic", color=C_ACC)

fig.suptitle("HENRI — Five Structural Gaps: premise audit and directive disposition  "
             "(HEAD 3ff5cb5, own measurements)",
             fontsize=12, fontweight="bold", color=C_NEU)

p = os.path.join(OUT, "five_gaps_audit.png")
fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("WROTE", p)
