#!/usr/bin/env python
"""Render the Zone A self-play sprint figures (deterministic, matplotlib).

Produces two PNGs under docs/diagrams/:
  1. selfplay_dream_loop.png   -- the focused test-time dream engine loop
  2. zoneA_milestones.png      -- milestone dependency graph with gate status

Deterministic: no randomness, fixed layout, no timestamps in the image.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "diagrams")
os.makedirs(OUT, exist_ok=True)

C_OK = "#2e7d32"
C_BLOCK = "#b71c1c"
C_NEU = "#37474f"
C_ACC = "#1565c0"


def fig_dream_loop() -> str:
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("HENRI Focused Latent Dream Loop — test-time adaptation\n"
                 "every stage label carries its evidence class",
                 fontsize=13, fontweight="bold", color=C_NEU, pad=14)

    def box(x, y, w, h, text, color=C_NEU, fc="#f5f7fa", fs=9.5):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.12",
            linewidth=1.6, edgecolor=color, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=color, wrap=True)

    def arrow(x1, y1, x2, y2, label="", color=C_NEU, style="-|>"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=1.8))
        if label:
            ax.text((x1 + x2) / 2 + 0.12, (y1 + y2) / 2, label,
                    fontsize=8.2, color=color, style="italic")

    box(3.3, 8.7, 3.4, 0.9, "SENSORY INGRESS\nARC grid / unseen task", C_ACC, "#e3f2fd")
    arrow(5.0, 8.7, 5.0, 8.05)

    box(3.1, 7.1, 3.8, 0.9, "Sagnac homodyne gate  [OBSERVED code]\nsearch veto: delta > 0.35 -> branch pruned")
    arrow(5.0, 7.1, 5.0, 6.45)

    box(2.6, 5.4, 4.8, 1.0,
        "EPISTEMIC TRIGGER: novel topology\nsurprise high -> enter the dream (egress latched shut)",
        C_BLOCK, "#fdecea")
    arrow(2.6, 5.9, 1.6, 5.9, "split warps")
    arrow(7.4, 5.9, 8.4, 5.9, "split warps")

    box(0.3, 4.0, 2.6, 1.6,
        "GENERATOR warps\npropose transition\noperators in S^(D-1)\n[DOC-only mechanism]",
        C_ACC, "#e3f2fd", 8.8)
    box(7.1, 4.0, 2.6, 1.6,
        "VERIFIER warps\npredict trajectory\ncompute M1 alignment\nreward  [OBSERVED code]",
        C_OK, "#e8f5e9", 8.8)

    arrow(1.6, 4.0, 2.9, 3.35, "candidate")
    arrow(8.4, 4.0, 7.1, 3.35, "score")

    box(2.6, 2.4, 4.8, 0.95,
        "M1 reward: r = |< grad L , P_e @ d_theta >|\n"
        "zero on mastered data - zero on noise - max at the frontier",
        C_OK, "#e8f5e9", 9.2)
    arrow(5.0, 2.4, 5.0, 1.75)

    box(2.5, 0.75, 5.0, 1.0,
        "SGLD viscoelastic creep on low-rank adapters\n"
        "theta <- theta - eta grad L + sqrt(2 T dt) noise",
        C_NEU, "#eceff1", 9.2)

    arrow(2.5, 1.25, 0.9, 1.25, "vacuum", C_BLOCK)
    ax.text(0.15, 0.62, "dark port\n(Landauer heat)", fontsize=8, color=C_BLOCK)

    # return loop
    ax.annotate("", xy=(9.6, 5.8), xytext=(9.6, 1.25),
                arrowprops=dict(arrowstyle="-|>", color=C_ACC, lw=1.8,
                                connectionstyle="arc3,rad=-0.35"))
    ax.text(9.75, 3.4, "repeat until\nconsensus", fontsize=8.2, color=C_ACC,
            style="italic", rotation=90, va="center")

    box(3.0, 3.75, 4.0, 0.85, "EGRESS SNAP to action\n(only after consensus)",
        C_NEU, "#eceff1", 9.0)
    arrow(9.6, 1.25, 7.0, 4.17, "")

    legend = [
        mpatches.Patch(color=C_OK, label="live code path (measured / unit-tested)"),
        mpatches.Patch(color=C_ACC, label="proposed mechanism (document only)"),
        mpatches.Patch(color=C_BLOCK, label="veto / dissipation path"),
    ]
    ax.legend(handles=legend, loc="lower left", fontsize=8.4, frameon=True)
    ax.text(0.0, -0.02, "Gate: the dream update must change the SELECTED ACTION, not only a score.",
            fontsize=8.6, color=C_BLOCK, style="italic")

    path = os.path.join(OUT, "selfplay_dream_loop.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_milestones() -> str:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Zone A Self-Play Sprint — milestone dependency and gate status",
                 fontsize=13, fontweight="bold", color=C_NEU, pad=14)

    def node(x, y, w, h, text, color, fc):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.12",
            linewidth=1.8, edgecolor=color, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9.2, color=color)

    def arrow(x1, y1, x2, y2, color=C_NEU):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))

    node(0.4, 7.6, 4.3, 1.7,
         "M1  gradient-alignment reward\nhenri_gradient_alignment_reward.py\n"
         "GATE A1-A5  +  28/28 unit tests\nSTATUS: GREEN (local, +CPU)",
         C_OK, "#e8f5e9")

    node(5.5, 7.6, 4.1, 1.7,
         "M3  focused latent dreamer\nhenri_latent_dreamer.py\n"
         "GATE: dream changes the selected action\nSTATUS: DESIGNED, not built",
         C_ACC, "#e3f2fd")

    arrow(4.7, 8.45, 5.5, 8.45, C_OK)

    node(0.4, 4.7, 4.3, 1.6,
         "M2  Stage-0 universal seeder\nstage0_universal_seeder.py\n"
         "bounded seeding run BEFORE 10B-token scale\nSTATUS: BLOCKED (GPU budget)",
         C_BLOCK, "#fdecea")

    node(5.5, 4.7, 4.1, 1.6,
         "Sprint 3  Sagnac stage separation\n0.35 = search veto (keep)\n"
         "0.0431 = crystallization setpoint (pre-Zone-C)\nSTATUS: CONSTRAINT RECORDED",
         C_ACC, "#e3f2fd")

    arrow(4.7, 8.0, 2.55, 6.3, C_NEU)
    arrow(7.55, 7.6, 7.55, 6.3, C_ACC)

    node(2.4, 1.5, 5.2, 1.7,
         "M4  benchmark gauntlet on unseen topologies\n"
         "ARC-AGI-3 / SciCode window 48 / VRAM ceiling\n"
         "STATUS: BLOCKED - no score may be claimed",
         C_BLOCK, "#fdecea")
    arrow(2.55, 4.7, 4.2, 3.2, C_BLOCK)
    arrow(7.55, 4.7, 6.0, 3.2, C_BLOCK)

    ax.text(0.4, 0.5,
            "Falsified this sprint:  HOPE/Behrouz citation (no such work found)  |  "
            "'RTX 5090 = 192 SMs' (ships 170)",
            fontsize=8.8, color=C_BLOCK)
    ax.text(0.4, 0.12,
            "Corrected:  synthesis-document reward formula is HYPOTHESIS-form, not a measured reproduction",
            fontsize=8.8, color=C_NEU)

    path = os.path.join(OUT, "zoneA_milestones.png")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    p1 = fig_dream_loop()
    p2 = fig_milestones()
    print("WROTE", os.path.normpath(p1))
    print("WROTE", os.path.normpath(p2))
