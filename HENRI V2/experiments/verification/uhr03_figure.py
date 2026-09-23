"""UHR-03 measured figure: WHEN does FORM B separate from FORM A?

Deterministic. Fixed seeds, fixed palette, fixed DPI, no randomness at render.
Reproduces the table in `uhr03_verdict.md` section 3b by re-deriving it, so the
figure and the verdict cannot drift apart.

CLAIM (HYPOTHESIS until this figure was rendered, then DERIVED):
    FORM B (compare against the RECORDED transition) degenerates onto FORM A
    (compare against the current state) whenever the recorded transition is
    near-identity, because U_t = exp(i theta.lambda) -> I as ||theta|| -> 0.
    It only becomes a distinct comparison domain above the K=8192 sampling band.

The OBSERVED live magnitude from the UHR-03 paired A/B is 1.8106915149473934e-06
-- a float32 residue of the identity, five orders below the first separating
magnitude. That is why "populate the store" was the wrong repair.
"""
import itertools
import math
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

SRC = r"C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2"
sys.path.insert(0, SRC)
import uhr02_exteroceptive_gate as G  # noqa: E402

OUT = pathlib.Path(SRC) / "experiments" / "verification" / "uhr03_measured.png"
K = 8192
OBSERVED_NORM = 1.8106915149473934e-06   # live, both arms, all 16 steps
BAND = G.sampling_band(K)


def gm_basis() -> torch.Tensor:
    """Gell-Mann lambda_1..lambda_8, [8,3,3] complex (production convention)."""
    L = []
    for j, k in itertools.combinations(range(3), 2):
        S = torch.zeros(3, 3, dtype=torch.complex64)
        S[j, k] = 1; S[k, j] = 1; L.append(S)
        A = torch.zeros(3, 3, dtype=torch.complex64)
        A[j, k] = -1j; A[k, j] = 1j; L.append(A)
    d = torch.zeros(3, 3, dtype=torch.complex64); d[0, 0] = 1; d[1, 1] = -1
    L.append(d)
    d2 = torch.zeros(3, 3, dtype=torch.complex64)
    d2[0, 0] = 1; d2[1, 1] = 1; d2[2, 2] = -2
    L.append(d2 / math.sqrt(3))
    return torch.stack(L)


def gens(mag: float, seed: int) -> list:
    g = torch.Generator().manual_seed(seed)
    th = torch.randn(8, generator=g)
    th = th / th.norm() * mag
    return [1j * sum(th[k] * B[k] for k in range(8))]


B = gm_basis()
gR = torch.Generator().manual_seed(0)
R = torch.randn(K, 8, generator=gR)
R = R / R.norm(dim=-1, keepdim=True)

MAGS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 3e-1, 6e-1]
rows = []
for m in MAGS:
    gc = gens(0.30, 11)          # candidate HELD FIXED (hold the operator fixed)
    gt = gens(m, 22)             # reference VARIED (the discriminating control)
    A_c, A_t = G.ad_of(gc, B), G.ad_of(gt, B)
    pc, pt = G.predict_next(R, A_c), G.predict_next(R, A_t)
    dA = G.delta(pc, R)          # FORM A
    dB = G.delta(pc, pt)         # FORM B
    rows.append((m, dA, dB, abs(dB - dA)))

first = next((m for m, _, _, g in rows if g > BAND), None)

fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=110)
ax.axhspan(0, BAND, color="#d9d9d9", alpha=0.55, zorder=0)
ax.text(MAGS[0] * 1.4, BAND * 0.34, "K=8192 sampling band  sigma=2.471e-03\nbelow this: FORM B is indistinguishable from FORM A",
        fontsize=7.5, color="#444444", va="center")

ax.plot(MAGS, [r[1] for r in rows], "o-", color="#1f4e79", lw=1.7, ms=4.5,
        label="FORM A  delta(pred_c, state)   -- magnitude-only")
ax.plot(MAGS, [r[2] for r in rows], "s-", color="#c0392b", lw=1.7, ms=4.5,
        label="FORM B  delta(pred_c, RECORDED transition)  -- content")
ax.plot(MAGS, [r[3] for r in rows], "^--", color="#7d3c98", lw=1.3, ms=4.0,
        label="|FORM B - FORM A|   (collapse when below band)")

ax.axvline(OBSERVED_NORM, color="#000000", lw=1.2, ls=":")
ax.annotate("OBSERVED live\n||theta||=1.81e-06\nU_t = I + O(1.8e-6)\n(identity residue,\nno transition)",
            xy=(OBSERVED_NORM, 0.10), xytext=(OBSERVED_NORM * 3.2, 0.115),
            fontsize=7.5, color="#000000",
            arrowprops=dict(arrowstyle="->", color="#000000", lw=1.0))
if first:
    ax.axvline(first, color="#117a65", lw=1.1, ls="-.")
    ax.annotate("first separation\n||theta||=1e-01", xy=(first, 0.033),
                xytext=(first * 0.30, 0.0175), fontsize=7.5, color="#117a65",
                arrowprops=dict(arrowstyle="->", color="#117a65", lw=1.0))

ax.set_xscale("log")
ax.set_xlabel("recorded transition magnitude   ||theta||   (log scale)", fontsize=9)
ax.set_ylabel("Sagnac-style residual  delta", fontsize=9)
ax.set_title("UHR-03: the exteroceptive comparison domain collapses below ||theta||~0.05\n"
             "carrier a0a9e4e | instance 52289752 | paired A/B, both arms exit 0 | DERIVED, seeds fixed",
             fontsize=9.5)
ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
ax.legend(fontsize=7.5, loc="upper left", framealpha=0.92)
ax.set_ylim(0.0, max(r[2] for r in rows) * 1.12)
fig.tight_layout()
fig.savefig(OUT)
print("WROTE", OUT)
print("band =", BAND)
for m, dA, dB, g in rows:
    print(f"  {m:9.1e}  A={dA:.9f}  B={dB:.9f}  |B-A|={g:.3e}  "
          f"{'COLLAPSED' if g < BAND else 'DISTINCT'}")
print("first separating magnitude:", first)
