"""K-BETA PROBE: is beta* a MECHANISM for the Lexical Snap, or decoration?

CLAIM UNDER TEST
    Blueprint HENRI-ARCH-2026-SYSTEMIC-EVALUATION-V3, sec 2.4 and Stage 5:
      "Calibrate the inverse temperature parameter to the empirical optimum:
       beta* = 26.10 (T* = 0.038316)"
      "This configuration prevents mutual information collapse and guarantees
       deterministic action snapping."

    The number 26.10 arrives from an EXTERNAL DOCUMENT. No artifact in this repo
    produced it. This probe does not assume it is right or wrong. It asks whether
    beta CAN do the causal work the blueprint assigns to it.

THREE SEPARABLE QUESTIONS
    Q1 HARD SNAP   : Is argmax_k softmax(beta * s)_k independent of beta?
                     If yes, beta does NOT cause the discrete snap.
    Q2 ITERATIVE   : Does beta change retrieval accuracy for the modern-Hopfield
                     energy descent over a real codebook? (weak separation theorem:
                     one-step softmax retrieval is beta-invariant in argmax; the
                     LARGE-beta limit recovers nearest-neighbour.)
    Q3 KERNEL      : Blueprint scores with Re(Psi^dag M_k) -- the REAL PART.
                     This repo's validated similarity kernel is |mean(conj(a)*b)|.
                     A global phase offset on the query rotates Re() toward zero
                     and leaves |.| invariant. A wave system has no absolute phase.

VERDICT RULE (pre-registered, before running)
    If Q1 shows exact invariance across the grid -> the blueprint's
    "hyperparameter lock prevents collapse" is REJECTED AS A MECHANISM.
    beta is then only a soft-readout sharpness / gradient-scale knob.

NOT A GATE ON PROJECT PERFORMANCE. Pure numpy/torch, no checkpoint, no store.
"""
import math
import os
import sys

import numpy as np
import torch

torch.manual_seed(20261003)
G = torch.Generator().manual_seed(20261003)

D = int(os.environ.get("KBD", "8192"))
K = int(os.environ.get("KBK", "256"))
TRIALS = int(os.environ.get("KBT", "200"))
STEPS = int(os.environ.get("KBSTEPS", "12"))
SIGMA = float(os.environ.get("KBSIGMA", "0.60"))     # rad phase noise on query
BETAS = [0.5, 1.0, 2.0, 5.0, 10.0, 26.10, 50.0, 100.0, 1000.0]


def unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def polar(ph):
    return torch.polar(torch.ones_like(ph), ph).to(torch.complex64)


# ---------------------------------------------------------------- codebook
# Phase-coded codebook: every row is a unit complex vector on S^{D-1}.
# This matches the codec family (phase structure), not dense Gaussian.
ph = (torch.rand(K, D, generator=G) * 2 * math.pi)
M = unit(polar(ph))                                  # [K, D] complex64

print(f"codebook: K={K} D={D} trials={TRIALS} sigma={SIGMA} steps={STEPS}")
print(f"codebook bytes: {M.numel() * M.element_size()}  "
      f"({M.numel() * M.element_size() / 2**20:.1f} MiB complex64)")

# ================================================================ Q1: HARD SNAP
s = torch.randn(64, generator=G)
top = int(s.argmax())
inv = all(int(torch.softmax(b * s, dim=0).argmax()) == top for b in BETAS)
print(f"\nQ1 HARD SNAP  argmax(softmax(beta*s)) vs argmax(s)")
print(f"   score vector s = {[round(float(x), 3) for x in s[:4]]} ...")
print(f"   argmax(s) = {top}")
for b in BETAS:
    print(f"      beta={b:<8} argmax={int(torch.softmax(b * s, dim=0).argmax())}")
print(f"   BETA_INVARIANT = {inv}")

# =========================================================== Q2: ITERATIVE
# Query = true pattern k* + phase noise, plus a GLOBAL phase offset (physically
# meaningless in a wave system, and exactly what separates the two kernels).
def corrupt(kstar, sigma, glob):
    q = ph[kstar] + sigma * torch.randn(D, generator=G)
    return polar(q + glob)


def retrieve(q, beta, steps, kernel):
    psi = q.clone()
    for _ in range(steps):
        ip = M @ torch.conj(psi)                     # [K] complex
        sc = ip.real if kernel == "re" else ip.abs()
        z = torch.softmax(beta * sc, dim=0)
        psi = unit((z.to(M.dtype)) @ M)     # z is REAL; M is complex64
    ip = M @ torch.conj(psi)
    sc = ip.real if kernel == "re" else ip.abs() / D
    return int(sc.argmax())


print(f"\nQ2 ITERATIVE RETRIEVAL  (kernel=re, no global offset)")
print(f"   {'beta':>8} {'acc':>8}   {'1-step acc':>11}")
res_q2 = {}
for b in BETAS:
    hit = one = 0
    for t in range(TRIALS):
        kstar = int(torch.randint(K, (1,), generator=G))
        q = corrupt(kstar, SIGMA, 0.0)
        hit += (retrieve(q, b, STEPS, "re") == kstar)
        one += (retrieve(q, b, 1, "re") == kstar)
    res_q2[b] = (hit / TRIALS, one / TRIALS)
    print(f"   {b:>8} {hit/TRIALS:>8.3f}   {one/TRIALS:>11.3f}")

# ================================================================ Q3: KERNEL
print(f"\nQ3 KERNEL  Re(Psi^dag M_k)  vs  |mean(conj(Psi)*M_k)|")
print(f"   Under a GLOBAL phase offset on the query (no physical meaning in a wave).")
print(f"   {'kernel':>6} {'glob=0':>9} {'glob=pi':>9} {'glob=rand':>10}")
res_q3 = {}
for kernel in ("re", "mag"):
    row = []
    for glob_mode in (0.0, math.pi, None):
        hit = 0
        for t in range(TRIALS):
            kstar = int(torch.randint(K, (1,), generator=G))
            g = 0.0 if glob_mode is not None else float(
                torch.rand(1, generator=G).item() * 2 * math.pi)
            if glob_mode is not None:
                g = glob_mode
            q = corrupt(kstar, SIGMA, g)
            hit += (retrieve(q, 26.10, STEPS, kernel) == kstar)
        row.append(hit / TRIALS)
    res_q3[kernel] = row
    print(f"   {kernel:>6} {row[0]:>9.3f} {row[1]:>9.3f} {row[2]:>10.3f}")

# ---------------------------------------------------------------- receipts
t_star_claim = 1.0 / 26.10
print(f"\nRECEIPT")
print(f"   beta_claim            = 26.10")
print(f"   T*_claim printed      = 0.038316")
print(f"   T* = 1/26.10 computed = {t_star_claim:.6f}")
print(f"   match to printed 6dp  = {abs(t_star_claim - 0.038316) < 5e-7}")

q1_reject = inv
q3_gap = res_q3["mag"][2] - res_q3["re"][2]

if q1_reject:
    print("   VERDICT = BETA_IS_NOT_THE_SNAP_MECHANISM")
    print("             argmax is beta-invariant on the whole grid; the discrete")
    print("             snap is caused by argmax, not by beta. beta* cannot")
    print("             'prevent collapse' by this route.")
else:
    print("   VERDICT = BETA_AFFECTS_SNAP (unexpected; re-read)")

print(f"   Q3 median gap (mag - re) under random global phase = {q3_gap:+.3f}")
