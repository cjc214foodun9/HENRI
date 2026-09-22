#!/usr/bin/env python3
"""Separate three hypotheses for why -ln Ptilde does not descend.

The analytic gradient is VERIFIED correct (float64 FD match at 5e-11), so the
fault must be in the training loop's mechanics. Three candidate causes:

  H1 FIXED-TARGET FITTING FAILS  -> the descent direction is wrong somewhere else.
  H2 OVERFITTING / MOVING TARGET -> loss descends on a fixed trajectory but not on
                                    held-out trajectories. The rule is fine; the
                                    experiment's metric (same-noise) is the issue.
  H3 STEP SIZE / SCALE           -> the source's rule SUMS the gradient over the
                                    whole trajectory (Eq. 8-9), so its natural
                                    learning rate is K times smaller than a
                                    per-step mean would need.

MEASUREMENTS (all on the real Langevin computer, no mock):
  A. trajectory sanity: does the noising trajectory carry the imposed pattern?
  B. fixed-trajectory descent: 200 steps on ONE trajectory, loss at start vs end
  C. held-out generalization: fit trajectory A, measure loss on unseen trajectory B
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch

from henri_thermo_langevin import LangevinComputer, LangevinConfig, make_stripe_patterns


def loss_of(model: LangevinComputer, traj: torch.Tensor) -> float:
    return model.reverse_logprob(traj)


def grad_of(model: LangevinComputer, traj: torch.Tensor):
    return model.reverse_logprob_grad(traj)


def main() -> int:
    cfg = LangevinConfig(n_units=48, n_visible=16, t_final=0.5)  # shorter: K=100
    patterns = make_stripe_patterns(16)
    torch.manual_seed(0)
    model = LangevinComputer(cfg, seed=0)

    # ---- A. trajectory sanity -------------------------------------------------
    pat = patterns[0]
    trajA, _ = model.noise_trajectory(pat, generator=torch.Generator().manual_seed(1))
    trajB, _ = model.noise_trajectory(patterns[1], generator=torch.Generator().manual_seed(2))
    vis0 = trajA[0, :16]
    visE = trajA[-1, :16]
    patn = pat / pat.norm().clamp_min(1e-9)
    print("=== A. TRAJECTORY SANITY ===")
    print(f"  |x(t0)|vis = {vis0.norm():.4f}   corr with pattern = "
          f"{float((vis0/vis0.norm().clamp_min(1e-9)) @ patn):+.4f}")
    print(f"  |x(tE)|vis = {visE.norm():.4f}   corr with pattern = "
          f"{float((visE/visE.norm().clamp_min(1e-9)) @ patn):+.4f}")
    print(f"  K steps = {trajA.shape[0]-1}")

    # ---- B. fixed-trajectory descent -----------------------------------------
    print("\n=== B. FIXED-TRAJECTORY DESCENT (200 steps on ONE trajectory) ===")
    for alpha, norm in ((1e-3, False), (1e-2, False), (0.05, True), (0.5, True)):
        torch.manual_seed(0)
        m = LangevinComputer(cfg, seed=0)
        t = trajA.clone()
        l0 = loss_of(m, t)
        for _ in range(200):
            gj, gb = grad_of(m, t)
            if norm:
                k = max(1, t.shape[0] - 1)
                gj, gb = gj / k, gb / k
            with torch.no_grad():
                m.J_raw += alpha * gj
                m.b += alpha * gb
        l1 = loss_of(m, t)
        print(f"  alpha={alpha:<6} normalize={str(norm):<5}  loss {l0:10.4f} -> {l1:10.4f}"
              f"   drop={(l0-l1)/abs(l0)*100:+7.3f}%")

    # ---- C. held-out generalization ------------------------------------------
    print("\n=== C. HELD-OUT GENERALIZATION (fit A -> measure on unseen B) ===")
    torch.manual_seed(0)
    m = LangevinComputer(cfg, seed=0)
    lA_before, lB_before = loss_of(m, trajA), loss_of(m, trajB)
    for _ in range(200):
        gj, gb = grad_of(m, trajA)
        k = max(1, trajA.shape[0] - 1)
        with torch.no_grad():
            m.J_raw += 0.05 * gj / k
            m.b += 0.05 * gb / k
    lA_after, lB_after = loss_of(m, trajA), loss_of(m, trajB)
    print(f"  train A: {lA_before:10.4f} -> {lA_after:10.4f}")
    print(f"  held  B: {lB_before:10.4f} -> {lB_after:10.4f}")
    print(f"  ||J|| after = {m.effective_J().norm():.6f}   ||b|| after = {m.b.norm():.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
