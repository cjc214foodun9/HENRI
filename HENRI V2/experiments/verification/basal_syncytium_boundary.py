"""G4 boundary characterization: locate and characterize the locking transition.

The coarse sweep showed a SHARP, non-monotonic response:

    decay   8 -> r 0.072      decay  96 -> r 0.020      decay 192 -> r 1.0000
    decay  16 -> r 0.128      decay 128 -> r 0.0018     decay 256 -> r 1.0000
    decay  32 -> r 0.360

A jump from 0.0018 to 1.0000 between decay 128 and 192 is not a smooth
crossover. Three explanations must be separated:

  H1  A genuine FIRST-ORDER transition. In the 1-D Kuramoto lattice the
      transition is known to be first order (nucleation of a coherent domain),
      so a jump is the expected signature, not a numerical fault.
  H2  Hysteresis. If H1 holds, the ordered state must be self-sustaining from
      an ordered initial condition even BELOW the random-start threshold.
  H3  Numerical artifact (a kernel that is effectively constant/global).

H3 is killed directly by measuring the kernel's effective support. H1 vs H2 are
separated by starting from an ordered initial condition and checking whether the
state persists.

The engineering consequence: a working point chosen at the random-start
threshold is a knife edge. A working point must be chosen with margin.
"""
from __future__ import annotations

import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from basal_boundary_engine import (
    SPEC_R_GATE,
    EvanescentKuramotoSyncytium,
    evanescent_kernel,
)

N = 8192


def kernel_support(lam: float) -> dict:
    """H3 kill: the kernel must NOT be effectively global."""
    k = evanescent_kernel(N, lam)
    idx = torch.arange(N, dtype=torch.float32)
    d = torch.minimum(idx, N - idx)
    return {
        "decay": lam,
        "fraction_of_ring": lam / N,
        "mass_within_1lam": float(k[d <= lam].sum()),
        "mass_within_2lam": float(k[d <= 2 * lam].sum()),
        "mass_within_4lam": float(k[d <= 4 * lam].sum()),
        "max_over_min_over_ring": float(k.max() / max(k.min(), 1e-30)),
        "effective_participation": float(1.0 / (k * k).sum()),  # ~2*lam
    }


def run(lam, K, steps, dt=0.01, seed=0, ordered_init=False, chunk=500):
    syn = EvanescentKuramotoSyncytium(
        num_channels=N, coupling_K=K, decay_length=lam, dt=dt,
        natural_frequency_scale=0.0, noise_temperature=0.0, seed=seed,
    )
    if ordered_init:
        syn.phases = torch.zeros(N, dtype=torch.float32, device=syn.device)
    else:
        syn.random_phases(seed=seed)
    traj = []
    for done in range(0, steps, chunk):
        out = syn.relax(min(chunk, steps - done))
        traj.append({"t": round((done + min(chunk, steps - done)) * dt, 2),
                     "r": round(float(out["r"]), 5)})
    return traj


def main():
    rep = {"num_channels": N, "r_gate": SPEC_R_GATE, "dt": 0.01}

    print("=== H3 kill: kernel effective support (an exponential, not a constant) ===")
    reps = {lam: kernel_support(lam) for lam in (128, 192, 256, 384)}
    rep["H3_kernel_support"] = {str(k): v for k, v in reps.items()}
    for lam, v in reps.items():
        print(f"  decay={lam:4d} frac={v['fraction_of_ring']:.5f} "
              f"mass(1L)={v['mass_within_1lam']:.4f} mass(2L)={v['mass_within_2lam']:.4f} "
              f"participation~{v['effective_participation']:.1f}")

    print("\n=== Fine sweep: random start, K=2.45, 4000 steps ===")
    fine = []
    for lam in (128, 144, 152, 160, 168, 176, 184, 192, 224, 256):
        tr = run(float(lam), 2.45, 4000)
        fine.append({"decay": lam, "fraction": lam / N, "r_final": tr[-1]["r"],
                     "r_peak": max(d["r"] for d in tr), "traj": tr})
        print(f"  decay={lam:4d} ({lam/N:.5f})  r_final={tr[-1]['r']:.4f}  "
              f"r_peak={fine[-1]['r_peak']:.4f}  clears={tr[-1]['r'] >= SPEC_R_GATE}")
    rep["fine_sweep_random_start"] = fine

    print("\n=== H2 hysteresis: ORDERED start (all phases 0), K=2.45, 4000 steps ===")
    hyst = []
    for lam in (32, 64, 96, 128, 160, 192):
        tr = run(float(lam), 2.45, 4000, ordered_init=True)
        hyst.append({"decay": lam, "fraction": lam / N, "r_final": tr[-1]["r"]})
        print(f"  decay={lam:4d}  r_final={tr[-1]['r']:.4f}  "
              f"persists={tr[-1]['r'] >= SPEC_R_GATE}")
    rep["hysteresis_ordered_start"] = hyst

    print("\n=== K-dependence at decay=256, random start ===")
    kdep = []
    for K in (0.5, 1.0, 2.45, 4.0, 8.0):
        tr = run(256.0, K, 4000)
        kdep.append({"K": K, "r_final": tr[-1]["r"]})
        print(f"  K={K:5.2f}  r_final={tr[-1]['r']:.4f}  "
              f"clears={tr[-1]['r'] >= SPEC_R_GATE}")
    rep["K_dependence_decay256"] = kdep

    out = os.path.join(_HERE, "basal_syncytium_boundary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    print(f"\nreceipt: {out}")


if __name__ == "__main__":
    main()
