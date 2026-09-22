#!/usr/bin/env python3
"""Settle the binding-norm algebra NUMERICALLY before trusting any capacity number.

WHY
    The first run of this sweep failed its own instrument check:
        | ||R (*) F|| - ||R|| | = 2.988e-03   vs a hard-coded tolerance of 1e-10
    I had baked in the assumption that circular convolution is EXACTLY
    norm-preserving for unit-modulus phasors. It is not. This module decides the
    question by computing the deviation and its SCALING IN D, because the correct
    law is a statement about scaling, not about a constant.

THE CANDIDATE LAWS
    L1  EXACT:        ||a (*) b|| = ||a|| ||b|| / sqrt(D)   for all D
    L2  UP TO 1/sqrtD: ||a (*) b||^2 = 1 + O(sqrt(3/D)) for L2-normalized phasors

    Derivation of L2. For a with |a_n| = 1/sqrt(D), the DFT FA_k has
    E|FA_k|^2 = sum_n |a_n|^2 = 1, so |FA_k|^2 ~ Exp(1). Then by Parseval

        ||a (*) b||^2 = (1/D) sum_k |FA_k|^2 |FB_k|^2 = (1/D) sum_k X_k Y_k

    with X,Y iid Exp(1). Mean = 1. Var = Var(XY)/D = 3/D, so the STANDARD
    DEVIATION of ||a (*) b|| is about sqrt(3/(4D)).

    At D = 65536 that is sqrt(3/262144) = 3.383e-03, and the single sample I
    measured was 2.988e-03. So L2 predicts the observation and L1 is falsified.

    Consequence for capacity: this O(1/sqrt(D)) deviation is NOT the crosstalk.
    It is the intrinsic non-unitarity of convolution on L2-normalized phasors.
    Crosstalk from M-1 interfering members is a DIFFERENT, much larger quantity.

WHAT IS CHECKED
    dev_std(D) * sqrt(D) must be approximately constant (= sqrt(3)/2 = 0.866).
    That is a scaling prediction, so it cannot be satisfied by accident at one D.
"""
from __future__ import annotations

import json
import math
import os
import platform
import sys

import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "binding_norm_algebra_observed.json")


def phasors_l2(count: int, dim: int, gen: torch.Generator) -> torch.Tensor:
    """Unit-modulus phases, L2-normalized so each vector has ||p||_2 = 1."""
    phases = torch.rand(count, dim, generator=gen, dtype=torch.float64) * (2.0 * math.pi)
    p = torch.complex(torch.cos(phases), torch.sin(phases))
    return p / p.norm(dim=-1, keepdim=True)


def bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft(torch.fft.fft(a, dim=-1) * torch.fft.fft(b, dim=-1), dim=-1)


def measure(dim: int, trials: int, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    a = phasors_l2(trials, dim, gen)
    b = phasors_l2(trials, dim, gen)
    out = bind(a, b)
    ratio = (out.norm(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1))).abs()
    dev = ratio - 1.0
    return {
        "dim": dim,
        "trials": trials,
        "mean_ratio": float(ratio.mean().item()),
        "dev_mean": float(dev.mean().item()),
        "dev_std": float(dev.std().item()),
        "predicted_dev_std_sqrt(3/(4D))": math.sqrt(3.0 / (4.0 * dim)),
        "dev_std_times_sqrtD": float(dev.std().item()) * math.sqrt(dim),
    }


def main() -> int:
    dims = [1024, 4096, 16384, 65536]
    trials = 512
    rows = [measure(d, trials, seed=7000 + d) for d in dims]

    print("=" * 90)
    print("BINDING NORM ALGEBRA: does ||a(*)b|| deviate as 1/sqrt(D)?  (float64)")
    print("=" * 90)
    print(f"{'D':>8} {'mean_ratio':>12} {'dev_std':>12} {'pred sqrt(3/4D)':>16} "
          f"{'dev_std*sqrt(D)':>16}")
    for r in rows:
        print(f"{r['dim']:>8} {r['mean_ratio']:>12.8f} {r['dev_std']:>12.3e} "
              f"{r['predicted_dev_std_sqrt(3/(4D))']:>16.3e} "
              f"{r['dev_std_times_sqrtD']:>16.4f}")

    # Scaling test: dev_std * sqrt(D) should be constant, and near sqrt(3)/2.
    scaled = [r["dev_std_times_sqrtD"] for r in rows]
    target = math.sqrt(3.0) / 2.0
    spread = max(scaled) / min(scaled)
    mean_scaled = sum(scaled) / len(scaled)
    constant_ok = spread < 1.5                     # constant across a 64x range in D
    value_ok = abs(mean_scaled - target) / target < 0.35

    # Predictions for the specific dim the project uses
    at65536 = next(r for r in rows if r["dim"] == 65536)
    print()
    print("VERDICT")
    print(f"  L1 EXACT norm preservation        : FALSIFIED "
          f"(dev_std = {at65536['dev_std']:.3e}, not ~0)")
    print(f"  L2 deviation ~ 1/sqrt(D)          : "
          f"{'CONFIRMED' if constant_ok and value_ok else 'NOT CONFIRMED'}")
    print(f"    dev_std*sqrt(D) mean            : {mean_scaled:.4f} "
          f"(theory sqrt(3)/2 = {target:.4f})")
    print(f"    constancy across D=1024..65536  : spread {spread:.3f}x "
          f"(need < 1.5x for a 64x range in D)")
    print(f"  => Circular convolution on L2-normalized phasors is norm-preserving")
    print(f"     only up to O(1/sqrt(D)); it is NOT exactly unitary.")

    out = {
        "module": "binding_norm_algebra",
        "evidence_class": "OBSERVED",
        "platform": platform.platform(),
        "torch": torch.__version__,
        "rows": rows,
        "verdicts": {
            "L1_exact_norm_preservation": "FALSIFIED",
            "L2_inverse_sqrt_D_deviation": "CONFIRMED" if (constant_ok and value_ok) else "NOT_CONFIRMED",
            "dev_std_sqrtD_mean": mean_scaled,
            "theory_sqrt3_over_2": target,
            "constancy_spread": spread,
            "dev_std_at_D65536": at65536["dev_std"],
        },
        "action": (
            "The instrument self-check must validate SCALING IN D, not compare "
            "against a hard-coded 1e-10 tolerance. Any invariant of the form "
            "| ||Psi (*) a|| - 1 | <= 1e-6 cannot hold for L2-normalized phasors "
            "at D=65536; the achievable bound is ~3.4e-3."
        ),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
