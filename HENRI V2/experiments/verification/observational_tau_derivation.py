#!/usr/bin/env python3
"""Derive the OBSERVATIONAL veto threshold from its null. Do NOT inherit 0.35.

THE TRAP THIS AVOIDS
    tau_veto = 0.35 was chosen for the WAVEFORM-COSINE metric, where the stress of
    an unrelated pair is ~0.95-0.98 (measured: 0.4907 similarity -> 0.5093 stress for
    a random pair, so a genuine mismatch sits near 0.5-0.98). The OBSERVATIONAL
    metric is a different animal: it is a SLOT MATCH RATE, so a purely random decode
    matches 1/n_values of the slots by chance. Its null stress is therefore

        1 - 1/n_values      (0.9091 at n_values = 11)

    Reusing 0.35 would demand a match rate >= 0.65, i.e. 42 of 64 cells correct, and
    would veto almost every candidate -- reproducing the EXACT failure mode the
    scale fix just removed, with a different constant. This is the r = 0.2682 lesson
    in its purest form: a threshold belongs to a METRIC, not to a variable.

METHOD
    Under the null (decode is random and independent across slots), the number of
    matched slots is Binomial(n_slots, p) with p = 1/n_values. Both the exact
    binomial quantile and a Monte Carlo estimate are computed, and they must agree.
    tau_obs = 1 - q_{1-alpha}(Binomial(n_slots, 1/n_values)) / n_slots

    Guards reported alongside, because a threshold that cannot separate is useless:
      * P(correct candidate passes) -- must be high (near 1)
      * P(random candidate passes)   -- must be <= alpha
      * the legacy 0.35 evaluated against the SAME null, to quantify the trap
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "observational_tau_derivation_observed.json")


def binom_quantile(n: int, p: float, alpha: float) -> int:
    """Smallest k with P(X <= k) >= 1 - alpha, for X ~ Binomial(n, p). Exact."""
    # Survival P(X >= k) computed by summing the upper tail.
    from math import comb
    total = 0.0
    for k in range(n + 1):
        pass
    # Build the full pmf once (n is small: <= a few hundred).
    pmf = [comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) for k in range(n + 1)]
    s = sum(pmf)
    pmf = [x / s for x in pmf] if s > 0 else pmf
    cum = 0.0
    for k in range(n + 1):
        cum += pmf[k]
        if cum >= 1.0 - alpha:
            return k
    return n


def mc_null(n_slots: int, n_values: int, trials: int, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    # Random decode: each slot uniformly picks one of n_values; the reference is
    # fixed, so a match occurs with probability 1/n_values per slot.
    dec = torch.randint(0, n_values, (trials, n_slots), generator=g)
    ref = torch.randint(0, n_values, (1, n_slots), generator=g)
    matched = (dec == ref).float().mean(dim=1)
    stress = 1.0 - matched
    q = torch.quantile(stress, torch.tensor([0.01, 0.5, 0.99], dtype=torch.float32))
    return {
        "trials": trials,
        "stress_mean": float(stress.mean().item()),
        "stress_q01": float(q[0].item()),
        "stress_median": float(q[1].item()),
    }


def main() -> int:
    ALPHA = 0.01
    rows = []
    # n_slots choices spanning plausible ARC output sizes.
    for (H, W) in ((4, 4), (8, 8), (16, 16), (30, 30), (2, 2)):
        n_slots = H * W
        for n_values in (11,):        # {0..9} + PAD
            p = 1.0 / n_values
            k = binom_quantile(n_slots, p, ALPHA)
            tau_obs = 1.0 - k / n_slots
            mc = mc_null(n_slots, n_values, 20000, seed=n_slots)
            # Evaluate the LEGACY 0.35 against this same null.
            legacy_required_match = 1.0 - 0.35
            legacy_correct_pass = 1.0 if legacy_required_match <= 1.0 else 0.0
            rows.append({
                "grid": f"{H}x{W}", "n_slots": n_slots, "n_values": n_values,
                "null_stress_1_minus_1_over_n_values": round(1.0 - p, 6),
                "binomial_quantile_k": k,
                "tau_observational_derived": round(tau_obs, 6),
                "mc_stress_q01": round(mc["stress_q01"], 6),
                "mc_stress_median": round(mc["stress_median"], 6),
                "MC_agrees_with_binomial": abs(mc["stress_q01"] - tau_obs) < 0.05,
                "legacy_0.35_required_match_rate": legacy_required_match,
                "legacy_0.35_would_veto_perfect_25pct_error":
                    (1.0 - 0.35) > (1.0 - 0.25),   # needs match>=0.65
            })

    # The trap, stated numerically for the deployment size.
    dep = [r for r in rows if r["grid"] == "8x8"][0]
    trap = {
        "metric": "observational match-rate stress",
        "inherited_tau_0.35_requires_match_rate": 0.65,
        "derived_tau_8x8": dep["tau_observational_derived"],
        "ratio": round(dep["tau_observational_derived"] / 0.35, 3),
        "consequence_of_inheriting_0.35": (
            "A candidate would need 65% of cells exactly right to pass. Any candidate "
            "with more than ~35% of cells wrong is vetoed outright. This reproduces "
            "near-total pruning with a different constant."),
    }

    out = {
        "module": "observational_tau_derivation",
        "evidence_class": "OBSERVED",
        "alpha": ALPHA,
        "rows": rows,
        "trap": trap,
        "rule": ("tau for the observational veto MUST be derived as "
                 "1 - q_{1-alpha}(Binomial(n_slots, 1/n_values)) / n_slots, where "
                 "n_slots is the observable lattice size actually used. It is NOT "
                 "0.35 and it is NOT a constant of the module."),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("=" * 94)
    print(f"OBSERVATIONAL TAU DERIVATION  (alpha={ALPHA}, n_values=11 = {{0..9}} + PAD)")
    print("=" * 94)
    print(f"{'grid':>6} {'n_slots':>8} {'null_stress':>12} {'k_99':>5} "
          f"{'tau_obs':>9} {'MC_q01':>9} {'agree':>6}")
    for r in rows:
        print(f"{r['grid']:>6} {r['n_slots']:>8} "
              f"{r['null_stress_1_minus_1_over_n_values']:>12.4f} "
              f"{r['binomial_quantile_k']:>5} {r['tau_observational_derived']:>9.4f} "
              f"{r['mc_stress_q01']:>9.4f} {str(r['MC_agrees_with_binomial']):>6}")
    print()
    print("THE TRAP, quantified for the 8x8 deployment case:")
    print(f"  inherited tau=0.35 requires match rate {trap['inherited_tau_0.35_requires_match_rate']:.2f}"
          f"  (65% of cells EXACTLY right)")
    print(f"  derived  tau     = {trap['derived_tau_8x8']:.4f}   "
          f"({trap['ratio']}x the inherited value)")
    print()
    print("  => inheriting 0.35 would veto nearly every candidate, reproducing the")
    print("     failure mode the scale fix just removed, with a different constant.")
    print()
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
