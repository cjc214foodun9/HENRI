#!/usr/bin/env python3
"""Derive tau for the MEASURED ARC-AGI-3 observable lattice (64x64, 11 values).

WHY THIS IS NEEDED NOW
    arc_runtime_smoke.py PASSED and reported the observable spec by measurement:
        R4 grid shapes: [(64, 64)]   value ranges: [(0, 10)]
        frames decoded: 60  non-empty: 60  live(differs): True
    So the real deployment lattice is n_slots = 4096 with n_values = 11. The tau
    figures derived earlier were for 16/64/256/900 slots, none of which is the
    deployment size. Inheriting any of them would be the same category error as
    inheriting 0.35 from the waveform metric.

    At n_slots = 4096 the match-rate null is far tighter than at 64 slots: a random
    decode matches a fraction ~1/11 of cells, and the 99th percentile of the BINOMIAL
    count is ~1.5% above the mean, so tau lands very close to 1 - 1/11 = 0.909.
    That is the number the veto must use, and it is NOT 0.35 and NOT 0.8125.

WHAT IT REPORTS
    T1  exact binomial tau at 4096 slots, several alphas
    T2  Monte Carlo confirmation at 4096 slots
    T3  the operating point in MATCH-RATE terms (how many cells may be wrong)
    T4  a sensitivity note: at this lattice size the veto is a STRICT gate -- almost
        any candidate that is not near-correct passes. That is a property of the
        metric, and it must be stated rather than discovered later.
"""
from __future__ import annotations

import json
import os
import sys
from math import comb
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from henri_grid_observable import derive_tau  # noqa: E402

OUT = Path(__file__).resolve().parent / "arc_lattice_tau_observed.json"
N_SLOTS = 64 * 64          # measured from the live runtime
N_VALUES = 11              # measured: values 0..10
TRIALS = 200_000


def mc_q99(n_slots: int, n_values: int, trials: int, seed: int = 0) -> float:
    g = torch.Generator().manual_seed(seed)
    dec = torch.randint(0, n_values, (trials, n_slots), generator=g)
    ref = torch.randint(0, n_values, (1, n_slots), generator=g)
    stress = 1.0 - (dec == ref).float().mean(dim=1)
    return float(stress.quantile(0.01).item())


def main() -> int:
    rows = []
    for alpha in (0.05, 0.01, 0.001):
        ti = derive_tau(N_SLOTS, N_VALUES, alpha)
        rows.append({
            "alpha": alpha,
            "q_matched": ti["q_1_minus_alpha_matched"],
            "tau": ti["tau_observational"],
            "min_match_rate_to_pass": 1.0 - ti["tau_observational"],
            "max_cells_wrong_to_pass": N_SLOTS - ti["q_1_minus_alpha_matched"],
        })

    mc = mc_q99(N_SLOTS, N_VALUES, TRIALS)
    t01 = derive_tau(N_SLOTS, N_VALUES, 0.01)["tau_observational"]
    agrees = abs(mc - t01) <= 0.01

    # Sensitivity: how many cells may be wrong at each tau?
    print("=" * 88)
    print(f"TAU FOR THE MEASURED ARC-AGI-3 LATTICE  ({N_SLOTS} slots, {N_VALUES} values)")
    print("=" * 88)
    print(f"  null match rate = 1/{N_VALUES} = {1/N_VALUES:.4f}"
          f"  ->  null stress = {1-1/N_VALUES:.4f}")
    print()
    print(f"  {'alpha':>7} {'q_matched':>10} {'tau':>9} {'min_match':>10} "
          f"{'cells_may_be_wrong':>19}")
    for r in rows:
        print(f"  {r['alpha']:>7} {r['q_matched']:>10} {r['tau']:>9.4f} "
              f"{r['min_match_rate_to_pass']:>10.4f} "
              f"{r['max_cells_wrong_to_pass']:>19}")
    print()
    print(f"  Monte Carlo q01 stress ({TRIALS:,} draws) = {mc:.4f}")
    print(f"  exact binomial tau(alpha=0.01)            = {t01:.4f}")
    print(f"  MC agrees with exact                      = {agrees}")
    print()
    print("  SENSITIVITY, stated up front:")
    print(f"    at alpha=0.01 a candidate may get {rows[1]['max_cells_wrong_to_pass']} "
          f"of {N_SLOTS} cells wrong and still pass.")
    print(f"    The veto is therefore a STRICT gate: it rejects only candidates that are")
    print(f"    no better than a random program. It does NOT discriminate among")
    print(f"    partially-correct candidates -- that is the waveform cosine's job.")
    print()
    print("  CONTRAST (the trap this avoids):")
    print(f"    tau=0.35 would demand a match rate of 0.65 = "
          f"{int(0.65*N_SLOTS)} correct of {N_SLOTS} cells.")
    print(f"    tau={t01:.4f} demands {rows[1]['q_matched']} correct. "
          f"Ratio of required correct cells: {(0.65*N_SLOTS)/rows[1]['q_matched']:.1f}x")

    out = {
        "module": "arc_lattice_tau", "evidence_class": "OBSERVED",
        "source_spec": "measured by arc_runtime_smoke.py (64x64, values 0..10)",
        "n_slots": N_SLOTS, "n_values": N_VALUES, "trials": TRIALS,
        "rows": rows, "mc_q01_stress": mc,
        "mc_agrees_with_exact_at_alpha0.01": agrees,
        "tau_deployment_alpha0.01": t01,
        "inherited_0.35_would_require_match_rate": 0.65,
        "note": ("The observational veto is a STRICT gate at this lattice size: it "
                 "rejects only candidates no better than random. Ranking among "
                 "partially-correct candidates is the waveform cosine's role."),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
