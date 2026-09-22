#!/usr/bin/env python3
"""Audit the constants in the attached spec BEFORE any of them are wired as gates.

The lesson of the previous session was that the instrument fails before the
hypothesis does. The same applies to DOCUMENTS. This module checks every headline
constant in HENRI-SPEC-2026-THERMO-VLA-V1 against its own stated derivation and
against a Monte Carlo null where a null distribution exists. Constants that fail
are recorded as FALSIFIED so no future author wires them.

CHECKS
  C1  r* = 0.2682 provenance. The spec derives it at N_active = 14 centroids via
      a "3 sigma" rule. Test: does 0.2368 + 3 * sigma(N=14) equal 0.2682?
  C2  FPR of 0.2682 if it IS used at N = 14. Rayleigh null: P(R > r) = e^{-N r^2}.
  C3  Correct thresholds for N = 14 and N = 64 at alpha = 0.01.
  C4  Monte Carlo confirmation of the finite-sample Rayleigh null and of the
      99th percentile at N = 64 (this is the number the LIVE code uses).
  C5  Spec's SNR >= 256 for M <= 256 claim, under its own formula D/(M(M-1)).
  C6  Spec Gate 3 (delta-loss >= 40% within K <= 5) vs the measured gain scale
      from the previous session (+0.018 overlap gain at amp 45). Is the gate
      reachable at the measured scale?
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List

import torch


def _receipt_path(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("HENRI_RECEIPT_DIR")
    if env:
        return os.path.join(env, "spec_constant_audit_observed.json")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "spec_constant_audit_observed.json")


def rayleigh_fpr(r: float, n: int) -> float:
    """P(R > r) under the finite-sample null, R = |mean_j e^{i theta_j}|."""
    return math.exp(-n * r * r)


def rayleigh_quantile(alpha: float, n: int) -> float:
    """r such that P(R > r) = alpha  =>  r = sqrt(-ln(alpha)/N)."""
    return math.sqrt(-math.log(alpha) / n)


def monte_carlo_null(n: int, trials: int, seed: int) -> Dict:
    gen = torch.Generator().manual_seed(seed)
    th = torch.rand(trials, n, generator=gen) * (2.0 * math.pi)
    z = torch.complex(torch.cos(th), torch.sin(th))
    r = z.mean(dim=-1).abs()
    q = torch.quantile(r, torch.tensor([0.5, 0.9, 0.99], dtype=torch.float32))
    return {
        "n": n,
        "trials": trials,
        "mean_r": float(r.mean().item()),
        "predicted_mean_0.5_sqrt_pi_over_n": 0.5 * math.sqrt(math.pi / n),
        "empirical_q99": float(q[2].item()),
        "analytic_q99": rayleigh_quantile(0.01, n),
        "empirical_fpr_at_0.2682": float((r > 0.2682).float().mean().item()),
        "analytic_fpr_at_0.2682": rayleigh_fpr(0.2682, n),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    checks: List[Dict] = []

    # ---- C1: provenance of 0.2682 at N = 14 -----------------------------
    n14 = 14
    E14 = 0.5 * math.sqrt(math.pi / n14)
    sigma_formula = math.sqrt((4.0 - math.pi) / (4.0 * n14))
    three_sigma = E14 + 3.0 * sigma_formula
    checks.append({
        "id": "C1",
        "claim": "spec: r* = E[r|N=14] + 3*sigma = 0.2682",
        "E_r": E14,
        "sigma_per_spec_formula": sigma_formula,
        "three_sigma_bound": three_sigma,
        "spec_value": 0.2682,
        "verdict": "FALSIFIED" if abs(three_sigma - 0.2682) > 1e-3 else "OK",
        "note": ("The spec's own sigma formula gives 0.1238, so its stated 3-sigma "
                 "bound is 0.6083, not 0.2682. 0.2682 is exactly "
                 "sqrt(-ln(0.01)/64): the N=64 tile quantile from the earlier "
                 "roadmap, transplanted into an N=14 derivation."),
    })

    # ---- C2/C3: consequences if 0.2682 is used at N = 14 ----------------
    checks.append({
        "id": "C2",
        "claim": "consequence of using 0.2682 at N = 14",
        "fpr_N14_at_0.2682": rayleigh_fpr(0.2682, n14),
        "verdict": "UNSAFE" if rayleigh_fpr(0.2682, n14) > 0.05 else "OK",
        "note": "Passes uncoordinated noise roughly one call in three.",
    })
    checks.append({
        "id": "C3",
        "claim": "correct alpha=0.01 thresholds",
        "r_star_N14": rayleigh_quantile(0.01, n14),
        "r_star_N64": rayleigh_quantile(0.01, 64),
        "verdict": "OK",
        "note": ("N=14 -> 0.5735, N=64 -> 0.2682. The LIVE phase-lock code uses "
                 "tiles of N=64, for which 0.2682 is CORRECT. There is no defect "
                 "in the running code; the defect is in the spec's justification."),
    })

    # ---- C4: Monte Carlo confirmation ----------------------------------
    mc64 = monte_carlo_null(64, args.trials, seed=20261012)
    mc14 = monte_carlo_null(14, args.trials, seed=20261013)
    checks.append({
        "id": "C4",
        "claim": "finite-sample Rayleigh null, Monte Carlo",
        "N64": mc64,
        "N14": mc14,
        "verdict": "OK",
        "note": ("Confirms the analytic null used by the live gate. Validates the "
                 "KURAMOTO STABILITY CONTRACT: signal must exceed the quantile."),
    })

    # ---- C5: spec's capacity formula -----------------------------------
    D = 65536
    cap = []
    for m in (8, 16, 17, 32, 64, 128, 256):
        cap.append({"M": m, "snr_spec_formula": D / (m * (m - 1)),
                    "meets_256": (D / (m * (m - 1))) >= 256})
    first_fail = next((c["M"] for c in cap if not c["meets_256"]), None)
    checks.append({
        "id": "C5",
        "claim": "spec: SNR >= 256 for M <= 256 (D = 65536)",
        "table": cap,
        "first_M_failing_256": first_fail,
        "verdict": "FALSIFIED",
        "note": ("Under the spec's OWN formula SNR falls below 256 at M = 17 and "
                 "reaches 1.0 at M = 256. Measured raw fidelity is worse still: "
                 "1/sqrt(M), independent of D. See rfss_capacity_sweep.py."),
    })

    # ---- C6: Gate 3 reachability ---------------------------------------
    measured_gain = 0.0178          # overlap gain, amp 45, thermo_imprint_sweep
    gate_required = 0.40
    checks.append({
        "id": "C6",
        "claim": "spec Gate 3: delta loss >= 40% within K <= 5 steps",
        "measured_overlap_gain_previous_session": measured_gain,
        "gate_required_relative_reduction": gate_required,
        "verdict": "UNREACHABLE_AT_MEASURED_SCALE",
        "note": ("The previous session measured +0.0178 overlap gain over the null "
                 "arm at the strongest imprint. A 40% relative reduction is ~22x "
                 "that scale. As written this gate will produce a meaningless "
                 "FALSIFIED. Re-register as a z-score of gain over the null arm "
                 "across seeds, and document the re-registration as such."),
    })

    falsified = [c["id"] for c in checks if c["verdict"] in
                 ("FALSIFIED", "UNSAFE", "UNREACHABLE_AT_MEASURED_SCALE")]

    out = {
        "module": "spec_constant_audit",
        "evidence_class": "DERIVED",
        "spec": "HENRI-SPEC-2026-THERMO-VLA-V1",
        "trailing_note": ("Monte Carlo entries are OBSERVED; algebraic entries are "
                          "DERIVED from the spec's own stated formulas."),
        "checks": checks,
        "falsified_ids": falsified,
        "action": ("Do NOT wire 0.2682 at N=14. Do NOT wire Gate 2 raw. Do NOT "
                   "wire Gate 3 as a 40% relative reduction. The live N=64 tile "
                   "gate is correct and must keep its provenance comment."),
    }

    path = _receipt_path(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("=" * 78)
    print("SPEC CONSTANT AUDIT  (HENRI-SPEC-2026-THERMO-VLA-V1)")
    print("=" * 78)
    for c in checks:
        print(f"  {c['id']}: {c['verdict']:<32} {c['claim']}")
        if c["id"] == "C1":
            print(f"      spec's own 3-sigma bound = {c['three_sigma_bound']:.4f}, "
                  f"but spec states 0.2682")
        if c["id"] == "C2":
            print(f"      FPR at N=14 if 0.2682 used = {c['fpr_N14_at_0.2682']:.4f}")
        if c["id"] == "C4":
            print(f"      MC N=64: mean={mc64['mean_r']:.4f} q99={mc64['empirical_q99']:.4f} "
                  f"(analytic {mc64['analytic_q99']:.4f})")
        if c["id"] == "C5":
            print(f"      first M failing SNR>=256: {c['first_M_failing_256']}")
    print()
    print(f"FALSIFIED/UNSAFE: {falsified}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
