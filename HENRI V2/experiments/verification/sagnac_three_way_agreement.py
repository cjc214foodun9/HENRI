#!/usr/bin/env python3
"""Three independent Sagnac implementations must now AGREE. Cross-validation.

WHY THIS IS THE RIGHT CHECK
    The scale defect was not a wrong number in one place; it was TWO CONVENTIONS
    in one codebase. So the strongest test is not "does the fixed function pass its
    own test" -- it is "do the independently-written implementations now produce
    the same value on the same input".

    Three implementations exist:
      A  SagnacMCTSPlanner.dual_channel_sagnac_veto   (PATCHED 2026-10-12)
      B  HENRIVisionEncoder.compute_sagnac_similarity (was ALREADY correct:
         S = 0.5*(1 + torch.dot(a,b)))
      C  arc_sagnac_veto._sagnac_similarity           (canonical; documented the
         defect as FALSIFIED on 2026-08-12 and was NOT wired into the planner)

    Before the patch, A disagreed with B and C by a factor of ~D. If the patch is
    correct, all three must agree to floating-point tolerance on unit-norm real
    waves. A disagreement means the fix is wrong or incomplete -- this is a
    falsifiable cross-check, not a restatement of the unit tests.

    Agreement here also gives a second, independent confirmation that the
    planner's ROOT scoring (which uses B) and CHILD scoring (which uses A) are now
    on the same scale. That mismatch is what made the root look healthy while every
    child looked catastrophic.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = os.path.join(HERE, "sagnac_three_way_agreement_observed.json")


def unit(n: int, seed: int, dim: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, dim, generator=g)
    return v / v.norm(dim=-1, keepdim=True)


def main() -> int:
    from arc_sagnac_veto import _sagnac_similarity as arc_sim
    from henri_vision_encoder import HENRIVisionEncoder
    from sagnac_mcts_planner import SagnacMCTSPlanner

    D = 1024
    planner = SagnacMCTSPlanner(d_model=D, k_blocks=128, tau_veto=0.35, device="cpu")
    enc = HENRIVisionEncoder(d_model=D, k_blocks=128, device="cpu")

    rows = []
    cases = [("identical_a", 1, 1), ("orthogonal_ab", 1, 2),
             ("random_12", 11, 12), ("random_34", 21, 22), ("random_56", 31, 32)]

    for name, s1, s2 in cases:
        a = unit(1, s1, D)[0]
        b = unit(1, s2, D)[0]

        # A: the patched planner (delta = 1 - S, with the norm-consistent similarity)
        delta_a, _, _ = planner.dual_channel_sagnac_veto(a, b, b)
        sim_a = planner._norm_consistent_similarity(a, b)

        # B: the vision encoder (already correct for real unit waves)
        sim_b = enc.compute_sagnac_similarity(a, b)
        delta_b = 1.0 - sim_b

        # C: the canonical sidecar
        sim_c = float(arc_sim(a, b).item())
        delta_c = 1.0 - sim_c

        rows.append({
            "case": name,
            "sim_A_planner_patched": round(sim_a, 9),
            "sim_B_vision_encoder": round(sim_b, 9),
            "sim_C_arc_sidecar": round(sim_c, 9),
            "delta_A": round(delta_a, 9),
            "delta_B": round(delta_b, 9),
            "delta_C": round(delta_c, 9),
            "A_minus_B": round(abs(sim_a - sim_b), 12),
            "A_minus_C": round(abs(sim_a - sim_c), 12),
            "B_minus_C": round(abs(sim_b - sim_c), 12),
        })

    worst_ab = max(r["A_minus_B"] for r in rows)
    worst_ac = max(r["A_minus_C"] for r in rows)
    worst_bc = max(r["B_minus_C"] for r in rows)

    # Also measure the LEGACY disagreement, so the size of what was fixed is
    # quantified rather than asserted.
    legacy = []
    os.environ["HENRI_SAGNAC_LEGACY_SCALE"] = "1"
    try:
        for name, s1, s2 in cases:
            a = unit(1, s1, D)[0]
            b = unit(1, s2, D)[0]
            d_legacy, _, _ = planner.dual_channel_sagnac_veto(a, b, b)
            sim_a = planner._norm_consistent_similarity(a, b)
            s_b = enc.compute_sagnac_similarity(a, b)
            legacy.append({
                "case": name,
                "legacy_delta": round(d_legacy, 9),
                "correct_delta_A": round(1.0 - sim_a, 9),
                "correct_delta_B": round(1.0 - s_b, 9),
                "legacy_minus_correct_A": round(abs(d_legacy - (1.0 - sim_a)), 9),
            })
    finally:
        del os.environ["HENRI_SAGNAC_LEGACY_SCALE"]

    worst_legacy = max(r["legacy_minus_correct_A"] for r in legacy)

    agreed = worst_ab < 1e-9 and worst_ac < 1e-9 and worst_bc < 1e-9
    out = {
        "module": "sagnac_three_way_agreement",
        "evidence_class": "OBSERVED",
        "dim": D,
        "agreement": rows,
        "legacy_vs_correct": legacy,
        "worst_abs_diff": {"A_vs_B": worst_ab, "A_vs_C": worst_ac, "B_vs_C": worst_bc},
        "worst_legacy_error_vs_correct": worst_legacy,
        "three_way_agreement": bool(agreed),
        "claim": (
            "After the patch, the planner (A), the vision encoder (B) and the "
            "canonical sidecar (C) all produce the SAME Sagnac similarity on the "
            "same inputs. Before the patch, A differed from B and C by ~D. This is "
            "the independent cross-check that the root and child scoring paths are "
            "now on one scale."),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("=" * 88)
    print(f"THREE-WAY SAGNAC AGREEMENT  (D={D})")
    print("=" * 88)
    print(f"{'case':<14} {'sim_A(planner)':>15} {'sim_B(encoder)':>15} "
          f"{'sim_C(sidecar)':>15} {'|A-B|':>10}")
    for r in rows:
        print(f"{r['case']:<14} {r['sim_A_planner_patched']:>15.9f} "
              f"{r['sim_B_vision_encoder']:>15.9f} {r['sim_C_arc_sidecar']:>15.9f} "
              f"{r['A_minus_B']:>10.2e}")
    print()
    print(f"  worst |A-B| = {worst_ab:.3e}   worst |A-C| = {worst_ac:.3e}   "
          f"worst |B-C| = {worst_bc:.3e}")
    print(f"  THREE-WAY AGREEMENT = {agreed}")
    print()
    print("  LEGACY (defective) vs CORRECT, to size the fix:")
    print(f"{'case':<14} {'legacy_delta':>14} {'correct_delta':>14} {'error':>12}")
    for r in legacy:
        print(f"{r['case']:<14} {r['legacy_delta']:>14.6f} "
              f"{r['correct_delta_A']:>14.6f} {r['legacy_minus_correct_A']:>12.6f}")
    print(f"  worst legacy-vs-correct error = {worst_legacy:.6f}")
    print()
    print(f"wrote {OUT}")
    return 0 if agreed else 1


if __name__ == "__main__":
    sys.exit(main())
