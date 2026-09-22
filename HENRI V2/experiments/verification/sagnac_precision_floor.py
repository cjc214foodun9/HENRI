#!/usr/bin/env python3
"""Is 2.98e-08 agreement, or is it a real disagreement? Settle it from precision.

THE QUESTION
    The three-way check measured worst |A-B| = 2.98e-08 between
      A  SagnacMCTSPlanner._norm_consistent_similarity  (patched)
      B  HENRIVisionEncoder.compute_sagnac_similarity   (pre-existing, correct)
      C  arc_sagnac_veto._sagnac_similarity             (canonical)
    and my gate demanded < 1e-9. The gate FAILED. Either the fix is subtly wrong,
    or my tolerance was calibrated for float64 while the comparison runs in float32.

    This is exactly the "instrument before hypothesis" question, so it is settled by
    measuring the precision floor of the actual computation rather than by relaxing
    the gate to whatever value passes.

METHOD
    B computes `0.5 * (1.0 + torch.dot(w1, w2))` in FLOAT32 and then np.clip.
    A computes `sum(a*b)/ (||a|| ||b||)` after `.item()` casts.
    So the attainable agreement is bounded by FLOAT32 rounding of the dot product,
    not by 1e-9. Measure that floor directly:
      * float32 dot of a unit vector with ITSELF (should be 1, will not be exactly)
      * the resulting 0.5*(1+dot) deviation
    If the observed cross-implementation difference is bounded by that measured
    floor, the implementations agree and the GATE was wrong.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "sagnac_precision_floor_observed.json")


def main() -> int:
    D = 1024
    floats = {}

    floats["float32_eps"] = float(torch.finfo(torch.float32).eps)
    floats["float64_eps"] = float(torch.finfo(torch.float64).eps)

    # --- measured float32 floor for the ENCODER's formula on identical input ---
    g = torch.Generator().manual_seed(1)
    v = torch.randn(D, generator=g)
    v = v / v.norm()

    devs_float32 = []
    for seed in range(1, 33):
        gg = torch.Generator().manual_seed(seed)
        w = torch.randn(D, generator=gg)
        w = w / w.norm()                      # float32 unit vector
        dot32 = torch.dot(w, w).item()        # == 1 in exact arithmetic
        s32 = 0.5 * (1.0 + dot32)
        devs_float32.append(abs(1.0 - s32))

    # Same computation promoted to float64, where the deviation should collapse.
    devs_float64 = []
    for seed in range(1, 33):
        gg = torch.Generator().manual_seed(seed)
        w = torch.randn(D, generator=gg).double()
        w = w / w.norm()
        dot64 = torch.dot(w, w).item()
        s64 = 0.5 * (1.0 + dot64)
        devs_float64.append(abs(1.0 - s64))

    floats["identical_input_float32_selfsim_max_dev"] = max(devs_float32)
    floats["identical_input_float32_selfsim_min_dev"] = min(devs_float32)
    floats["identical_input_float64_selfsim_max_dev"] = max(devs_float64)

    # --- the observed cross-implementation difference being adjudicated ---
    floats["observed_worst_abs_diff_A_vs_B"] = 2.98e-08
    floats["observed_worst_abs_diff_A_vs_C"] = 2.049e-08
    floats["legacy_error_vs_correct"] = 0.999023

    # THE ADJUDICATION. float32 eps = 1.19e-07. A difference of 2.98e-08 is
    # 0.25 * eps, i.e. INSIDE one float32 rounding step, and it is the same order
    # as the measured self-similarity deviation of the encoder's own formula on
    # IDENTICAL input. Therefore the three implementations agree, and the 1e-9 gate
    # was a float64 tolerance applied to a float32 pipeline.
    floor32 = floats["identical_input_float32_selfsim_max_dev"]
    tol = max(4.0 * floats["float32_eps"], 2.0 * floor32)
    floats["derived_tolerance"] = tol
    floats["agreement_within_float32"] = (
        floats["observed_worst_abs_diff_A_vs_B"] <= tol
        and floats["observed_worst_abs_diff_A_vs_C"] <= tol)
    # Sanity: the legacy error must be FAR outside this tolerance, or the tolerance
    # would be so loose as to hide the defect.
    floats["legacy_error_detected_at_this_tolerance"] = (
        floats["legacy_error_vs_correct"] > 100.0 * tol)

    ok = (floats["agreement_within_float32"]
          and floats["legacy_error_detected_at_this_tolerance"]
          and floats["identical_input_float64_selfsim_max_dev"] < 1e-15)

    out = {"module": "sagnac_precision_floor", "evidence_class": "OBSERVED",
           "dim": D, "measured": floats,
           "conclusion": (
               "The three implementations agree to within float32 machine precision "
               "(2.98e-08 <= 4*float32_eps = 4.77e-07, and equal in order to the "
               "measured float32 self-similarity deviation). The 1e-9 gate was a "
               "float64 tolerance applied to a float32 pipeline: an INSTRUMENT error, "
               "not an implementation error. The legacy defect (0.999) is 3+ orders "
               "of magnitude outside the corrected tolerance, so the tolerance does "
               "not hide the defect it exists to catch."),
           "gate_corrected_to": tol,
           "verdict": "AGREE_WITHIN_FLOAT32" if ok else "DISAGREE"}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("=" * 78)
    print("SAGNAC PRECISION FLOOR: is 2.98e-08 agreement or disagreement?")
    print("=" * 78)
    print(f"  float32 eps                                  = {floats['float32_eps']:.3e}")
    print(f"  float64 eps                                  = {floats['float64_eps']:.3e}")
    print(f"  encoder formula on IDENTICAL input, float32  = "
          f"{floats['identical_input_float32_selfsim_max_dev']:.3e} max dev")
    print(f"  same, float64                                = "
          f"{floats['identical_input_float64_selfsim_max_dev']:.3e} max dev")
    print()
    print(f"  observed |A-B|                               = "
          f"{floats['observed_worst_abs_diff_A_vs_B']:.3e}")
    print(f"  observed |A-C|                               = "
          f"{floats['observed_worst_abs_diff_A_vs_C']:.3e}")
    print(f"  derived tolerance (4*eps32, 2*measured)      = {tol:.3e}")
    print()
    print(f"  AGREE WITHIN FLOAT32                         = "
          f"{floats['agreement_within_float32']}")
    print(f"  legacy error 0.999 still detected            = "
          f"{floats['legacy_error_detected_at_this_tolerance']}  "
          f"(0.999 > 100x tol = {100*tol:.3e})")
    print()
    print(f"VERDICT: {out['verdict']}")
    print(f"wrote {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
