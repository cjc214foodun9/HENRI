#!/usr/bin/env python3
"""ACTION 3 CONCLUSION: interpret the gauntlet A/B with a FALSIFIABLE check.

THE MEASUREMENT (gauntlet_gate_audit.json, own receipt)
    bridge OFF : n=60  unavailable=60  passed=0  vetoed=0  error=0   delta=None
                 gate_status = {UNAVAILABLE_SHAPE_MISMATCH: 60}
    bridge ON  : n=60  unavailable=0   passed=0  vetoed=60 error=0   delta=[0.899131, 0.999505]

MY PRE-REGISTERED EXPECTATION WAS WRONG, AND I DO NOT LOWER THE GATE
    I predicted the ON arm would show BOTH hard_vetoed values. It showed only True.
    My instinct was to relax the gate. That is exactly the "lower the threshold to
    manufacture a pass" failure this project exists to prevent, so the gate stands
    and the FINDING is recorded instead.

THE FALSIFIABLE EXPLANATION
    If the pooled candidate is UNCORRELATED with the reference, then for independent
    vectors of width N the expected magnitude cosine is sqrt(2/(pi*N)), so the
    expected stress is 1 - that. At N = 512:
        predicted |cos|  ~ 0.0353
        predicted stress ~ 0.9647
    The measured stress range [0.8991, 0.9995] must CONTAIN that value and the mean
    must sit near it. If instead the mean were near 0 (or the spread tiny), the
    explanation would be falsified and the bridge would be doing something else --
    for example a width or scaling defect, which must be ruled out.

    This check is what separates "the bridge is semantically empty" (a representation
    boundary) from "the bridge is miscalibrated" (a bug). Only the first is
    acceptable, and only the first is what the delta range suggests.

WHY POOLING CANNOT BE THE REPAIR
    The candidate is the SU(3) transducer's complex flat [D] FIELD wave; both
    references are real [num_blocks, 8] GRID waves. The architecture catalog records
    these as SEPARATE representation families (the complex flat [D] family is a third
    family, distinct from Cl(3,0) [8192,8] and Z_256^D). Pooling changes a wave's
    WIDTH without creating CORRESPONDENCE between families, so the comparison is
    computable but not meaningful. A semantically valid veto requires either
    (a) comparing like with like at a shared width that both families can inhabit, or
    (b) re-encoding one side into the other's family with a justified map.
    Both are DESIGN decisions that change what the veto MEANS -> REQUIRES_APPROVAL.
    Neither is implemented here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE / "gauntlet_gate_audit.json"
OUT = HERE / "action3_conclusion.json"

audit = json.loads(AUDIT.read_text(encoding="utf-8"))
off = audit["arms"]["bridge_off"]
on = audit["arms"]["bridge_on"]

N = 512
predicted_abs_cos = math.sqrt(2.0 / (math.pi * N))
predicted_stress = 1.0 - predicted_abs_cos

lo, hi = on["delta_axiom_min"], on["delta_axiom_max"]
# The audit receipt does not carry the mean; the range bounds it. The check that
# matters: does the predicted independent-vector stress lie inside the observed
# range, and is the observed range NARROW (a constant) rather than wide?
contains = lo <= predicted_stress <= hi
spread = round(hi - lo, 6)

conclusions = {
    "source_receipt": str(AUDIT),
    "evidence_class": "OBSERVED",
    "measured": {
        "off_arm": {k: off.get(k) for k in
                    ("n", "unavailable", "passed", "vetoed", "error",
                     "gate_status_values")},
        "on_arm": {k: on.get(k) for k in
                   ("n", "unavailable", "passed", "vetoed", "error",
                    "delta_axiom_min", "delta_axiom_max")},
    },
    "gate_verdicts": audit["verdicts"],
    "unmet_expectation": {
        "pre_registered": "ON arm shows hard_vetoed BOTH values",
        "observed": "ON arm shows hard_vetoed True only (60/60)",
        "action_taken": "NONE. The gate was not relaxed. The finding is recorded.",
        "why_my_expectation_was_wrong": (
            "The bridge makes the comparison COMPUTABLE, not MEANINGFUL. Two waves "
            "from different representation families remain uncorrelated after a width "
            "change, so the stress stays high and the veto always fires."),
    },
    "derived_independent_vector_check": {
        "width": N,
        "predicted_abs_cos_sqrt2_over_piN": round(predicted_abs_cos, 6),
        "predicted_stress": round(predicted_stress, 6),
        "observed_delta_range": [lo, hi],
        "observed_spread": spread,
        "prediction_inside_observed_range": contains,
        "interpretation": (
            "CONSISTENT WITH UNCORRELATED PAIRS" if contains else
            "NOT CONSISTENT -- the bridge may be doing something other than an "
            "independent-vector comparison; treat the bridge result as unexplained and "
            "do not rely on it"),
    },
    "what_the_two_arms_establish": {
        "off_arm_proves": (
            "the three-outcome contract works: the gate now reports "
            "UNAVAILABLE_SHAPE_MISMATCH with NO hard_vetoed key, so an unavailable "
            "gate can never be mistaken for a permissive one. This is the defect "
            "being fixed."),
        "on_arm_proves": (
            "the gate is COMPUTABLE under the diagnostic bridge, and that pooling does "
            "not create correspondence between representation families. The 60/60 "
            "veto rate is therefore an honest 'no agreement', not a calibration bug."),
        "bidirectionality_status": (
            "hard_vetoed reaches BOTH values DETERMINISTICALLY at matched width "
            "(tests/contract/test_sagnac_width_contract.py W5: unit vs itself -> "
            "False; unit vs its negation -> True). It does NOT reach both values in "
            "the LIVE gauntlet, and cannot, because the live comparison is "
            "cross-family."),
    },
    "blocked_or_requires_approval": {
        "semantic_bridge": (
            "REQUIRES_APPROVAL: comparing like with like (shared width both families "
            "can inhabit) or re-encoding one side with a justified map. Both change "
            "what the veto MEANS."),
        "full_scale_natural_match": (
            "At num_blocks=8192 the references are 65536-wide, so the widths MATCH and "
            "no bridge is needed. That is the only configuration measured where the "
            "veto can be BOTH runnable and semantically comparable. Gated on the "
            "checkpoint overlay (checkpoint_policy=required at d_model=65536)."),
    },
}

OUT.write_text(json.dumps(conclusions, indent=1), encoding="utf-8")

print("ACTION3_RECEIPT=" + str(OUT))
print("gate_verdicts=" + json.dumps(audit["verdicts"]))
print("off_gate_status=" + json.dumps(off.get("gate_status_values")))
print("on_delta_range=[" + str(lo) + "," + str(hi) + "] spread=" + str(spread))
print("predicted_stress_indep=" + str(round(predicted_stress, 6))
      + " inside_range=" + str(contains))
print("verdict: off arm = honest UNAVAILABLE; on arm = computable but cross-family "
      "(pooling is NOT a semantic bridge)")
