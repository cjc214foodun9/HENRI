"""Correct the E4c receipt: separate the MECHANICAL gate outcome from the
HONEST reading. The mechanical gate fired PASS; the data says the opposite.

Reads the real remote receipt and re-derives the verdict from the arm table.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SRC = Path(r"C:\Users\chan\henri-telemetry\e3\e4c_verdict.json")
OUT = Path(r"C:\Users\chan\henri-telemetry\e3\e4c_verdict_corrected.json")

d = json.loads(SRC.read_text())
arms = d["6_arms"]
A = arms["a_untied_trainable"]
B = arms["b_tied_frozen"]
C = arms["c_tied_trainable"]
D = arms["d_untied_frozen"]
c2 = d["2_C2_token_stream"]
base = d["4_construct"]["C2_strongest_baseline_p1"]
oracle = c2["oracle_p1"]

# --- mechanical gate outcome (what the harness printed) --------------------
mech_best_trainable = max(A["p_at_1"], C["p_at_1"])
mech_gate_c = mech_best_trainable >= base + 0.05
mech_gate_d = C["p_at_1"] >= A["p_at_1"] - 0.02

# --- honest reading --------------------------------------------------------
# Arm b is ZERO-TRAINING and is the strongest arm. It reproduces the backbone's
# own head. Both trainable arms ended BELOW it.
training_hurt = (B["p_at_1"] > A["p_at_1"]) and (B["p_at_1"] > C["p_at_1"])
backbone_dominant = abs(B["p_at_1"] - oracle) < 0.05
c_ascended = C.get("loss_last", 0) > C.get("loss_first", 0)

corrected = {
    "carrier": "E4c",
    "source_receipt": str(SRC),
    "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
    "arms_reported": {
        "a_untied_trainable": {"P@1": A["p_at_1"], "P@5": A["p_at_5"],
                               "CE": A["ce"], "train_loss": [A.get("loss_first"),
                                                             A.get("loss_last")]},
        "b_tied_frozen_ZERO_TRAINING": {"P@1": B["p_at_1"], "P@5": B["p_at_5"],
                                        "CE": B["ce"]},
        "c_tied_trainable": {"P@1": C["p_at_1"], "P@5": C["p_at_5"],
                             "CE": C["ce"], "train_loss": [C.get("loss_first"),
                                                           C.get("loss_last")]},
        "d_untied_frozen": {"P@1": D["p_at_1"], "P@5": D["p_at_5"], "CE": D["ce"]},
    },
    "construct_C2": {"oracle_P@1": oracle, "oracle_P@5": c2["oracle_p5"],
                     "strongest_trivial_baseline_P@1": base,
                     "unigram": c2["unigram_p1"], "lasttok": c2["lasttok_p1"],
                     "trigram": c2["trigram_p1"]},
    "MECHANICAL_gate_outcome": {
        "G-C_movement": bool(mech_gate_c),
        "G-D_attribution_parity": bool(mech_gate_d),
        "harness_printed": d["verdict"],
        "numeral_that_triggered_it": mech_best_trainable,
        "WHY_IT_IS_A_FALSE_POSITIVE": (
            "G-C compared the best TRAINABLE arm (0.321, arm c) against the "
            "trivial baseline (0.184) and read 'above baseline' as learning. "
            "But arm c STARTED at the pretrained optimum (arm c is initialized "
            "from the same teacher table as arm b, whose zero-training score is "
            "0.441) and training DROVE IT DOWN to 0.321 with the training loss "
            "ASCENDING 0.99 -> 2.55. That is degradation of a good init, not "
            "learning. G-D then compared two degraded arms against each other "
            "and reported 'parity'. Both gates are invalidated by arm b."
        ),
    },
    "HONEST_findings": {
        "1_feature_path_verified": bool(backbone_dominant),
        "1_detail": (f"arm b (tied, ZERO training) P@1={B['p_at_1']:.4f} "
                     f"reproduces the backbone's own head "
                     f"oracle P@1={oracle:.4f} -> the frozen hidden-state path "
                     f"and the frozen table are correct"),
        "2_training_hurt": bool(training_hurt),
        "2_detail": (f"arm b (frozen, no training) {B['p_at_1']:.4f} > "
                     f"arm c (trainable, teacher init) {C['p_at_1']:.4f} and > "
                     f"arm a (trainable, random init) {A['p_at_1']:.4f}. "
                     f"Both trainable arms are WORSE than the frozen arm."),
        "3_arm_c_diverged": bool(c_ascended),
        "3_detail": (f"arm c training loss {C.get('loss_first')} -> "
                     f"{C.get('loss_last')} — ASCENDING. lr=3e-4 AdamW + "
                     f"wd=1e-4 for 1000 steps on 32k samples is destructive to a "
                     f"pretrained readout (weight decay shrinks the 18T-token "
                     f"solution; the batch is far too small to re-fit it)."),
        "4_backbone_dominant": bool(backbone_dominant),
        "4_detail": ("The best arm is the backbone's OWN head with ZERO training. "
                     "Therefore the L2 scorer at this layer is entirely the "
                     "pretrained backbone. HENRI contributes nothing measurable "
                     "here. No HENRI capability, benchmark score, or VLA progress "
                     "may be claimed from this carrier."),
        "5_user_directive_status": (
            "FALSIFIED on this scaffold. The directive 'untie the readout — "
            "train a [896->V] head with CE, dropping the frozen-tied constraint' "
            "predicts the untied trainable arm (a) should recover the gap. "
            "Measured: arm a (0.191) is 2.3x WORSE than the frozen tied arm b "
            "(0.441). Untying + training did not recover the gap; it widened it."
        ),
    },
    "CORRECTED_VERDICT": "E4C_FALSIFIED_TRAINING_DEGRADES__BACKBONE_DOMINANT",
    "consequences": {
        "E4d": ("BLOCKED by this result. E4d was specced as 'scale the winning "
                "arm from E4c'. The winning arm is ZERO-TRAINING (the backbone's "
                "own head), so scaling it measures the backbone on more data, "
                "not a HENRI mechanism. Proceeding would produce a backbone "
                "score mislabeled as HENRI progress."),
        "E4e": ("BLOCKED: there is no HENRI-produced scorer worth adapting yet. "
                "Zone C accretion on a scorer that reduces to a frozen backbone "
                "head would be learning nothing."),
        "contract_amendment": ("The §2 amendment itself is NOT falsified — the "
                               "backbone demonstrably supplies token-level "
                               "competence (oracle 0.433 vs trivial 0.184). What "
                               "is falsified is that TRAINING a HENRI readout on "
                               "top of it helps at this scale."),
        "salvage_path_HYPOTHESIS": ("If a trainable HENRI component is wanted at "
                                    "this layer, it must be a residual/adapter "
                                    "that starts at IDENTITY (zero-init delta) so "
                                    "training cannot degrade below the frozen "
                                    "baseline, and it must use lr much lower with "
                                    "weight_decay=0."),
    },
    "labels": ["OBSERVED", "CONDITIONAL_FRESH_SPLIT",
               "CONDITIONAL_CONTAMINATION_UNDISCLOSED"],
    "governance": {
        "sealed_verdict_was": d["verdict"],
        "sealed_verdict_is_invalid": True,
        "reason": "mechanical gate read degradation as movement",
        "corrected_by": "inspection of the arm table (arm b is zero-training and best)",
    },
}
OUT.write_text(json.dumps(corrected, indent=2))
sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
print(json.dumps(corrected["HONEST_findings"], indent=2))
print("\nCORRECTED_VERDICT =", corrected["CORRECTED_VERDICT"])
print("wrote", OUT)
print("sha256", sha)
