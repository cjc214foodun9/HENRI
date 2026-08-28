"""Executable emergence-gate verifier for the Universal VLA Pathway doc §4.

Doc: Project_HENRI_Universal_VLA___Nested_Recursive_Pathway.md (2026-08-28,
sha256 eb92bcb8a3a7c5cec1d9b078ac14fd2f3ddeb93fc52cd72a29b477da341cfc7c).

Fail-closed: a gate whose required fields are absent is NOT_MEASURABLE,
never PASS. Gate 6 follows the doc formula literally, which is mutually
exclusive (dnu>0 and T<=0.02) AND (dnu<=0 and T>=0.50); literal execution
therefore always yields FAIL with an explanatory reason — the doc formula
is unsatisfiable, which is a finding, not a pass.
"""
import math

GATES = (
    "gate_1_procrustes_grounded",
    "gate_2_vacuum_orthogonality",
    "gate_3_light_cone_adaptive",
    "gate_4_sagnac_branch_pruning",
    "gate_5_egress_mutual_info",
    "gate_6_valence_thermodynamic_coupling",
)

_DOC = "HENRI-ARCH-2026-08-UNIVERSAL-VLA-PATHWAY"


def _nm(required):
    return {"status": "NOT_MEASURABLE", "reason": f"missing fields: {sorted(required)}", "doc": _DOC}


def compute_emergence_gates(telemetry):
    """telemetry: dict with any of the gate fields. Returns dict[gate -> dict]."""
    out = {}

    r1 = {"goal_wave_norm", "task_functor_error"}
    if not r1.issubset(telemetry):
        out["gate_1_procrustes_grounded"] = _nm(r1)
    else:
        ok = (telemetry["goal_wave_norm"] == 1.0) and (telemetry["task_functor_error"] < 0.05)
        out["gate_1_procrustes_grounded"] = {
            "status": "PASS" if ok else "FAIL",
            "goal_wave_norm": telemetry["goal_wave_norm"],
            "task_functor_error": telemetry["task_functor_error"],
            "doc": _DOC,
        }

    r2 = {"vacuum_correlation"}
    if not r2.issubset(telemetry):
        out["gate_2_vacuum_orthogonality"] = _nm(r2)
    else:
        lim = 3.0 / math.sqrt(65536)
        ok = telemetry["vacuum_correlation"] <= lim
        out["gate_2_vacuum_orthogonality"] = {
            "status": "PASS" if ok else "FAIL",
            "vacuum_correlation": telemetry["vacuum_correlation"],
            "limit": lim,
            "doc": _DOC,
        }

    r3 = {"sagnac_stress", "horizon"}
    if not r3.issubset(telemetry):
        out["gate_3_light_cone_adaptive"] = _nm(r3)
    else:
        s, h = telemetry["sagnac_stress"], telemetry["horizon"]
        ok = (s > 0.35 and h == 1) or (s <= 0.08 and h == 16)
        out["gate_3_light_cone_adaptive"] = {
            "status": "PASS" if ok else "FAIL",
            "sagnac_stress": s,
            "horizon": h,
            "doc": _DOC,
        }

    r4 = {"invalid_branch_rejection_rate"}
    if not r4.issubset(telemetry):
        out["gate_4_sagnac_branch_pruning"] = _nm(r4)
    else:
        ok = telemetry["invalid_branch_rejection_rate"] >= 0.999
        out["gate_4_sagnac_branch_pruning"] = {
            "status": "PASS" if ok else "FAIL",
            "invalid_branch_rejection_rate": telemetry["invalid_branch_rejection_rate"],
            "doc": _DOC,
        }

    r5 = {"i_norm_egress"}
    if not r5.issubset(telemetry):
        out["gate_5_egress_mutual_info"] = _nm(r5)
    else:
        ok = telemetry["i_norm_egress"] >= 0.85
        out["gate_5_egress_mutual_info"] = {
            "status": "PASS" if ok else "FAIL",
            "i_norm_egress": telemetry["i_norm_egress"],
            "doc": _DOC,
        }

    r6 = {"delta_nu", "langevin_temp"}
    if not r6.issubset(telemetry):
        out["gate_6_valence_thermodynamic_coupling"] = _nm(r6)
    else:
        dnu, T = telemetry["delta_nu"], telemetry["langevin_temp"]
        ok = (dnu > 0 and T <= 0.02) and (dnu <= 0 and T >= 0.50)
        out["gate_6_valence_thermodynamic_coupling"] = {
            "status": "PASS" if ok else "FAIL",
            "delta_nu": dnu,
            "langevin_temp": T,
            "reason": "doc formula is mutually exclusive (dnu>0 and T<=0.02) AND (dnu<=0 and T>=0.50); "
                      "literal execution is unsatisfiable by construction",
            "doc": _DOC,
        }

    return out
