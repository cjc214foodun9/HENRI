"""Contract tests for the doc-gated emergence verifier (2026-08-28).

Doc: Project_HENRI_Universal_VLA___Nested_Recursive_Pathway.md §4
(sha256 eb92bcb8a3a7c5cec1d9b078ac14fd2f3ddeb93fc52cd72a29b477da341cfc7c).
Fail-closed: missing fields -> NOT_MEASURABLE, never PASS.
"""
import math
import pytest
from arc_emergence_gates import compute_emergence_gates, GATES


def test_empty_stream_is_fail_closed():
    r = compute_emergence_gates({})
    assert set(r.keys()) == set(GATES)
    for g in GATES:
        assert r[g]["status"] == "NOT_MEASURABLE", g


def test_partial_stream_is_fail_closed():
    r = compute_emergence_gates({"sagnac_stress": 0.5, "horizon": 1})
    assert r["gate_1_procrustes_grounded"]["status"] == "NOT_MEASURABLE"
    assert r["gate_3_light_cone_adaptive"]["status"] == "PASS"


def test_gate1_thresholds():
    ok = compute_emergence_gates(
        {"goal_wave_norm": 1.0, "task_functor_error": 0.01})["gate_1_procrustes_grounded"]
    assert ok["status"] == "PASS"
    assert compute_emergence_gates(
        {"goal_wave_norm": 1.0, "task_functor_error": 0.1})["gate_1_procrustes_grounded"]["status"] == "FAIL"
    assert compute_emergence_gates(
        {"goal_wave_norm": 0.99, "task_functor_error": 0.01})["gate_1_procrustes_grounded"]["status"] == "FAIL"


def test_gate2_threshold():
    lim = 3.0 / math.sqrt(65536)
    assert compute_emergence_gates({"vacuum_correlation": lim})["gate_2_vacuum_orthogonality"]["status"] == "PASS"
    assert compute_emergence_gates({"vacuum_correlation": lim + 1e-6})["gate_2_vacuum_orthogonality"]["status"] == "FAIL"


def test_gate3_cases():
    assert compute_emergence_gates({"sagnac_stress": 0.4, "horizon": 1})["gate_3_light_cone_adaptive"]["status"] == "PASS"
    assert compute_emergence_gates({"sagnac_stress": 0.05, "horizon": 16})["gate_3_light_cone_adaptive"]["status"] == "PASS"
    assert compute_emergence_gates({"sagnac_stress": 0.2, "horizon": 2})["gate_3_light_cone_adaptive"]["status"] == "FAIL"


def test_gate4_threshold():
    assert compute_emergence_gates({"invalid_branch_rejection_rate": 0.9995})["gate_4_sagnac_branch_pruning"]["status"] == "PASS"
    assert compute_emergence_gates({"invalid_branch_rejection_rate": 0.5})["gate_4_sagnac_branch_pruning"]["status"] == "FAIL"


def test_gate5_threshold():
    assert compute_emergence_gates({"i_norm_egress": 0.9})["gate_5_egress_mutual_info"]["status"] == "PASS"
    assert compute_emergence_gates({"i_norm_egress": 0.5})["gate_5_egress_mutual_info"]["status"] == "FAIL"


def test_gate6_doc_literal_unsatisfiable():
    # Doc formula: (dnu>0 and T<=0.02) AND (dnu<=0 and T>=0.50) — mutually
    # exclusive; literal execution must FAIL, never PASS.
    r = compute_emergence_gates({"delta_nu": 1.0, "langevin_temp": 0.01})["gate_6_valence_thermodynamic_coupling"]
    assert r["status"] == "FAIL"
    assert "doc formula" in r["reason"]
