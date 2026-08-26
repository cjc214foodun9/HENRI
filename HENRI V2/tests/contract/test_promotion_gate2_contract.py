"""Contract test: promotion_gate2_contract.json (Gate 2 Procrustes validation).

Pure-python, no torch import. Validates the frozen preregistration contract:
margins, controls, fixture, verdict precedence, legacy-gate rejection, scope.
"""
import json
from pathlib import Path

CONTRACT = Path(__file__).resolve().parents[2] / "experiments" / "verification" / "promotion_gate2_contract.json"


def _load() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_contract_file_exists_and_parses():
    assert CONTRACT.exists(), f"missing contract: {CONTRACT}"
    c = _load()
    assert c["contract_id"].startswith("promotion_gate2")
    assert c["schema_version"] == 1


def test_margins_frozen():
    c = _load()
    m = c["margins"]
    assert m["recon_cos_min"] == 0.95
    assert m["known_cos_min"] == 0.95
    assert m["shuffled_cos_max"] == 0.60
    assert m["orthogonality_err_max"] == 1e-4


def test_fixture_frozen():
    c = _load()
    f = c["fixture"]
    assert f["seed"] == 20260826
    assert f["num_blocks"] == 8192
    assert f["block_dim"] == 8
    assert f["m_total"] == 14
    assert f["m_calibration"] == 10
    assert f["m_heldout"] == 4
    assert "episode/task-disjoint" in f["split"]
    assert "no fixed points" in f["shuffle"]


def test_controls_present():
    c = _load()
    assert set(c["controls"]) == {"C1", "C2", "C3"}
    assert "known-transform" in c["controls"]["C1"]
    assert "calibration only" in c["controls"]["C2"]
    assert "deranged" in c["controls"]["C3"]


def test_verdict_precedence_total_order():
    c = _load()
    vp = c["verdict_precedence"]
    assert vp == [
        "BLOCKED_INFRA",
        "FAIL_SHAPE",
        "FAIL_ORTHOGONALITY",
        "FAIL_KNOWN_TRANSFORM",
        "FAIL_SHUFFLE_CONTROL",
        "FAIL_RECONSTRUCTION",
        "GATE2_VALIDATION_PASS",
    ]
    assert len(vp) == len(set(vp)), "verdict precedence must be a total order"


def test_accept_criteria_consistent():
    c = _load()
    assert c["accept_criteria"]["verdict"] == "GATE2_VALIDATION_PASS"
    req = c["accept_criteria"]["requires"]
    assert any("C2" in r for r in req)
    assert any("C1" in r for r in req)
    assert any("C3" in r for r in req)


def test_legacy_gate_rejected():
    c = _load()
    assert "self-referential" in c["legacy_gate_rejected"]["reasons"][0]
    assert "scale-dependent" in c["legacy_gate_rejected"]["reasons"][1]


def test_kill_experiments_and_scope():
    c = _load()
    assert len(c["kill_experiments"]) == 3
    assert any("FAIL_RECONSTRUCTION" in k for k in c["kill_experiments"])
    assert any("FAIL_SHUFFLE_CONTROL" in k for k in c["kill_experiments"])
    assert "no reimplementation" in c["scope"]
    assert "no production enablement" in c["scope"]
