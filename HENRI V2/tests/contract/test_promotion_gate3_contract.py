"""Contract test: promotion_gate3_contract.json (Gate 3 egress legal bounds).

Pure-python, no torch import. Validates the frozen preregistration contract:
wiring-defect correction, fixture, checks G1-G6, verdict precedence, scope.
"""
import json
from pathlib import Path

CONTRACT = Path(__file__).resolve().parents[2] / "experiments" / "verification" / "promotion_gate3_contract.json"


def _load() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_contract_file_exists_and_parses():
    assert CONTRACT.exists(), f"missing contract: {CONTRACT}"
    c = _load()
    assert c["contract_id"].startswith("promotion_gate3")
    assert c["schema_version"] == 1


def test_wiring_defect_corrected():
    c = _load()
    d = c["wiring_defect_corrected"]
    assert "sagnac_mcts_planner.py" in d["proposed_target"]
    assert "arc_curriculum_replay.py" in d["live_target"]
    assert "DIAGNOSTIC" in d["syntax_error_gate"]
    assert "default-OFF" in d["lexical_snap"]


def test_fixture_frozen():
    c = _load()
    f = c["fixture"]
    assert f["seed"] == 20260826
    assert f["num_blocks"] == 8192
    assert f["block_dim"] == 8
    assert f["d_model"] == 65536
    assert f["max_grid_dim"] == 128
    assert "99" in f["oob_color_grid"]
    assert "200x200" in f["oversized_grid"]


def test_checks_g1_g6_present():
    c = _load()
    assert set(c["checks"]) == {"G1", "G2", "G3", "G4", "G5", "G6"}
    assert "raise" in c["checks"]["G1"]
    assert "clamped" in c["checks"]["G2"]
    assert "[1, 8192, 8]" in c["checks"]["G3"]
    assert "EgressFailClosedError" in c["checks"]["G4"]
    assert "EgressFailClosedError" in c["checks"]["G5"]
    assert "EgressFailClosedError" in c["checks"]["G6"]


def test_verdict_precedence_total_order():
    c = _load()
    vp = c["verdict_precedence"]
    assert vp == [
        "BLOCKED_INFRA",
        "FAIL_DIMENSIONAL_BOUNDS",
        "FAIL_PALETTE_LEGALITY",
        "FAIL_EGRESS_FAIL_CLOSED",
        "GATE3_VALIDATION_PASS",
    ]
    assert len(vp) == len(set(vp)), "verdict precedence must be a total order"


def test_accept_criteria_consistent():
    c = _load()
    assert c["accept_criteria"]["verdict"] == "GATE3_VALIDATION_PASS"
    req = c["accept_criteria"]["requires"]
    assert any("G1" in r for r in req)
    assert any("G2" in r for r in req)
    assert any("G6" in r for r in req)


def test_scope_restricted():
    c = _load()
    assert "no production change" in c["scope"]
    assert "no capability claim" in c["scope"]
    assert "no benchmark score" in c["scope"]
    assert "no snap promotion" in c["scope"]
