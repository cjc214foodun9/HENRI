"""Contract test: promotion_gate1_contract.json (Gate 1 few-shot scaling prereg).

Pure-python, no torch import. Validates the frozen preregistration contract:
budgets, arms, Delta-I algebra, primary budget, trend gate, verdict precedence,
split rule, support requirements, flags, and evidence labels.
"""
import json
from pathlib import Path

CONTRACT = Path(__file__).resolve().parents[2] / "experiments" / "verification" / "promotion_gate1_contract.json"


def _load() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_contract_file_exists_and_parses():
    assert CONTRACT.exists(), f"missing contract: {CONTRACT}"
    c = _load()
    assert c["contract_id"].startswith("promotion_gate1")
    assert c["schema_version"] == 1


def test_budgets_frozen_and_primary_inside():
    c = _load()
    assert c["budgets"] == [1, 2, 5, 10, 32]
    assert c["primary_budget"] == 32
    assert c["primary_budget"] in c["budgets"]


def test_arm_set_exact():
    c = _load()
    assert set(c["arms"]) == {"R", "S", "A", "N", "P"}
    for arm, desc in c["arms"].items():
        assert isinstance(desc, str) and len(desc) > 8


def test_deltaI_algebra_consistent():
    c = _load()
    d = c["metrics"]["delta_I"]
    # DeltaI must equal L_S - L_R under the I definition (I_a = L_N - L_a).
    assert "DeltaI(n) = I_R(n) - I_S(n) = L_S(n) - L_R(n)" in d
    assert "I_a(n) = L_N(n) - L_a(n)" in c["metrics"]["improvement"]


def test_trend_gate_thresholds():
    c = _load()
    g = c["scaling_trend_gate"]
    assert g["spearman_rho_min"] == 0.60
    assert g["improvement_min"] == 0.02
    assert "log(n)" in g["rho_on"]
    assert "I_R(32) - I_R(1)" in g["improvement_on"]


def test_verdict_precedence_total_order():
    c = _load()
    vp = c["verdict_precedence"]
    assert vp == [
        "BLOCKED_INFRA",
        "BLOCKED_NO_EVAL_COVERAGE",
        "FALSIFIED_NO_ENGAGEMENT",
        "ENGAGED",
        "ACTION_INFORMATION_GAIN",
        "FEW_SHOT_SCALING",
    ]
    assert len(vp) == len(set(vp)), "verdict precedence must be a total order (unique entries)"


def test_accept_criteria_consistent():
    c = _load()
    assert c["accept_criteria"]["verdict"] == "FEW_SHOT_SCALING"
    req = c["accept_criteria"]["requires"]
    assert any("DeltaI(32) > 0" in r for r in req)
    assert any("trend gate" in r for r in req)


def test_kill_experiments_present():
    c = _load()
    assert len(c["kill_experiments"]) == 3
    assert any("shuffled beats real" in k for k in c["kill_experiments"])


def test_split_rule_lexicographic():
    c = _load()
    assert c["split"]["episode_ordering"] == "lexicographic by episode_id string"
    assert c["split"]["evaluation"].startswith("last K_eval")
    assert "hash-pinned" in c["split"]["manifest"]


def test_support_and_default_off():
    c = _load()
    assert c["support"]["primary_budget_min_per_action"] == 2
    assert c["support"]["low_shot_diagnostic_only"] == [1, 2, 5]
    assert c["flags"]["HENRI_FREEZE_LEARNING"] == "1"
    assert c["flags"]["HENRI_GATE1_ONLINE_ADAPTATION"].startswith("0")
