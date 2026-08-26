"""Contract test: Gate 1 execution carrier pure algebra.

Validates the frozen Gate 1 execution logic WITHOUT torch: derangement
construction, lexicographic split, Delta-I algebra, Spearman rho, paired
bootstrap percentile, verdict mapping, and frozen contract constants.
"""
import json
import math
import os
import sys

HENRI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))  # <repo>/HENRI V2
sys.path.insert(0, os.path.join(HENRI_DIR, "experiments", "verification"))

from promotion_gate1_execute import (  # noqa: E402
    BUDGETS, PRIMARY_BUDGET, VERDICT_ORDER,
    bootstrap_percentile, delta_i, lexicographic_split, make_derangement,
    spearman_rho, verdict_for,
)


def _load_contract():
    p = os.path.join(HENRI_DIR, "experiments", "verification",
                     "promotion_gate1_contract.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def test_budgets_match_contract():
    c = _load_contract()
    assert c["budgets"] == BUDGETS
    assert c["primary_budget"] == PRIMARY_BUDGET
    assert sorted(c["arms"]) == sorted(["R", "S", "A", "N", "P"])


def test_verdict_precedence_matches_contract():
    c = _load_contract()
    # Contract lists the precedence order top-down (highest first).
    assert VERDICT_ORDER == c["verdict_precedence"]


def test_delta_i_algebra():
    assert delta_i(0.5, 0.3) == 0.2
    assert delta_i(0.3, 0.3) == 0.0
    assert delta_i(0.2, 0.4) == -0.2


def test_make_derangement_no_fixed_points():
    ids = ["a0", "a1", "a2", "a3", "a4"]
    for seed in range(20):
        p = make_derangement(ids, seed)
        assert sorted(p) == sorted(ids)
        assert all(p[a] != a for a in ids)
        assert len(set(p.values())) == len(ids)


def test_lexicographic_split_rule():
    ids = ["e", "b", "d", "a", "c"]
    cal, evl = lexicographic_split(ids, k_eval=3)
    assert cal == ["a", "b"]
    assert evl == ["c", "d", "e"]
    cal2, evl2 = lexicographic_split(["z", "y", "x"], k_eval=2)
    assert cal2 == ["x"] and evl2 == ["y", "z"]


def test_spearman_monotone_and_inverse():
    assert abs(spearman_rho([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) - 1.0) < 1e-9
    assert abs(spearman_rho([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) + 1.0) < 1e-9
    # log-budget grid from the contract
    n_log = [math.log(n) for n in BUDGETS]
    assert abs(spearman_rho(n_log, [0.01 * i for i in range(1, 6)]) - 1.0) < 1e-9


def test_bootstrap_paired_percentile():
    # Constant positive differences -> lb > 0.
    paired = [(0.30, 0.10)] * 20
    ci = bootstrap_percentile(paired, b=2000, seed=7)
    assert ci["mean"] > 0.0 and ci["lb"] > 0.0
    # Constant zero differences -> CI includes 0.
    ci0 = bootstrap_percentile([(0.2, 0.2)] * 20, b=2000, seed=7)
    assert abs(ci0["mean"]) < 1e-12


def test_verdict_mapping_frozen():
    # FEW_SHOT_SCALING: DeltaI(32) > 0 lb > 0, trend passes.
    i_r = {1: 0.001, 2: 0.002, 5: 0.004, 10: 0.008, 32: 0.03}
    v = verdict_for({"mean": 0.05, "lb": 0.01}, i_r, engaged=True)
    assert v == "FEW_SHOT_SCALING"
    # ACTION_INFORMATION_GAIN: lb > 0 but trend fails (flat I_R).
    i_flat = {1: 0.01, 2: 0.01, 5: 0.01, 10: 0.01, 32: 0.01}
    v2 = verdict_for({"mean": 0.05, "lb": 0.01}, i_flat, engaged=True)
    assert v2 == "ACTION_INFORMATION_GAIN"
    # ENGAGED: CI includes 0.
    v3 = verdict_for({"mean": 0.001, "lb": -0.01}, i_r, engaged=True)
    assert v3 == "ENGAGED"
    # FALSIFIED_NO_ENGAGEMENT.
    v4 = verdict_for({"mean": 0.05, "lb": 0.01}, i_r, engaged=False)
    assert v4 == "FALSIFIED_NO_ENGAGEMENT"


def test_kill_experiment_conditions():
    c = _load_contract()
    kills = c["kill_experiments"]
    assert len(kills) == 3
    assert any("DeltaI(32) <= 0" in k for k in kills)
    assert any("I_R(32) <= 0" in k for k in kills)
    assert any("DeltaI(32) < 0" in k for k in kills)
