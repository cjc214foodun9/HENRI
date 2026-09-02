"""Contract tests for Carrier P1 goal-grounded policy steering engine.

Packet: Carrier_P1_SpecContract___Alignment_Probe.md
(SHA-256 06e667c33134f82924c0d9500dfa8c8ee8ab5c1e2dd6484478296b4161fd3989).
Carrier: P1_GOAL_GROUNDED_POLICY_STEERING.
Base: arc_g7_calibrated_engine.G7CalibratedAffordanceEngine @ 4c71d4d.

Covers: C1 default-OFF flag fail-closed; C2 unbound fallback byte-identical to
the inherited G7/G4 scorer; C3 action-conditioned potential-drop discrimination
on the full-wave top-k arm (packet eq. 3.2); C4 score-formula recomputation
j(a) = (clamp(DeltaV(a), -1, 1) + 1) * (pi_a)^H; C5 bridge-arm fail-closed
without ingress; C6 P1 verdict vocabulary (LG1/LG2/LG3 mapping).
"""
import math
import sys
import pathlib

import pytest
import torch
import torch.nn.functional as F

TESTS = pathlib.Path(__file__).resolve()
ROOT = TESTS.parents[2]  # .../HENRI V2 (code dir)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "verification"))

from arc_p1_goal_steering_engine import (  # noqa: E402
    DV_CLAMP,
    FLAG_P1,
    P1_LG1_MIN_DELTA_NU,
    P1GoalSteeringEngine,
    require_p1_flag,
)
from arc_g7_calibrated_engine import G7CalibratedAffordanceEngine  # noqa: E402

SEED = 20260930
HORIZON = 2


def _unit(t):
    return F.normalize(t.reshape(-1), p=2, dim=-1)


def _householder(v, u):
    """Orthogonal reflection mapping unit v -> unit u (exact up to fp)."""
    a = v - u
    return torch.eye(v.numel()) - 2.0 * torch.outer(a, a) / torch.dot(a, a)


def _engine(cls=P1GoalSteeringEngine, n_actions=2, device="cpu",
            trans_g4=None, masks=None, bridge_flags=None, bridge_trans=None,
            ingress=None, full_goals=None):
    trans_g4 = trans_g4 or {}
    masks = masks or {}
    bridge_flags = bridge_flags or {}
    bridge_trans = bridge_trans or {}
    g = torch.Generator().manual_seed(SEED)
    t_pow = torch.eye(64).repeat(n_actions, HORIZON, 1, 1)
    eng = cls(
        transitions_g4=trans_g4,
        topk_masks=masks,
        theta=[0.0] * n_actions,
        tau=[1.0] * n_actions,
        bridge_transitions=bridge_trans,
        bridge_route_flags=bridge_flags,
        generators=torch.zeros(n_actions, 64, 64),
        transitions=[torch.eye(64) for _ in range(n_actions)],
        t_pow=t_pow,
        recon={},
        tau_cal=[1.0] * n_actions,
        action_names=["ACTION%d" % (i + 1) for i in range(n_actions)],
        n_actions=n_actions,
        seed=SEED,
        horizon=HORIZON,
        device=device,
        waypoints=None,
        ingress=ingress,
    )
    if full_goals is not None:
        eng._p1_full_goals = dict(full_goals)
        for name in full_goals:
            eng.p1_bind_env_goal(name)
    return eng


# ---------------------------------------------------------------------------
# C1: default-OFF flag fail-closed
# ---------------------------------------------------------------------------

def test_p1_flag_fail_closed(monkeypatch):
    monkeypatch.delenv(FLAG_P1, raising=False)
    with pytest.raises(SystemExit):
        require_p1_flag()
    monkeypatch.setenv(FLAG_P1, "1")
    require_p1_flag()  # must not raise


# ---------------------------------------------------------------------------
# C2: unbound fallback is byte-identical to the inherited scorer
# ---------------------------------------------------------------------------

def test_p1_unbound_fallback_matches_g7():
    e7 = _engine(cls=G7CalibratedAffordanceEngine)
    e1 = _engine(cls=P1GoalSteeringEngine)
    psi64 = _unit(torch.randn(64, generator=torch.Generator().manual_seed(1)))
    psi_full = _unit(torch.randn(4, 8, generator=torch.Generator().manual_seed(2))).view(4, 8)
    o7 = e7.score_all_actions(psi64, psi_full, None)
    o1 = e1.score_all_actions(psi64, psi_full, None)
    assert o7 == o1
    assert set(o1) == {"ACTION1", "ACTION2"}


# ---------------------------------------------------------------------------
# C3: action-conditioned potential-drop discrimination (full-wave top-k arm)
# ---------------------------------------------------------------------------

def _goal_discrimination_fixture(seed=5):
    g = torch.Generator().manual_seed(seed)
    e0 = torch.zeros(8)
    e0[0] = 1.0
    th = 0.5
    u = torch.zeros(8)
    u[0] = math.cos(th)
    u[1] = math.sin(th)
    u2 = torch.zeros(8)
    u2[0] = -math.sin(th)
    u2[1] = math.cos(th)          # u2 perpendicular to u
    H0 = _householder(e0, u)      # maps psi block0 e0 -> u (toward goal)
    H1 = _householder(e0, u2)     # maps psi block0 e0 -> u2 (orthogonal)
    psi = torch.randn(4, 8, generator=g)
    psi[0] = e0
    psi[1:] = psi[1:] * 0.01      # concentrate mass on block 0
    psi = _unit(psi).view(4, 8)
    g_raw = psi.clone()
    g_raw[0] = H0 @ psi[0]
    goal = _unit(g_raw)
    trans_g4 = {0: {0: H0}, 1: {0: H1}}
    masks = {0: torch.tensor([0]), 1: torch.tensor([0])}
    return psi, goal, trans_g4, masks


def test_p1_goal_discrimination_full_wave():
    psi, goal, trans_g4, masks = _goal_discrimination_fixture()
    eng = _engine(trans_g4=trans_g4, masks=masks, full_goals={"e1": goal})
    psi64 = _unit(torch.randn(64, generator=torch.Generator().manual_seed(6)))
    out = eng.score_all_actions(psi64, psi, None)
    drops = eng.p1_last["potential_drops"]
    base = eng.p1_last["base_align"]
    # Action 0 moves the state toward the goal; action 1 moves it orthogonal.
    assert drops[0] > 0.1
    assert drops[1] < -0.1
    assert drops[0] > drops[1] + 0.5
    assert 0.0 < base < 1.0
    assert max(out, key=out.get) == "ACTION1"


def test_p1_score_formula_recomputed():
    """j(a) = (clamp(DeltaV(a), -1, 1) + 1) * (pi_a)^H with pi = 0.5 (no pairs)."""
    psi, goal, trans_g4, masks = _goal_discrimination_fixture()
    eng = _engine(trans_g4=trans_g4, masks=masks, full_goals={"e1": goal})
    psi64 = _unit(torch.randn(64, generator=torch.Generator().manual_seed(7)))
    out = eng.score_all_actions(psi64, psi, None)
    drops = eng.p1_last["potential_drops"]
    expected = [
        (min(max(float(d), -DV_CLAMP), DV_CLAMP) + 1.0) * (0.5 ** HORIZON)
        for d in drops
    ]
    for i in range(2):
        assert abs(out["ACTION%d" % (i + 1)] - expected[i]) < 1e-5


# ---------------------------------------------------------------------------
# C5: bridge arm fail-closed without ingress
# ---------------------------------------------------------------------------

def test_p1_bridge_arm_fail_closed_no_ingress():
    bridge_flags = {0: True}
    bridge_trans = {0: {0: torch.eye(8)}}
    goal = _unit(torch.randn(32, generator=torch.Generator().manual_seed(8)))
    eng = _engine(bridge_flags=bridge_flags, bridge_trans=bridge_trans,
                  ingress=None, full_goals={"e1": goal})
    psi64 = _unit(torch.randn(64, generator=torch.Generator().manual_seed(9)))
    psi_full = _unit(torch.randn(4, 8, generator=torch.Generator().manual_seed(10))).view(4, 8)
    out = eng.score_all_actions(psi64, psi_full, None)
    assert eng.p1_last["potential_drops"][0] == 0.0
    assert all(math.isfinite(v) for v in out.values())


# ---------------------------------------------------------------------------
# C6: P1 verdict vocabulary
# ---------------------------------------------------------------------------

def test_p1_verdict_vocabulary():
    eng = _engine()
    assert eng._decide_verdict(None, 0, None, None, 0, 0) == \
        "P1_GATE_LG2_SOLVED_FAILED"
    assert eng._decide_verdict(2.5, 1, None, None, 10, 1) == \
        "P1_GATE_LG3_LATENCY_FAILED"
    assert eng._decide_verdict(None, 1, 0.01, None, 10, 1) == \
        "P1_GATE_LG1_STAGNATION"
    assert eng._decide_verdict(None, 1, None, 0.2, 10, 1) == \
        "P1_GATE_G4_AFFORDANCE_FAILED"
    assert eng._decide_verdict(None, 1, 0.10, 0.001, 10, 1) == \
        "P1_POLICY_GROUNDING_VERIFIED"
    assert P1_LG1_MIN_DELTA_NU == 0.0500
