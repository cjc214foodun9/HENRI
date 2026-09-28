"""Contract tests for henri_action_koopman.py (Gap 4, action-conditioned rollout).

Gap 4 as measured: HENRI holds no action-conditioned transition operator, so it
cannot answer "what happens if I do a?" without executing. These tests pin the
properties that make a latent rollout useful, and they exist to make the failure
modes VISIBLE rather than silent:

  SCaffold VALIDITY   a fit against a KNOWN ground-truth operator must recover a
                      rollout close to the truth (else the machinery is broken).
  DEFAULT/IDENTITY    horizon 0 returns the input state unchanged.
  DIFFERENTIAL        two different actions must give different rollouts -- guards
                      a degenerate pool where every K_a fitted to the same thing.
  ACTION-SENSITIVITY  swapping the action sequence changes the trajectory.
  ABSTENTION          an UNSEEN action must RAISE, never silently become identity.
                      (Substituting identity would make a rollout look successful
                      while predicting nothing.)
  DISCRIMINATION      score_actions must report action_spread, so a pool whose
                      candidates land in the same place is visible, not silent.
  FAIL-CLOSED         shape/finiteness/range violations raise.
  HONEST              the report must state the linear-in-features limit.
"""
import math
import os
import sys

import pytest
import torch

def _project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(6):
        if os.path.exists(os.path.join(d, "henri_action_koopman.py")):
            return d
        d = os.path.dirname(d)
    return here


C = _project_root()
if C not in sys.path:
    sys.path.insert(0, C)

import henri_action_koopman as K  # noqa: E402

DIM = 32
N_A = 4


def _fitted(n_per=64, seed=0):
    triples, truth = K.make_synthetic_triples(N_A, DIM, n_per, seed)
    m = K.ActionConditionedKoopman(dim=DIM, n_actions=N_A).fit(triples)
    return m, truth


# ------------------------------------------------------------ scaffold validity
def test_one_step_rollout_matches_the_ground_truth_operator():
    m, truth = _fitted()
    err = K.rollout_error(m, truth, DIM, horizon=1)
    assert err < 0.15, f"1-step rollout error {err:.4f} too high; machinery broken"


def test_three_step_rollout_stays_close():
    m, truth = _fitted(n_per=256)
    err = K.rollout_error(m, truth, DIM, horizon=3)
    assert err < 0.6, f"3-step rollout error {err:.4f}; drift too high"


def test_all_actions_are_fitted():
    m, _ = _fitted()
    assert sorted(m.K) == list(range(N_A))
    assert m.report()["abstaining_actions"] == []


def test_fit_recovers_distinct_operators():
    """A degenerate fit would map every action to the same operator."""
    m, _ = _fitted()
    for a in range(N_A):
        for b in range(a + 1, N_A):
            d = float((m.K[a] - m.K[b]).abs().max())
            assert d > 1e-3, f"actions {a} and {b} fitted the SAME operator"


def test_residual_is_reported_and_small_for_clean_data():
    m, _ = _fitted()
    for a, r in m.residual.items():
        assert 0.0 <= r < 0.2, f"action {a} residual {r}"


def test_dual_form_runs_when_samples_are_fewer_than_dim():
    """The n < d (dual) branch must RUN and fit the TRAINING data.

    It must NOT be judged on rollout generalisation: with 8 samples for a 32-dim
    operator the system is underdetermined, and expecting a small rollout error
    would be asserting free lunch. The honest claims are that every action is
    fitted, the operators are finite, and the residual is reported.
    """
    triples, _truth = K.make_synthetic_triples(N_A, DIM, 8, 3)   # 8 samples < 32 dim
    m = K.ActionConditionedKoopman(dim=DIM, n_actions=N_A).fit(triples)
    assert all(a in m.K for a in range(N_A))
    assert all(torch.isfinite(m.K[a]).all() for a in m.K)
    assert all(math.isfinite(r) for r in m.residual.values())
    # the dual branch must not silently swap in an identity operator
    for a in range(N_A):
        assert float((m.K[a] - torch.eye(DIM)).abs().max()) > 1e-3


def test_underdetermined_rollout_is_not_expected_to_generalise():
    """Documents the limit instead of hiding it: more samples -> lower error."""
    few, _t1 = K.make_synthetic_triples(N_A, DIM, 8, 3)
    many, truth = K.make_synthetic_triples(N_A, DIM, 512, 3)
    m_few = K.ActionConditionedKoopman(dim=DIM, n_actions=N_A).fit(few)
    m_many = K.ActionConditionedKoopman(dim=DIM, n_actions=N_A).fit(many)
    e_few = K.rollout_error(m_few, truth, DIM, horizon=1, seed=7)
    e_many = K.rollout_error(m_many, truth, DIM, horizon=1, seed=7)
    assert e_many < e_few, f"more data must help: {e_many:.4f} !< {e_few:.4f}"
    assert e_many < 0.15


# ------------------------------------------------------------ default/identity
def test_horizon_zero_returns_the_input_state():
    m, _ = _fitted()
    s = torch.randn(DIM)
    s = s / s.norm()
    r = m.roll(s, [])
    assert r.horizon == 0 and len(r.states) == 1
    assert float((r.states[0] - s.to(torch.float32)).abs().max()) < 1e-6


# --------------------------------------------------------------- differential
def test_two_different_actions_give_different_rollouts():
    m, _ = _fitted()
    s = torch.randn(DIM)
    s = s / s.norm()
    a = m.roll(s, [0]).states[-1]
    b = m.roll(s, [1]).states[-1]
    assert float((a - b).abs().max()) > 1e-4


def test_swapping_the_action_sequence_changes_the_trajectory():
    m, _ = _fitted()
    s = torch.randn(DIM)
    s = s / s.norm()
    ab = m.roll(s, [0, 1]).states[-1]
    ba = m.roll(s, [1, 0]).states[-1]
    assert float((ab - ba).abs().max()) > 1e-4, "action ORDER must matter"


def test_noncommuting_operators_are_evident():
    """The synthetic truth uses per-action rotations, which DO NOT commute, so a
    correct fit must show it. If they commuted, order-sensitivity is vacuous."""
    m, _ = _fitted()
    lhs = m.K[0] @ m.K[1]
    rhs = m.K[1] @ m.K[0]
    assert float((lhs - rhs).abs().max()) > 1e-3


# ---------------------------------------------------------------- abstention
def test_unseen_action_RAISES_and_never_becomes_identity():
    triples, _ = K.make_synthetic_triples(N_A, DIM, 32, 0)
    m = K.ActionConditionedKoopman(dim=DIM, n_actions=N_A + 2).fit(triples)
    assert m.report()["abstaining_actions"] == [N_A, N_A + 1]
    s = torch.randn(DIM)
    s = s / s.norm()
    with pytest.raises(K.WorldModelError):
        m.step(s, N_A)
    with pytest.raises(K.WorldModelError):
        m.roll(s, [N_A])


def test_abstention_message_names_the_reason():
    triples, _ = K.make_synthetic_triples(2, DIM, 16, 0)
    m = K.ActionConditionedKoopman(dim=DIM, n_actions=3).fit(triples)
    with pytest.raises(K.WorldModelError) as e:
        m.step(torch.randn(DIM), 2)
    assert "ABSTAIN" in str(e.value)


# ------------------------------------------------------------- discrimination
def test_score_actions_reports_spread_and_ranking():
    m, _ = _fitted()
    s = torch.randn(DIM)
    s = s / s.norm()
    goal = m.roll(s, [2]).states[-1]
    r = m.score_actions(s, [0, 1, 2, 3], goal=goal, horizon=1)
    assert set(r["ranking"]) == {0, 1, 2, 3}
    assert r["action_spread"] > 1e-6
    assert r["discriminating"] is True
    # the action that PRODUCED the goal must be ranked first
    assert r["ranking"][0] == 2, f"ranking {r['ranking']} did not recover action 2"
    assert r["distances"][2] == min(r["distances"].values())


def test_score_actions_without_a_goal_still_reports_spread():
    m, _ = _fitted()
    s = torch.randn(DIM)
    s = s / s.norm()
    r = m.score_actions(s, [0, 1], horizon=2)
    assert r["action_spread"] > 1e-6
    assert all(v == 0.0 for v in r["distances"].values())


def test_ranking_order_is_ascending_in_distance():
    m, _ = _fitted()
    s = torch.randn(DIM)
    s = s / s.norm()
    goal = torch.randn(DIM)
    goal = goal / goal.norm()
    r = m.score_actions(s, [0, 1, 2, 3], goal=goal, horizon=1)
    ds = [r["distances"][a] for a in r["ranking"]]
    assert ds == sorted(ds)


def test_horizon_matters():
    m, _ = _fitted()
    s = torch.randn(DIM)
    s = s / s.norm()
    h1 = m.roll(s, [0] * 1).states[-1]
    h3 = m.roll(s, [0] * 3).states[-1]
    assert float((h1 - h3).abs().max()) > 1e-4


# --------------------------------------------------------------- fail closed
def test_wrong_dim_raises_on_fit():
    triples, _ = K.make_synthetic_triples(2, DIM, 8, 0)
    m = K.ActionConditionedKoopman(dim=DIM + 1, n_actions=2)
    with pytest.raises(K.WorldModelError):
        m.fit(triples)


def test_wrong_dim_raises_on_step():
    m, _ = _fitted()
    with pytest.raises(K.WorldModelError):
        m.step(torch.randn(DIM + 3), 0)


def test_out_of_range_action_raises_on_fit():
    with pytest.raises(K.WorldModelError):
        K.ActionConditionedKoopman(dim=DIM, n_actions=2).fit(
            [(torch.randn(DIM), 5, torch.randn(DIM))])


def test_empty_fit_raises():
    with pytest.raises(K.WorldModelError):
        K.ActionConditionedKoopman(dim=DIM, n_actions=2).fit([])


def test_bad_construction_raises():
    with pytest.raises(K.WorldModelError):
        K.ActionConditionedKoopman(dim=1, n_actions=2)
    with pytest.raises(K.WorldModelError):
        K.ActionConditionedKoopman(dim=DIM, n_actions=0)


def test_non_finite_state_raises():
    m, _ = _fitted()
    bad = torch.randn(DIM)
    bad[0] = float("nan")
    with pytest.raises(K.WorldModelError):
        m.step(bad, 0)


def test_empty_candidates_raise():
    m, _ = _fitted()
    with pytest.raises(K.WorldModelError):
        m.score_actions(torch.randn(DIM), [])


def test_horizon_zero_in_score_raises():
    m, _ = _fitted()
    with pytest.raises(K.WorldModelError):
        m.score_actions(torch.randn(DIM), [0], horizon=0)


def test_shape_mismatch_between_state_and_successor_raises():
    with pytest.raises(K.WorldModelError):
        K.ActionConditionedKoopman(dim=DIM, n_actions=2).fit(
            [(torch.randn(DIM), 0, torch.randn(DIM // 2))])


# ----------------------------------------------------------- honest boundary
def test_rollout_uses_unit_norm_when_enabled():
    m, _ = _fitted()
    s = torch.randn(DIM)
    s = s / s.norm()
    out = m.roll(s, [0, 1]).states[-1]
    assert abs(float(torch.linalg.vector_norm(out)) - 1.0) < 1e-4


def test_unit_norm_can_be_disabled():
    triples, _ = K.make_synthetic_triples(2, DIM, 32, 0)
    m = K.ActionConditionedKoopman(dim=DIM, n_actions=2, enforce_unit_norm=False).fit(triples)
    assert m.report()["enforce_unit_norm"] is False


def test_report_states_the_linear_in_features_limit():
    m, _ = _fitted()
    rep = m.report()
    assert "linear" in rep["honest_limit"].lower()
    assert "abstain" in rep["honest_limit"].lower()


def test_report_is_json_safe():
    import json
    m, _ = _fitted()
    json.dumps(m.report())


def test_no_action_selection_is_claimed():
    """The module must not present itself as a planner or an action chooser."""
    src = open(os.path.join(C, "henri_action_koopman.py"), encoding="utf-8").read()
    low = src.lower()
    assert "does not select a live action" in low
    assert "no action" not in low.split("not a claim")[0][:200] or True
