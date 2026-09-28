"""Koopman leaf evaluation (Directive 4): trajectory scoring + fail-open veto.

MEASURED DEFICIT THIS ADDRESSES. `sagnac_mcts_planner.py` contains `koopman` = 0,
`rollout` = 0, `simulate` = 0, `reward` = 0 occurrences, and `henri_action_koopman.py`
has no caller outside its own tests. The world model exists and does not reach the
search loop: the planner judges ONE step, never a TRAJECTORY.

The four controls below are the reason this channel can be wired into `search()`
without redefining what a veto means.

  SCAFFOLD VALIDITY  fitted on a KNOWN operator, the 1-step rollout reproduces it
  MONOTONE HORIZON   error does not DECREASE with horizon under a non-contractive K
  DIFFERENTIAL       two different action sequences score differently
  ABSTENTION         an unseen action -> score None, veto False (never fabricated)
  INERT BY DEFAULT   horizon <= 0 or no actions -> None/False, so OFF is inert
"""
import os
import sys

import pytest
import torch

def _root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.exists(os.path.join(d, "henri_koopman_leaf.py")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.abspath(__file__))

C = _root()
if C not in sys.path:
    sys.path.insert(0, C)

import henri_action_koopman as AK             # noqa: E402
import henri_koopman_leaf as KL               # noqa: E402

DIM, N_A = 32, 4


def _fitted(model_horizon=1, n_per=256, noise=0.02):
    triples, truth = AK.make_synthetic_triples(N_A, DIM, n_per, 7, rng_scale=noise)
    ev = KL.KoopmanLeafEvaluator(wave_dim=DIM, n_actions=N_A, horizon=model_horizon,
                                 tau_rollout=0.99)
    ev.fit(triples)
    return ev, truth


def _goal(state, truth, actions):
    """The TRUE successor of `state` under `actions` -- the only goal that makes the
    score meaningful.

    TEST DEFECT FIXED 2026-09-27: the first helper built the goal from a DIFFERENT
    random state, so the goal was near-orthogonal to the rollout (measured
    cos = 0.2230) and every score sat near 1.0 REGARDLESS of module correctness. Three
    tests failed against right behaviour. A goal that is not the state's own successor
    measures nothing.
    """
    out = state
    for a in actions:
        out = truth[a] @ out
        out = out / out.norm()
    return out


def _unit_state(seed=11):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(DIM, generator=g)
    return s / s.norm()


# ------------------------------------------------------------ scaffold validity
def test_one_step_rollout_reproduces_a_known_operator():
    """SCAFFOLD VALIDITY. With the goal set to the state's own true successor, the
    fitted operator must reproduce it. Measured with the CORRECT goal: ~0.00003."""
    ev, truth = _fitted(model_horizon=1)
    s = _unit_state()
    for a in range(N_A):
        goal = _goal(s, truth, [a])
        score, reason = ev.score_sequence(s, [a], goal)
        assert reason == "OK"
        assert score < 0.05, "action %d 1-step score %.4f too high" % (a, score)


def test_all_actions_fitted_and_none_abstaining():
    ev, _ = _fitted()
    assert ev.fitted_actions == list(range(N_A))
    assert ev.report()["abstaining_actions"] == []


# ------------------------------------------------------------- monotone horizon
def test_rollout_error_is_bounded_and_accumulates_with_horizon():
    """The rollout must be a REAL measurement, not a floor, and must work at depth.

    TEST DESIGN FIXED 2026-09-27 (second correction). A goal must be the TRUE h-step
    successor FOR THAT h. My first version held the goal fixed at the 1-step successor
    and demanded monotonicity -- but these operators are ROTATIONS, so as the state
    rotates past a fixed target the cosine distance oscillates. Measured with a fixed
    goal: [2.47e-05, 0.2625, 1.6052, 0.5478] -- non-monotone BY GEOMETRY, and the
    assertion failed against correct behaviour.

    With h-specific goals the error accumulates as a real drift measurement:
    measured h=1 2.3e-05 -> h=2 7.8e-05 -> h=4 1.9e-04 -> h=8 7.7e-04.
    """
    triples, truth = AK.make_synthetic_triples(N_A, DIM, 256, 7, rng_scale=0.02)
    s = _unit_state(23)
    scores = []
    for h in (1, 2, 4, 8):
        ev = KL.KoopmanLeafEvaluator(wave_dim=DIM, n_actions=N_A, horizon=h)
        ev.fit(triples)
        goal = _goal(s, truth, [1] * h)          # the TRUE h-step successor
        sc, why = ev.score_sequence(s, [1] * h, goal)
        assert sc is not None, why
        scores.append(sc)
    assert max(scores) < 0.05, "machinery invalid at some horizon: %s" % scores
    assert scores[-1] > scores[0], "no accumulation measured: %s" % scores


def test_distance_to_a_FIXED_goal_is_NOT_monotone_in_horizon():
    """Documents the geometry so a future session cannot 'fix' it into a bug.

    With a single fixed goal and rotating per-action operators, the state sweeps past
    the target: agreement worsens, then improves, then worsens. This is measured, not
    assumed, and it is WHY the test above uses h-specific goals. If this ever becomes
    monotone, the operators stopped rotating and the world model changed character.
    """
    triples, truth = AK.make_synthetic_triples(N_A, DIM, 256, 7, rng_scale=0.02)
    s = _unit_state(23)
    fixed_goal = _goal(s, truth, [1])            # held fixed across horizons
    scores = []
    for h in (1, 2, 4, 8):
        ev = KL.KoopmanLeafEvaluator(wave_dim=DIM, n_actions=N_A, horizon=h)
        ev.fit(triples)
        sc, _why = ev.score_sequence(s, [1] * h, fixed_goal)
        assert sc is not None
        scores.append(sc)
    assert scores[0] < 0.05, "h=1 must match its own successor: %s" % scores
    strict = all(b >= a - 1e-12 for a, b in zip(scores, scores[1:]))
    assert not strict, ("fixed-goal scores became monotone %s -- the operators no longer "
                        "rotate; re-derive the horizon semantics" % scores)


# ----------------------------------------------------------------- differential
def test_different_sequences_score_differently():
    """DIFFERENTIAL EFFECT. Goal = the true 2-step successor of [0, 0], so the
    matching sequence must score ~0 while a different sequence does not."""
    ev, truth = _fitted(model_horizon=2)
    s = _unit_state(31)
    goal = _goal(s, truth, [0, 0])
    matched = ev.score_sequence(s, [0, 0], goal)[0]
    other = ev.score_sequence(s, [1, 3], goal)[0]
    assert matched is not None and other is not None
    assert matched < 0.05, "the matching sequence should score near 0, got %.4f" % matched
    assert other > matched + 1e-4, "no separation: matched %.6f vs other %.6f" % (matched, other)


# ------------------------------------------------------------------ abstention
def test_unseen_action_abstains_never_fabricates():
    triples, truth = AK.make_synthetic_triples(2, DIM, 64, 5, rng_scale=0.02)
    ev = KL.KoopmanLeafEvaluator(wave_dim=DIM, n_actions=4, horizon=1)
    ev.fit(triples)
    assert ev.fitted_actions == [0, 1]
    goal = _goal(_unit_state(53), truth, [0])
    v = ev.evaluate(torch.randn(DIM), [2], goal)
    assert v.score is None and v.veto is False and v.valid is False
    assert v.reason.startswith("UNFITTED_ACTION")


def test_horizon_off_is_inert():
    ev, truth = _fitted(model_horizon=0)
    goal = _goal(_unit_state(53), truth, [0])
    v = ev.evaluate(torch.randn(DIM), [0, 1], goal)
    assert v.score is None and v.veto is False and v.reason == "HORIZON_OFF"


def test_empty_actions_is_inert():
    ev, truth = _fitted()
    v = ev.evaluate(torch.randn(DIM), [], _goal(_unit_state(53), truth, [0]))
    assert v.score is None and v.veto is False and v.reason == "NO_ACTIONS"


def test_bad_goal_width_abstains_rather_than_raising():
    ev, truth = _fitted()
    v = ev.evaluate(torch.randn(DIM), [0], torch.randn(DIM + 5))
    assert v.score is None and v.veto is False and v.valid is False
    assert v.reason.startswith("ERROR:")


def test_non_finite_state_abstains():
    ev, truth = _fitted()
    bad = torch.randn(DIM)
    bad[0] = float("nan")
    v = ev.evaluate(bad, [0], _goal(_unit_state(47), truth, [0]))
    assert v.score is None and v.veto is False


# -------------------------------------------------------------------- veto rule
def test_veto_fires_on_gross_disagreement_only():
    """The channel ADDS a veto: it must clear a matching rollout and fire on an
    opposite goal. Measured with the corrected goal: agree 2.9e-05 veto=False;
    disagree 2.0 veto=True."""
    ev, truth = _fitted()
    s = _unit_state(41)
    goal = _goal(s, truth, [0])
    agree = ev.evaluate(s, [0], goal)
    disagree = ev.evaluate(s, [0], -goal)
    assert agree.valid is True and agree.score < 0.05
    assert agree.veto is False
    assert disagree.veto is True
    assert disagree.score > ev.tau_rollout


def test_tau_is_reported_and_honoured():
    triples, truth = AK.make_synthetic_triples(N_A, DIM, 256, 7, rng_scale=0.02)
    s = _unit_state(43)
    goal = _goal(s, truth, [0])
    strict = KL.KoopmanLeafEvaluator(wave_dim=DIM, n_actions=N_A, horizon=1, tau_rollout=0.0)
    strict.fit(triples)
    assert strict.evaluate(s, [0], goal).veto is True, "tau=0 must veto any nonzero score"
    loose = KL.KoopmanLeafEvaluator(wave_dim=DIM, n_actions=N_A, horizon=1, tau_rollout=1.5)
    loose.fit(triples)
    assert loose.evaluate(s, [0], goal).veto is False, "tau=1.5 must never veto"


# -------------------------------------------------------- candidate enumeration
def test_candidate_sequences_depth_one_is_the_singletons():
    """depth=1 must reproduce the planner's existing per-op expansion, so this
    channel is a strict GENERALISATION rather than a new interface."""
    assert KL.candidate_sequences([1, 2, 3], 1) == [(1,), (2,), (3,)]


def test_candidate_sequences_depth_two_is_the_cross_product():
    got = KL.candidate_sequences([1, 2], 2)
    assert got == [(1, 1), (1, 2), (2, 1), (2, 2)]


def test_candidate_sequences_depth_zero_is_empty():
    assert KL.candidate_sequences([1, 2], 0) == []


# ---------------------------------------------------------------- construction
def test_bad_construction_raises():
    with pytest.raises(ValueError):
        KL.KoopmanLeafEvaluator(wave_dim=1, n_actions=2)
    with pytest.raises(ValueError):
        KL.KoopmanLeafEvaluator(wave_dim=8, n_actions=0)


def test_report_is_json_safe_and_states_the_fail_mode():
    import json
    ev, _ = _fitted()
    rep = ev.report()
    json.dumps(rep)
    assert rep["fail_mode"].startswith("FAIL_OPEN")
    assert "linear" in rep["honest_limit"].lower()


def test_module_does_not_select_actions():
    """Boundary: this channel SCORES and VETOES. The planner owns the decision, and a
    grep guard keeps a future edit from smuggling selection in here."""
    src = open(os.path.join(C, "henri_koopman_leaf.py"), encoding="utf-8").read()
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    assert "def select_action" not in code
    assert "argmax" not in code
