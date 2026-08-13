"""Contract tests for the Phase 7.9e score-progress causal probe."""

import pytest
import numpy as np
from types import SimpleNamespace

from score_progress_causal_probe import (
    PERMUTATIONS,
    ProbeResult,
    VERDICT_PROGRESS_INFORMATIVE,
    VERDICT_PROGRESS_INERT,
    VERDICT_NO_PROGRESS,
    VERDICT_SCORECARD_UNAVAILABLE,
    VERDICT_INVALID_REPLAY,
    VERDICT_INSUFFICIENT_BRANCHES,
    VERDICT_BUDGET,
    action_cycle,
    compute_horizon_stats,
    frame_delta_nu,
    frame_signature,
    holm_correct,
    mutual_information,
    permutation_test,
    read_levels_completed,
    run_probe,
    verdict_from_stats,
)


# --------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------

def test_action_cycle_deterministic():
    a = ["a", "b", "c"]
    assert action_cycle(a, 0, 4) == ["a", "b", "c", "a"]
    assert action_cycle(a, 2, 3) == ["c", "a", "b"]
    assert action_cycle(a, 0, 0) == []
    assert action_cycle([], 0, 4) == []
    # determinism
    assert action_cycle(a, 5, 7) == action_cycle(a, 5, 7)


def test_frame_signature_deterministic():
    f1 = [[1, 2], [3, 4]]
    f2 = [[1, 2], [3, 4]]
    f3 = [[1, 2], [3, 5]]
    assert frame_signature(f1) == frame_signature(f2)
    assert frame_signature(f1) != frame_signature(f3)


def test_frame_delta_nu_basic():
    b = [[1, 2], [3, 4]]
    assert frame_delta_nu(b, b) == 0
    assert frame_delta_nu(b, [[1, 9], [3, 4]]) == 1
    assert frame_delta_nu(b, [[1, 2, 0], [3, 4, 0]]) == -1  # shape mismatch


def test_holm_correct_standard():
    assert holm_correct([]) == []
    assert holm_correct([0.01, 0.04, 0.2]) == pytest.approx([0.03, 0.08, 0.2])
    assert holm_correct([0.02, 0.03]) == pytest.approx([0.04, 0.04])  # running-max step-down
    # capped at 1.0 and monotone: both adjusted values are 1.0
    assert holm_correct([0.9, 0.5]) == pytest.approx([1.0, 1.0])


def test_mutual_information_dependent_gt_independent():
    rng = np_random = __import__("numpy").random.default_rng(0)
    n = 400
    # dependent: action perfectly predicts outcome
    dep_a = np_random.integers(0, 2, n)
    dep_o = dep_a.copy()
    mi_dep = mutual_information(dep_a, dep_o, 2, 2)
    # independent
    ind_o = np_random.integers(0, 2, n)
    mi_ind = mutual_information(dep_a, ind_o, 2, 2)
    assert mi_dep > mi_ind + 0.05


def test_permutation_inert():
    per_round = {r: {0: False, 1: False, 2: False} for r in range(30)}
    t, p, n_ge = permutation_test(per_round, seed=1)
    assert t == 0.0
    assert p == 1.0
    assert n_ge == PERMUTATIONS  # every null permutation ties T_obs=0


def test_permutation_discordant_consistent_winner():
    per_round = {r: {0: True, 1: False, 2: False} for r in range(30)}
    t, p, n_ge = permutation_test(per_round, seed=1)
    assert t == pytest.approx(1.0)
    assert p < 0.05


def test_permutation_deterministic_seed():
    per_round = {r: {0: (r % 3 == 0), 1: False, 2: False} for r in range(30)}
    assert permutation_test(per_round, seed=7) == permutation_test(per_round, seed=7)


# --------------------------------------------------------------------------
# horizon stats
# --------------------------------------------------------------------------

def _rows(rounds, n_actions, horizon, progress_fn):
    rows = []
    for r in range(rounds):
        for a in range(n_actions):
            prog = bool(progress_fn(r, a))
            rows.append({
                "round": r, "action": a, "action_name": f"a{a}",
                "horizon": horizon, "progress": prog, "win": False,
                "delta_levels": 1 if prog else 0, "delta_cells": 0,
                "levels_branch": 0, "levels_t": 1 if prog else 0,
            })
    return rows


def test_horizon_stats_discordant():
    rows = _rows(40, 3, 16, lambda r, a: a == 0)
    s = compute_horizon_stats(rows, 16, seed=1)
    assert s["valid_branches"] == 40
    assert s["discordant_fraction"] == 1.0
    assert s["mean_rd"] == pytest.approx(1.0)
    assert s["p_rd1"] == pytest.approx(1.0)
    assert s["perm_p_raw"] < 0.05
    assert s["progress_events"] == 40
    assert s["mi_label"] == "derived-secondary"


def test_horizon_stats_inert():
    rows = _rows(40, 3, 4, lambda r, a: False)
    s = compute_horizon_stats(rows, 4, seed=1)
    assert s["discordant_fraction"] == 0.0
    assert s["mean_rd"] == 0.0
    assert s["perm_p_raw"] == 1.0
    assert s["progress_events"] == 0


def test_horizon_delayed_credit():
    # progress appears ONLY at horizon 16 (delayed credit assignment)
    rows = _rows(40, 3, 1, lambda r, a: False)
    rows += _rows(40, 3, 16, lambda r, a: a == 0)
    s1 = compute_horizon_stats(rows, 1, seed=1)
    s16 = compute_horizon_stats(rows, 16, seed=1)
    assert s1["progress_events"] == 0
    assert s1["discordant_fraction"] == 0.0
    assert s16["progress_events"] == 40
    assert s16["discordant_fraction"] == 1.0


def test_horizon_stats_incomplete_round_excluded():
    # a round with a missing action must be excluded from the matched set
    rows = _rows(30, 3, 1, lambda r, a: a == 0)
    rows.append({"round": 99, "action": 0, "action_name": "a0", "horizon": 1,
                 "progress": True, "win": False, "delta_levels": 1,
                 "delta_cells": 0, "levels_branch": 0, "levels_t": 1})
    s = compute_horizon_stats(rows, 1, seed=1)
    assert s["valid_branches"] == 30


# --------------------------------------------------------------------------
# verdicts
# --------------------------------------------------------------------------

def _res(**kw):
    base = dict(env="t", verdict="", reason="", rounds=240, valid_branches=200,
                budget_hit=False, reset_failures=0, replay_mismatches=0,
                env_step_errors=0, scorecard_failures=0, total_steps=0,
                total_resets=0, wall_seconds=0.0)
    base.update(kw)
    return ProbeResult(**base)


def _passing_horizon(events=10, discordant=0.2, rd=0.1, p=0.01):
    return {"valid_branches": 200, "progress_events": events,
            "discordant_fraction": discordant, "mean_rd": rd, "p_rd1": 0.0,
            "perm_t_obs": 0.1, "perm_p_raw": p, "perm_n_ge": 1,
            "mi_nats": 0.1, "mi_label": "derived-secondary"}


def test_verdict_informative():
    res = _res()
    ph = {"1": _passing_horizon(), "4": _passing_horizon(), "16": _passing_horizon()}
    v, reason = verdict_from_stats(res, ph)
    assert v == VERDICT_PROGRESS_INFORMATIVE


def test_verdict_inert():
    res = _res()
    ph = {"1": _passing_horizon(), "4": _passing_horizon(discordant=0.02),
          "16": _passing_horizon()}
    v, reason = verdict_from_stats(res, ph)
    assert v == VERDICT_PROGRESS_INERT


def test_verdict_no_progress():
    res = _res()
    ph = {"1": _passing_horizon(events=0), "4": _passing_horizon(events=0),
          "16": _passing_horizon(events=0)}
    v, reason = verdict_from_stats(res, ph)
    assert v == VERDICT_NO_PROGRESS


def test_verdict_invalid_replay():
    res = _res(replay_mismatches=1)
    ph = {"1": _passing_horizon(), "4": _passing_horizon(), "16": _passing_horizon()}
    v, reason = verdict_from_stats(res, ph)
    assert v == VERDICT_INVALID_REPLAY


def test_verdict_scorecard_unavailable():
    res = _res(scorecard_failures=2)
    ph = {"1": _passing_horizon(), "4": _passing_horizon(), "16": _passing_horizon()}
    v, reason = verdict_from_stats(res, ph)
    assert v == VERDICT_SCORECARD_UNAVAILABLE


def test_verdict_insufficient_branches():
    res = _res(valid_branches=10)
    ph = {"1": _passing_horizon(events=5), "4": _passing_horizon(events=5),
          "16": _passing_horizon(events=5)}
    v, reason = verdict_from_stats(res, ph)
    assert v == VERDICT_INSUFFICIENT_BRANCHES


def test_verdict_budget():
    res = _res(budget_hit=True)
    ph = {"1": _passing_horizon(), "4": _passing_horizon(), "16": _passing_horizon()}
    v, reason = verdict_from_stats(res, ph)
    assert v == VERDICT_BUDGET


# --------------------------------------------------------------------------
# scorecard read (fail-closed)
# --------------------------------------------------------------------------

class _NoScorecardGame:
    scorecard_id = None


class _RaisingArcade:
    def get_scorecard(self, scid):
        raise RuntimeError("scorecard down")


class _EmptyArcade:
    def get_scorecard(self, scid):
        return SimpleNamespace(environments=[])


class _GoodArcade:
    def get_scorecard(self, scid):
        return SimpleNamespace(environments=[
            SimpleNamespace(levels_completed=3),
            SimpleNamespace(levels_completed=None),
        ])


def test_read_levels_completed_fail_closed():
    assert read_levels_completed(_NoScorecardGame(), _GoodArcade())[0] is None
    assert read_levels_completed(_NoScorecardGame(), _GoodArcade())[1] == "NO_SCORECARD_ID"
    assert read_levels_completed(SimpleNamespace(scorecard_id="x"),
                                 _RaisingArcade())[1] == "SCORECARD_READ_EXCEPTION"
    levels, status = read_levels_completed(SimpleNamespace(scorecard_id="x"),
                                           _EmptyArcade())
    assert levels is None and status == "SCORECARD_DELTA_UNAVAILABLE"
    levels, status = read_levels_completed(SimpleNamespace(scorecard_id="x"),
                                           _GoodArcade())
    assert levels == 3 and status == "SCORECARD_DELTA_OK"


# --------------------------------------------------------------------------
# end-to-end fake environment (collector plumbing)
# --------------------------------------------------------------------------

class _FakeObs:
    def __init__(self, level):
        self.frame = np.array([[level, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.status = "WIN" if level >= 10 else None


class _FakeAction:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<{self.name}>"


class _FakeGame:
    def __init__(self, advance_name="b", advance_names=None):
        self.action_space = [_FakeAction("a"), _FakeAction(advance_name),
                             _FakeAction("c")]
        self.scorecard_id = "sc_fake"
        self.level = 0
        self.advance_name = advance_name
        self.advance_names = advance_names if advance_names is not None else {advance_name}

    def reset(self):
        self.level = 0
        return _FakeObs(self.level)

    def step(self, action):
        if action.name in self.advance_names:
            self.level += 1
        return _FakeObs(self.level)


class _FakeArcade:
    def __init__(self, game, fail=False):
        self.game = game
        self.fail = fail

    def get_scorecard(self, scid):
        if self.fail:
            raise RuntimeError("scorecard down")
        return SimpleNamespace(environments=[
            SimpleNamespace(levels_completed=self.game.level)])


def test_run_probe_fake_informative(monkeypatch):
    import score_progress_causal_probe as sp
    monkeypatch.setattr(sp, "HORIZONS", (1,))
    monkeypatch.setattr(sp, "H_MAX", 1)
    game = _FakeGame()
    arcade = _FakeArcade(game)
    res = sp.run_probe(game, arcade, "fake", rounds=210, seed=1,
                       payloads=False, budget_sec=3600.0)
    assert res.verdict == VERDICT_PROGRESS_INFORMATIVE
    assert res.replay_mismatches == 0
    assert res.env_step_errors == 0
    assert res.reset_failures == 0
    assert res.scorecard_failures == 0
    assert res.valid_branches >= 200
    assert res.distinct_initial_states == 1
    h1 = res.per_horizon["1"]
    assert h1["discordant_fraction"] == 1.0
    assert h1["mean_rd"] == pytest.approx(1.0)
    assert h1["perm_p_raw"] < 0.05
    assert h1["progress_events"] >= 1
    assert res.per_action["b"]["progress_rate"] == pytest.approx(1.0)
    assert res.per_action["a"]["progress_rate"] == pytest.approx(0.0)


def test_run_probe_fake_no_progress(monkeypatch):
    import score_progress_causal_probe as sp
    monkeypatch.setattr(sp, "HORIZONS", (1,))
    monkeypatch.setattr(sp, "H_MAX", 1)
    game = _FakeGame(advance_names=set())  # no action advances -> zero progress
    arcade = _FakeArcade(game)
    res = sp.run_probe(game, arcade, "fake", rounds=210, seed=1,
                       payloads=False, budget_sec=3600.0)
    assert res.verdict == VERDICT_NO_PROGRESS
    assert res.replay_mismatches == 0
    assert res.env_step_errors == 0
    assert res.scorecard_failures == 0
    assert res.valid_branches >= 200


def test_run_probe_fake_scorecard_blocked(monkeypatch):
    import score_progress_causal_probe as sp
    monkeypatch.setattr(sp, "HORIZONS", (1,))
    monkeypatch.setattr(sp, "H_MAX", 1)
    game = _FakeGame()
    arcade = _FakeArcade(game, fail=True)
    res = sp.run_probe(game, arcade, "fake", rounds=30, seed=1,
                       payloads=False, budget_sec=3600.0)
    assert res.verdict == VERDICT_SCORECARD_UNAVAILABLE
    assert res.scorecard_failures >= 1
