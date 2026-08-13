"""Contract tests for the Phase 7.9f curriculum replay driver.

Coverage (pre-registered, 2026-08-13):
1. deterministic_split: 16-env universe -> 12 discovery + 4 held-out,
   stable under salt, disjoint, no env lost.
2. curriculum_compressibility: bounded [0,1]; random wave < constant wave;
   degenerate (constant) wave -> ~1.0 (top-10% energy concentration).
3. frame_delta_nu: counts changed cells; shape mismatch -> 0.
4. _scorecard_increased: strict increase only; None -> False.
5. decide_verdict: every pre-registered verdict reached from counters;
   zero-progress healthy -> BLOCKED_NO_PROGRESS_EVENTS;
   scorecard-unavailable -> INVALID_SCORECARD_SEMANTICS;
   step errors -> INVALID_PLUMBING; infra -> BLOCKED_INFRASTRUCTURE;
   events + branches -> PASS_PROGRESS_ENV_FOUND; sparse -> INCONCLUSIVE.
6. sans_buffer_status: active only >= 50 rows, >= 2 distinct labels,
   >= 1 contributing env; otherwise inactive fail-closed.
7. build_aggregate: reducer-derived aggregate (never overwrites per-env);
   reports missing envs; complete flag.
8. run_env_replay: zero-progress env -> verdict BLOCKED_NO_PROGRESS_EVENTS
   with harness counters healthy (uses a stub game/arcade/policy).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from arc_curriculum_replay import (
    UNIVERSE,
    VERDICT_INFRASTRUCTURE,
    VERDICT_NO_PROGRESS,
    VERDICT_PLUMBING,
    VERDICT_PROGRESS_FOUND,
    VERDICT_SCORECARD_INVALID,
    VERDICT_SETUP,
    VERDICT_SPARSE,
    EnvCounters,
    _scorecard_increased,
    build_aggregate,
    curriculum_compressibility,
    decide_verdict,
    deterministic_split,
    frame_delta_nu,
    sans_buffer_status,
)


# ---------------------------------------------------------------------------
# 1. Split
# ---------------------------------------------------------------------------


class TestSplit:
    def test_universe_is_16(self):
        assert len(UNIVERSE) == 16
        assert len(set(UNIVERSE)) == 16

    def test_split_disjoint_and_complete(self):
        disc, held = deterministic_split(UNIVERSE, "p79f-split-v1", 12)
        assert len(disc) == 12
        assert len(held) == 4
        assert set(disc) | set(held) == set(UNIVERSE)
        assert set(disc) & set(held) == set()

    def test_split_deterministic(self):
        d1, h1 = deterministic_split(UNIVERSE, "p79f-split-v1", 12)
        d2, h2 = deterministic_split(UNIVERSE, "p79f-split-v1", 12)
        assert d1 == d2 and h1 == h2

    def test_split_recorded_values(self):
        # Sealed at pre-registration time (2026-08-13).
        disc, held = deterministic_split(UNIVERSE, "p79f-split-v1", 12)
        assert disc == ["cn04", "ka59", "g50t", "sb26", "ar25", "lp85",
                        "dc22", "m0r0", "bp35", "ls20", "re86", "ft09"]
        assert held == ["s5i5", "r11l", "cd82", "lf52"]


# ---------------------------------------------------------------------------
# 2. Compressibility
# ---------------------------------------------------------------------------


class TestCompressibility:
    def test_bounded(self):
        w = torch.randn(4096)
        k = curriculum_compressibility(w)
        assert 0.0 <= k <= 1.0

    def test_constant_wave_high(self):
        w = torch.ones(4096)
        k = curriculum_compressibility(w)
        # A constant wave concentrates all energy in the DC bin.
        assert k > 0.9

    def test_random_lower_than_constant(self):
        torch.manual_seed(0)
        rnd = curriculum_compressibility(torch.randn(4096))
        const = curriculum_compressibility(torch.ones(4096))
        assert rnd < const


# ---------------------------------------------------------------------------
# 3. Frame delta
# ---------------------------------------------------------------------------


class TestFrameDelta:
    def test_counts_changed_cells(self):
        before = [[1, 1], [2, 2]]
        after = [[1, 3], [2, 2]]
        assert frame_delta_nu(before, after) == 1

    def test_shape_mismatch_zero(self):
        assert frame_delta_nu([[1]], [[1], [2]]) == 0


# ---------------------------------------------------------------------------
# 4. Strict scorecard increase
# ---------------------------------------------------------------------------


class TestScorecardIncreased:
    def test_strict_increase(self):
        assert _scorecard_increased(3, 2) is True
        assert _scorecard_increased(2, 2) is False
        assert _scorecard_increased(1, 2) is False

    def test_none_false(self):
        assert _scorecard_increased(None, 2) is False


# ---------------------------------------------------------------------------
# 5. Verdicts
# ---------------------------------------------------------------------------


def _counters(**kw) -> EnvCounters:
    base = dict(
        resets=60, reset_failures=0, replay_mismatches=0, env_step_errors=0,
        egress_failures=0, scorecard_failures=0, total_steps=960,
        valid_branches=60, scorecard_events=0, scorecard_delta_sum=0,
        frame_rows=0, progress_branches=0, progress_rows=0, explored_steps=0,
    )
    base.update(kw)
    return EnvCounters(**base)


class TestVerdicts:
    def test_no_progress_healthy(self):
        v, _ = decide_verdict(_counters(), 60)
        assert v == VERDICT_NO_PROGRESS

    def test_scorecard_invalid(self):
        v, _ = decide_verdict(_counters(scorecard_failures=3), 60)
        assert v == VERDICT_SCORECARD_INVALID

    def test_infrastructure(self):
        v, _ = decide_verdict(_counters(reset_failures=40), 60)
        assert v == VERDICT_INFRASTRUCTURE

    def test_plumbing(self):
        v, _ = decide_verdict(_counters(env_step_errors=5), 60)
        assert v == VERDICT_PLUMBING

    def test_progress_found(self):
        v, _ = decide_verdict(_counters(scorecard_events=3,
                                        progress_branches=2), 60)
        assert v == VERDICT_PROGRESS_FOUND

    def test_sparse_events(self):
        v, _ = decide_verdict(_counters(scorecard_events=1,
                                        valid_branches=20), 60)
        assert v == VERDICT_SPARSE


# ---------------------------------------------------------------------------
# 6. SANS buffer gate
# ---------------------------------------------------------------------------


class TestSansBuffer:
    def test_inactive_empty(self):
        st = sans_buffer_status([])
        assert st["buffer_active"] is False
        assert st["status"] == "BLOCKED_SANS_BUFFER_INSUFFICIENT"

    def test_active_at_floor(self):
        rows = [{"action": f"ACTION{1 + (i % 3)}", "delta_nu": 2,
                 "horizon": 4, "scorecard_delta": 1} for i in range(50)]
        payloads = [{"env": "cn04", "verdict": VERDICT_PROGRESS_FOUND,
                     "sans_rows": rows}]
        st = sans_buffer_status(payloads)
        assert st["buffer_active"] is True
        assert st["rows"] == 50
        assert st["distinct_labels"] == 3

    def test_inactive_too_few(self):
        rows = [{"action": "ACTION1", "delta_nu": 2} for _ in range(10)]
        payloads = [{"env": "cn04", "verdict": VERDICT_PROGRESS_FOUND,
                     "sans_rows": rows}]
        st = sans_buffer_status(payloads)
        assert st["buffer_active"] is False


# ---------------------------------------------------------------------------
# 7. Aggregate reducer (immutable per-env, reducer-derived)
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_reducer_reports_missing(self, tmp_path: Path):
        (tmp_path / "cn04.json").write_text(
            json.dumps({"env": "cn04", "x": 1}), encoding="utf-8")
        agg = build_aggregate([tmp_path / "cn04.json"], ["cn04", "ka59"])
        assert agg["complete"] is False
        assert agg["missing_envs"] == ["ka59"]
        assert agg["env_count"] == 1


# ---------------------------------------------------------------------------
# 8. run_env_replay zero-progress env (stub harness)
# ---------------------------------------------------------------------------


class _StubFrame:
    def __init__(self, grid):
        self.frame = [grid]


class _StubObs:
    def __init__(self, grid):
        self.frame = [np.asarray(grid, dtype=np.int64)]
        self.state = type("S", (), {"name": "NOT_FINISHED"})()


class _StubGame:
    def __init__(self, grid, actions):
        self._grid = grid
        self.action_space = actions
        self.scorecard_id = "stub-sc"

    def reset(self):
        return _StubObs(self._grid)

    def step(self, action):
        return _StubObs(self._grid)


class _StubArcade:
    def get_scorecard(self, _sid):
        env_score = type("ES", (), {"levels_completed": 0})()
        return type("SC", (), {"environments": [env_score]})()


class _StubPolicy:
    def __init__(self):
        self.prev_wave = None
        self.camera = None
        self.payloads = False

    def step(self, game, grid):
        return _StubObs(grid), {
            "ok": True, "error": None, "action": game.action_space[0],
            "action_name": getattr(game.action_space[0], "name", "ACTION1"),
            "efe": 0.0, "spread": 0.0, "explored": False, "delta_nu": 0,
        }


class TestRunEnvReplay:
    def test_zero_progress_no_events(self, tmp_path: Path):
        from arc_curriculum_replay import run_env_replay
        grid = [[0] * 8 for _ in range(8)]
        actions = [type("A", (), {"name": f"ACTION{i}"})()
                   for i in range(1, 7)]
        game = _StubGame(grid, actions)
        arcade = _StubArcade()
        policy = _StubPolicy()
        counters, payload = run_env_replay(
            game, arcade, "cn04", rounds=45, seed=20260813, policy=policy,
            out_dir=tmp_path, budget_sec=60.0)
        assert payload["verdict"] == VERDICT_NO_PROGRESS
        assert payload["counters"]["scorecard_events"] == 0
        assert payload["counters"]["env_step_errors"] == 0
        # Deterministic resets -> no mismatches, no infra blame.
        assert payload["counters"]["replay_mismatches"] == 0
        assert payload["counters"]["reset_failures"] == 0
