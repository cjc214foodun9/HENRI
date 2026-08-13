"""Contract tests for the Phase 7.9f curriculum replay driver.

Coverage (pre-registered, 2026-08-13; extended 2026-08-13 for Reference 3
contract fixes):

1. deterministic_split: 16-env universe -> 12 discovery + 4 held-out,
   stable under salt, disjoint, no env lost.
2. curriculum_compressibility: bounded [0,1]; random < constant.
3. frame_delta_nu: counts changed cells; shape mismatch -> 0.
4. _scorecard_increased: strict increase only; None -> False.
5. decide_verdict: every pre-registered verdict reachable.
6. sans_buffer_status: discovery-only gate; held-out rows NEVER count;
   active only >= 50 rows, >= 2 distinct labels, >= 1 contributing env.
7. build_aggregate: reducer-derived from immutable per-env artifacts.
8. run_env_replay zero-progress: BLOCKED_NO_PROGRESS_EVENTS, harness healthy.
9. Matched first-action counterfactuals:
   - every legal first action gets an identical baseline branch
     (matched_branches == n_actions, zero unmatched/mismatch);
   - branch order does not alter pairing (deterministic rerun identical);
   - non-reproducible reset fails closed (BLOCKED_INFRASTRUCTURE);
   - the frozen continuation runs to H_MAX after EVERY candidate
     (total_steps == anchor_prefix + n_actions*(prefix+candidate+H_MAX));
   - candidate-specific first actions are actually stepped.
10. Trainable SANS rows:
    - hidden feature uses the PRODUCTION unbinder_hidden path (real
      unbinder_hidden + flatten_uwe over a fake transducer/tokenizer);
    - only strict levels_completed increases emit rows;
    - row schema, hidden shape/dtype/hash/provenance validated;
    - zero progress emits no trainable buffer and no manifest;
    - a missing/invalid hidden feature blocks the row (blocked_rows +
      NO_HIDDEN_FEATURE), never admits a fake row.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

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
    dataset_digest,
    decide_verdict,
    deterministic_split,
    frame_delta_nu,
    hidden_sha256,
    sans_buffer_status,
    validate_hidden,
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
        k = curriculum_compressibility(torch.ones(4096))
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
        valid_branches=60, matched_branches=0, unmatched_branches=0,
        hidden_failures=0, scorecard_events=0, scorecard_delta_sum=0,
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
# 6. SANS buffer gate (discovery-only, split-filtered)
# ---------------------------------------------------------------------------


class TestSansBuffer:
    def test_inactive_empty(self):
        st = sans_buffer_status([])
        assert st["buffer_active"] is False
        assert st["status"] == "BLOCKED_SANS_BUFFER_INSUFFICIENT"

    def test_active_at_floor(self):
        rows = [{"action": f"ACTION{1 + (i % 3)}", "env": "cn04",
                 "split": "discovery", "delta_nu": 2,
                 "horizon": 4, "scorecard_delta": 1} for i in range(50)]
        payloads = [{"env": "cn04", "verdict": VERDICT_PROGRESS_FOUND,
                     "split": "discovery", "sans_rows": rows}]
        st = sans_buffer_status(payloads)
        assert st["buffer_active"] is True
        assert st["rows"] == 50
        assert st["distinct_labels"] == 3
        assert st["contributing_envs"] == 1

    def test_inactive_too_few(self):
        rows = [{"action": "ACTION1", "env": "cn04", "split": "discovery",
                 "delta_nu": 2} for _ in range(10)]
        payloads = [{"env": "cn04", "verdict": VERDICT_PROGRESS_FOUND,
                     "split": "discovery", "sans_rows": rows}]
        st = sans_buffer_status(payloads)
        assert st["buffer_active"] is False

    def test_heldout_rows_never_enter_discovery_gate(self):
        disc = [{"action": f"ACTION{1 + (i % 2)}", "env": "cn04",
                 "split": "discovery", "delta_nu": 2, "horizon": 4,
                 "scorecard_delta": 1} for i in range(60)]
        held = [{"action": "ACTION1", "env": "s5i5", "split": "heldout",
                 "delta_nu": 2, "horizon": 4, "scorecard_delta": 1}
                for _ in range(60)]
        payloads = [
            {"env": "cn04", "verdict": VERDICT_PROGRESS_FOUND,
             "split": "discovery", "sans_rows": disc},
            {"env": "s5i5", "verdict": VERDICT_PROGRESS_FOUND,
             "split": "heldout", "sans_rows": held},
        ]
        st = sans_buffer_status(payloads)
        assert st["rows"] == 60          # held-out excluded
        assert st["contributing_envs"] == 1
        assert st["buffer_active"] is True  # 60 rows, 2 labels, 1 env


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
# 8-10. Stub harness + run_env_replay contracts
# ---------------------------------------------------------------------------


class _StubFrame:
    def __init__(self, grid):
        self.frame = [grid]


class _StubObs:
    def __init__(self, grid):
        self.frame = [np.asarray(grid, dtype=np.int64)]
        self.state = type("S", (), {"name": "NOT_FINISHED"})()


def _actions(n=6):
    return [type("A", (), {"name": f"ACTION{i}"})() for i in range(1, n + 1)]


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
    """Frozen policy for the zero-progress stub harness."""

    def __init__(self):
        self.prev_wave = None
        self.camera = None
        self.payloads = False
        self.hidden_dim = None
        self.egress = None

    def step(self, game, grid):
        return _StubObs(grid), {
            "ok": True, "error": None, "action": game.action_space[0],
            "action_name": getattr(game.action_space[0], "name", "ACTION1"),
            "efe": 0.0, "spread": 0.0, "explored": False, "delta_nu": 0,
        }

    def hidden_feature(self, grid):
        return None, "NO_HIDDEN_FEATURE"


class TestRunEnvReplay:
    def test_zero_progress_no_events(self, tmp_path: Path):
        from arc_curriculum_replay import run_env_replay
        grid = [[0] * 8 for _ in range(8)]
        actions = _actions()
        game = _StubGame(grid, actions)
        arcade = _StubArcade()
        policy = _StubPolicy()
        counters, payload = run_env_replay(
            game, arcade, "cn04", rounds=45, seed=20260813, policy=policy,
            out_dir=tmp_path, budget_sec=120.0)
        assert payload["verdict"] == VERDICT_NO_PROGRESS
        assert payload["counters"]["scorecard_events"] == 0
        assert payload["counters"]["env_step_errors"] == 0
        assert payload["counters"]["replay_mismatches"] == 0
        assert payload["counters"]["reset_failures"] == 0
        # No trainable rows and no manifest for a zero-progress env.
        assert payload["sans_rows"] == []
        assert not (tmp_path / "cn04_sans" / "manifest.json").exists()

    def test_all_first_actions_get_identical_baseline(self, tmp_path: Path):
        """Matched counterfactual: every legal first action is stepped from
        the same verified branch state; the frozen continuation runs to
        H_MAX after EVERY candidate."""
        from arc_curriculum_replay import H_MAX, PREFIX_LEN, run_env_replay
        grid = [[0] * 8 for _ in range(8)]
        actions = _actions()
        game = _StubGame(grid, actions)
        counters, payload = run_env_replay(
            game, _StubArcade(), "cn04", rounds=1, seed=20260813,
            policy=_StubPolicy(), out_dir=tmp_path, budget_sec=120.0)
        n = len(actions)
        # 6 candidate branches all matched (identical baseline).
        assert payload["counters"]["matched_branches"] == n
        assert payload["counters"]["unmatched_branches"] == 0
        assert payload["counters"]["replay_mismatches"] == 0
        # Anchor prefix + per-branch (prefix + candidate + H_MAX continuation).
        expected_steps = PREFIX_LEN + n * (PREFIX_LEN + 1 + H_MAX)
        assert payload["counters"]["total_steps"] == expected_steps

    def test_branch_order_does_not_alter_pairing(self, tmp_path: Path):
        from arc_curriculum_replay import run_env_replay
        grid = [[0] * 8 for _ in range(8)]
        actions = _actions()
        outs = []
        for _ in range(2):
            counters, payload = run_env_replay(
                _StubGame(grid, actions), _StubArcade(), "cn04", rounds=3,
                seed=20260813, policy=_StubPolicy(), out_dir=tmp_path,
                budget_sec=120.0)
            outs.append((counters, payload))
        c1, p1 = outs[0]
        c2, p2 = outs[1]
        assert c1.matched_branches == c2.matched_branches
        assert p1["counters"] == p2["counters"]

    def test_non_reproducible_reset_fails_closed(self, tmp_path: Path):
        """A reset that cannot reproduce the baseline frame hash must fail
        closed (BLOCKED_INFRASTRUCTURE), never silently compare unmatched
        branches."""
        from arc_curriculum_replay import run_env_replay

        class _UnstableGame(_StubGame):
            def __init__(self):
                super().__init__([[0] * 8 for _ in range(8)], _actions())
                self._n = 0

            def reset(self):
                self._n += 1
                return _StubObs([[self._n % 3] * 8 for _ in range(8)])

        counters, payload = run_env_replay(
            _UnstableGame(), _StubArcade(), "cn04", rounds=8, seed=20260813,
            policy=_StubPolicy(), out_dir=tmp_path, budget_sec=120.0)
        assert payload["verdict"] == VERDICT_INFRASTRUCTURE
        assert payload["counters"]["replay_mismatches"] >= 8


# ---------------------------------------------------------------------------
# 9. Trainable SANS rows (Reference 3: exact production hidden path)
# ---------------------------------------------------------------------------


class _ProgressGame(_StubGame):
    """Grid stays identical under prefix replay; step changes 2 cells."""

    def __init__(self):
        super().__init__([[0] * 8 for _ in range(8)], _actions())
        self.scorecard_id = "prog-sc"
        self._steps_since_reset = 0

    def reset(self):
        self._steps_since_reset = 0
        return _StubObs(self._grid)

    def step(self, action):
        self._steps_since_reset += 1
        g = [row[:] for row in self._grid]
        g[0][0] = 1
        g[0][1] = 1
        return _StubObs(g)


class _ProgressArcade:
    def __init__(self, game):
        self._game = game

    def get_scorecard(self, _sid):
        levels = 1 if self._game._steps_since_reset >= 6 else 0
        env_score = type("ES", (), {"levels_completed": levels})()
        return type("SC", (), {"environments": [env_score]})()


class _ProgressPolicy(_StubPolicy):
    """Steps the REAL game (advances its step counter) and reports the
    resulting changed frame (delta_nu=2) plus a valid hidden feature."""

    def __init__(self):
        super().__init__()
        self.hidden_dim = 8

    def step(self, game, grid):
        obs = game.step(game.action_space[0])  # advances _steps_since_reset
        return obs, {
            "ok": True, "error": None, "action": game.action_space[0],
            "action_name": getattr(game.action_space[0], "name", "ACTION1"),
            "efe": 0.0, "spread": 0.0, "explored": False, "delta_nu": 2,
        }

    def hidden_feature(self, grid):
        return torch.ones(8, dtype=torch.float32), ""


class _NoHiddenPolicy(_ProgressPolicy):
    def hidden_feature(self, grid):
        return None, "NO_HIDDEN_FEATURE"


class TestSansRows:
    def test_progress_emits_trainable_rows_and_manifest(self, tmp_path: Path):
        from arc_curriculum_replay import run_env_replay
        game = _ProgressGame()
        counters, payload = run_env_replay(
            game, _ProgressArcade(game), "cn04", rounds=1, seed=20260813,
            policy=_ProgressPolicy(), out_dir=tmp_path, budget_sec=120.0,
            env_id="cn04-0000000", split="discovery")
        assert payload["counters"]["scorecard_events"] >= 6
        rows = payload["sans_rows"]
        assert len(rows) == 6  # one row per matched branch (6 first actions)
        for row in rows:
            assert row["schema_id"] == "henri.sans-row.v1"
            assert row["split"] == "discovery"
            assert row["env_id"] == "cn04-0000000"
            assert row["action_index"] == 0
            assert row["scorecard_delta"] == 1
            assert len(row["hidden_sha256"]) == 64
            assert row["hidden_shape"] == [8]
            assert row["hidden_dtype"] == "torch.float32"
            assert row["candidate_action"] in {"ACTION1", "ACTION2",
                                               "ACTION3", "ACTION4",
                                               "ACTION5", "ACTION6"}
        # Lossless feature artifacts + immutable manifest.
        sans_dir = tmp_path / "cn04_sans"
        manifest = json.loads(
            (sans_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["row_count"] == 6
        assert manifest["env"] == "cn04"
        assert manifest["split"] == "discovery"
        assert manifest["hidden_dim"] == 8
        assert len(manifest["dataset_digest"]) == 64
        for row in rows:
            assert (sans_dir / row["feature_file"]).exists()

    def test_zero_progress_emits_no_trainable_buffer(self, tmp_path: Path):
        from arc_curriculum_replay import run_env_replay
        game = _StubGame([[0] * 8 for _ in range(8)], _actions())
        counters, payload = run_env_replay(
            game, _StubArcade(), "cn04", rounds=1, seed=20260813,
            policy=_StubPolicy(), out_dir=tmp_path, budget_sec=120.0)
        assert payload["sans_rows"] == []
        assert payload["blocked_rows"] == []
        assert not (tmp_path / "cn04_sans" / "manifest.json").exists()

    def test_missing_hidden_feature_blocks_row(self, tmp_path: Path):
        from arc_curriculum_replay import run_env_replay
        game = _ProgressGame()
        counters, payload = run_env_replay(
            game, _ProgressArcade(game), "cn04", rounds=1, seed=20260813,
            policy=_NoHiddenPolicy(), out_dir=tmp_path, budget_sec=120.0)
        assert payload["sans_rows"] == []  # no fake row admitted
        assert len(payload["blocked_rows"]) >= 1
        assert all("NO_HIDDEN_FEATURE" in b["reason"]
                   for b in payload["blocked_rows"])
        assert payload["counters"]["hidden_failures"] >= 1
        assert not (tmp_path / "cn04_sans" / "manifest.json").exists()

    def test_hidden_feature_uses_production_unbinder_path(self):
        """hidden_feature must reproduce the run_sans_play extraction:
        real unbinder_hidden + flatten_uwe over encode_spatial_grid output."""
        from arc_curriculum_replay import EFEPlayPolicy

        class _FakeUnbinder:
            def __init__(self):
                self.down_proj = nn.Linear(16, 4, bias=False)
                self.layer_norm = nn.LayerNorm(4)
                self.act = nn.GELU()

        class _FakeTransducer:
            def __init__(self):
                self.unbinder = _FakeUnbinder()
                self.d_model = 16
                self.hidden_dim = 4

        class _FakeTokenizer:
            def encode_spatial_grid(self, grid):
                return torch.ones(1, 2, 8)  # [B, num_blocks, 8]

        policy = EFEPlayPolicy(
            orch=None, tokenizer=_FakeTokenizer(), egress=_FakeTransducer(),
            device="cpu", payloads=False, camera=None, seed=0)
        h, err = policy.hidden_feature([[0] * 8 for _ in range(8)])
        assert err == ""
        assert h.shape == (4,)
        assert h.dtype == torch.float32
        assert validate_hidden(h, 4) == ""

    def test_no_egress_means_no_hidden_feature(self):
        from arc_curriculum_replay import EFEPlayPolicy
        policy = EFEPlayPolicy(
            orch=None, tokenizer=None, egress=None, device="cpu",
            payloads=False, camera=None, seed=0)
        h, err = policy.hidden_feature([[0]])
        assert h is None
        assert err == "NO_HIDDEN_FEATURE"


# ---------------------------------------------------------------------------
# 10. Hidden feature validation + digests
# ---------------------------------------------------------------------------


class TestHiddenValidation:
    def test_valid(self):
        assert validate_hidden(torch.ones(8, dtype=torch.float32), 8) == ""

    def test_not_tensor(self):
        assert validate_hidden(None) == "NOT_A_TENSOR"
        assert validate_hidden([1.0, 2.0]) == "NOT_A_TENSOR"

    def test_wrong_dtype(self):
        assert validate_hidden(torch.ones(8, dtype=torch.float64),
                               8).startswith("DTYPE")

    def test_wrong_rank(self):
        assert validate_hidden(torch.ones(2, 4)).startswith("RANK")

    def test_wrong_dim(self):
        assert validate_hidden(torch.ones(4, dtype=torch.float32),
                               8).startswith("SHAPE")

    def test_non_finite(self):
        bad = torch.ones(8, dtype=torch.float32) * float("nan")
        assert validate_hidden(bad, 8) == "NON_FINITE"

    def test_hashes_deterministic_and_distinct(self):
        a = torch.randn(16, dtype=torch.float32)
        b = torch.randn(16, dtype=torch.float32)
        assert hidden_sha256(a) == hidden_sha256(a.clone())
        assert hidden_sha256(a) != hidden_sha256(b)
        assert len(hidden_sha256(a)) == 64
        # Single-element dataset digest equals the row digest convention.
        assert dataset_digest([a]) == hidden_sha256(a)
