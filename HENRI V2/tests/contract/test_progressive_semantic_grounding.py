"""Contract tests for the Phase 8 Progressive Semantic Grounding engine.

Covers: feature gate OFF (no allocation), no-demo fail-closed, functor
compile + goal binding, macro-option construction, vmap-loop EFE agreement,
schema completeness, and the diagnostic-only / no-scorecard invariant.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "HENRI V2"))

import pytest  # noqa: E402
import torch  # noqa: E402

from progressive_semantic_grounding_engine import (  # noqa: E402
    FEATURE_FLAG,
    SCHEMA_ID,
    STATUS_EMPTY,
    STATUS_FALSIFIED,
    STATUS_FEATURE_DISABLED,
    STATUS_NO_DEMOS,
    STATUS_OK,
    MacroOption,
    ProgressiveSemanticGroundingEngine,
    build_macro_options,
    compile_functor_wave,
)


@pytest.fixture(scope="module")
def harness():
    from efe_planner import EFEPlanner
    from henri_vision_encoder import HENRIVisionEncoder
    from darwinian_phase_swarm import HenriSwarmOrchestrator
    from arcengine import GameAction  # production import

    SCALE = dict(num_experts=64, d_model=512, r_rank=8, num_blocks=64)
    device = "cpu"
    torch.manual_seed(20260814)
    orch = HenriSwarmOrchestrator(
        action_enum_class=GameAction,
        constraint_weight_max=5.0,
        constraint_reject_thresh=0.38,
        beta_pragmatic=1.0,
        lambda_goal=0.5,
        learnable_actions=False,
        chimera_mode=False,
        chimera_alpha=1.4,
        chimera_explorer_fraction=0.25,
        happy_tensor_cut=False,
        external_outcome_efe=False,
        external_eig_weight=0.25,
        external_task_weight=1.0,
        task_weighted_eig=False,
        task_eig_gamma=4.0,
        **SCALE,
    ).to(device)
    orch.eval()
    tokenizer = HENRIVisionEncoder(
        d_model=SCALE["d_model"], k_blocks=SCALE["num_blocks"],
        block_dim=8, max_grid_dim=30, device=device)
    eng = ProgressiveSemanticGroundingEngine(
        orch.planner, tokenizer, device=device,
        num_blocks=SCALE["num_blocks"], block_dim=8)
    return orch, tokenizer, eng, SCALE


def make_grid(n=9, step=3, color=1):
    return [[color if (r % step == 0 and c % step == 0) else 0 for c in range(n)]
            for r in range(n)]


# --- feature gate ----------------------------------------------------------
def test_feature_gate_off_no_allocation(harness):
    _, _, eng, _ = harness
    # Ensure flag OFF.
    os.environ.pop(FEATURE_FLAG, None)
    s = eng.status()
    assert s["status"] == STATUS_FEATURE_DISABLED
    assert s["diagnostic_only"] is True and s["score_eligible"] is False
    r = eng.plan(make_grid(), demo_pairs=None, boundary_batch=None)
    assert r["status"] == STATUS_FEATURE_DISABLED
    assert "did not allocate" in r["reason"]


def test_feature_gate_on_requires_demos(harness):
    _, _, eng, _ = harness
    os.environ[FEATURE_FLAG] = "1"
    try:
        r = eng.plan(make_grid(), demo_pairs=None, boundary_batch=None)
        assert r["status"] == STATUS_NO_DEMOS
        assert "never fabricate demos" in r["reason"]
        assert r["score_eligible"] is False
    finally:
        os.environ.pop(FEATURE_FLAG, None)


# --- functor compile --------------------------------------------------------
def test_functor_compile_and_goal_bind(harness):
    _, tokenizer, eng, scale = harness
    os.environ[FEATURE_FLAG] = "1"
    try:
        grid = make_grid()
        grid2 = make_grid(step=4)
        pairs = [(grid, grid2), (grid2, grid)]
        res = eng.compile_task_functor(pairs, task_id="t")
        assert res.demo_pair_count == 2
        assert len(res.w_task_sha256) == 64
        assert len(res.goal_wave_sha256) == 64
        obs = tokenizer.encode_spatial_grid(grid).squeeze(0)
        goal = eng.goal_bind(obs)
        assert tuple(goal.shape) == (scale["num_blocks"], 8)
        assert abs(goal.norm().item() - 1.0) < 1e-3  # unit-normalized
    finally:
        os.environ.pop(FEATURE_FLAG, None)


def test_goal_bind_fails_closed_without_wtask(harness):
    orch, tokenizer, _eng, scale = harness
    os.environ[FEATURE_FLAG] = "1"
    try:
        # Fresh engine: no W_task compiled in this instance.
        fresh = ProgressiveSemanticGroundingEngine(
            orch.planner, tokenizer, device="cpu",
            num_blocks=scale["num_blocks"], block_dim=8)
        obs = tokenizer.encode_spatial_grid(make_grid()).squeeze(0)
        with pytest.raises(RuntimeError, match="BLOCKED_NO_DEMONSTRATIONS"):
            fresh.goal_bind(obs)
    finally:
        os.environ.pop(FEATURE_FLAG, None)


# --- macro-options ----------------------------------------------------------
def test_macro_option_payload_shape():
    opt = MacroOption(object_id=1, kind="translate", dx=1, dy=0,
                      bbox=(0, 0, 3, 3), area=4, color=2)
    p = opt.to_payload(action_id=6)
    assert set(p.keys()) == {"action", "x", "y"}
    assert p["action"] == 6


def test_build_macro_options_bounded():
    class Obj:
        bbox = (0, 0, 2, 2)
        area = 9
        color = 1
        object_id = 0

    opts = build_macro_options([Obj()], (10, 10), max_options=16)
    assert len(opts) <= 16
    kinds = {o.kind for o in opts}
    assert kinds <= {"translate", "rotate", "color"}


# --- vmap-loop agreement ----------------------------------------------------
def test_vmap_loop_agreement(harness):
    _, tokenizer, eng, scale = harness
    os.environ[FEATURE_FLAG] = "1"
    try:
        grid = make_grid()
        pairs = [(grid, make_grid(step=4)), (make_grid(step=4), grid)]
        eng.compile_task_functor(pairs, task_id="t")
        obs = tokenizer.encode_spatial_grid(grid).squeeze(0)
        goal = eng.goal_bind(obs)
        opts = eng.options_from_grid(grid)
        assert opts, "expected macro-options"
        waves, _labels = eng.option_waves(grid, opts)
        bnd = torch.nn.functional.normalize(
            torch.randn(1, scale["num_blocks"], 8), p=2, dim=-1)
        efe_loop = eng.score(obs, waves, bnd, goal_wave=goal)
        efe_bat = eng.score_batched(obs, waves, bnd, goal_wave=goal)
        assert tuple(efe_loop.shape) == (waves.shape[0],)
        agree = float((efe_loop - efe_bat).abs().max().item())
        assert agree <= 1e-6, f"vmap-loop disagreement {agree:.3e}"
    finally:
        os.environ.pop(FEATURE_FLAG, None)


# --- integrated plan --------------------------------------------------------
def test_plan_pipeline_ok(harness):
    _, _, eng, _ = harness
    os.environ[FEATURE_FLAG] = "1"
    try:
        grid = make_grid()
        grid2 = make_grid(step=4)
        r = eng.plan(grid, demo_pairs=[(grid, grid2), (grid2, grid)],
                     boundary_batch=torch.nn.functional.normalize(
                         torch.randn(1, 64, 8), p=2, dim=-1),
                     task_id="t", top_k=3)
        assert r["schema_id"] == SCHEMA_ID
        assert r["score_eligible"] is False and r["diagnostic_only"] is True
        assert r["num_options"] > 0
        if r["status"] == STATUS_OK:
            # Functor passed the held-out recovery gate -> ranked top-k.
            assert len(r["ranked"]) == 3
            for row in r["ranked"]:
                assert "option" in row and "efe" in row and "payload" in row
                assert set(row["payload"].keys()) == {"action", "x", "y"}
            assert r["agreement_max_abs_diff"] is not None
            assert r["agreement_max_abs_diff"] <= 1e-6
            assert r["functor"] is not None
        else:
            # The held-out gate may legitimately falsify on synthetic pairs
            # (identity wins). Fail-closed: no ranked options, typed status.
            assert r["status"] == STATUS_FALSIFIED
            assert r["ranked"] == []
    finally:
        os.environ.pop(FEATURE_FLAG, None)


def test_plan_empty_objects(harness):
    _, _, eng, _ = harness
    os.environ[FEATURE_FLAG] = "1"
    try:
        empty = [[0] * 9 for _ in range(9)]
        # Two pairs so functor compile succeeds; the grid has no objects.
        r = eng.plan(empty, demo_pairs=[(empty, empty), (empty, empty)],
                     boundary_batch=None, task_id="t")
        assert r["status"] == STATUS_EMPTY
        assert r["num_options"] == 0
        assert r["score_eligible"] is False
    finally:
        os.environ.pop(FEATURE_FLAG, None)


def test_plan_functor_failure_fail_closed(harness):
    """Single-pair demos hold out the only pair -> no train pairs -> the
    functor cannot compile; plan() must return a typed failure, not raise."""
    _, _, eng, _ = harness
    os.environ[FEATURE_FLAG] = "1"
    try:
        grid = make_grid()
        r = eng.plan(grid, demo_pairs=[(grid, grid)],
                     boundary_batch=None, task_id="t")
        assert r["status"] in (STATUS_NO_DEMOS, STATUS_FALSIFIED)
        assert r["score_eligible"] is False
    finally:
        os.environ.pop(FEATURE_FLAG, None)
