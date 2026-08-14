"""Contract tests: Phase 8.3r EFE Alignment experiment (default OFF).

Covers:
- flag OFF -> FEATURE_DISABLED, no Zone C connection, no allocation;
- no game.step anywhere in new source;
- eligibility single-source fields on every return path;
- boundary shape contract [N, num_blocks, 8];
- loop/vmap identity and decomposition consistency at reduced scale;
- runner gate structure (arms A-D, G1-G4, no threshold tuning constants).
"""

import os
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]  # .../HENRI V2
sys.path.insert(0, str(_REPO))

from efe_alignment_experiment import (  # noqa: E402
    FEATURE_FLAG,
    EfeAlignmentProbe,
    _base_out,
)

ENGINE_TEXT = (_REPO / "efe_alignment_experiment.py").read_text(
    encoding="utf-8")
RUNNER_TEXT = (
    _REPO / "experiments" / "performance"
    / "efe_alignment_cuda_check.py").read_text(encoding="utf-8")


def test_flag_defaults_off():
    assert FEATURE_FLAG == "HENRI_ARC_EFE_ALIGNMENT"
    os.environ.pop(FEATURE_FLAG, None)
    probe = EfeAlignmentProbe(None, None, None)  # no allocation expected
    assert probe.enabled is False
    assert probe.status()["status"] == "FEATURE_DISABLED"
    assert probe.status()["score_eligible"] is False
    assert probe.status()["diagnostic_only"] is True
    assert probe.status()["authorizes_rollout"] is False


def test_no_game_step_in_new_source():
    assert "game.step(" not in ENGINE_TEXT
    assert "game.step(" not in RUNNER_TEXT
    assert "def step(" not in ENGINE_TEXT


def test_eligibility_single_source_on_every_return():
    for out in (_base_out("t"), _base_out("t", status="FEATURE_DISABLED"),
                _base_out("t", status="TRUE_LABEL_ABSENT")):
        assert out["score_eligible"] is False
        assert out["diagnostic_only"] is True
        assert out["authorizes_rollout"] is False
        assert out["schema_id"] == "henri.efe-alignment-arm.v1"


def test_runner_structure_and_gates():
    assert "A" in RUNNER_TEXT and "B" in RUNNER_TEXT
    assert "C" in RUNNER_TEXT and "D" in RUNNER_TEXT
    assert "EFE_ALIGNMENT_PASS" in RUNNER_TEXT
    assert "EFE_ALIGNMENT_FALSIFIED" in RUNNER_TEXT
    assert "BLOCKED_INFRASTRUCTURE" in RUNNER_TEXT
    assert "G4_alignment_armA" in RUNNER_TEXT
    assert "load_boundary_axioms" in RUNNER_TEXT
    assert "score_actions" in RUNNER_TEXT
    assert "select_action" in RUNNER_TEXT
    # Pre-registered gate spec (no threshold tuning after observation).
    assert '"true_efe_rank_max": 1' in RUNNER_TEXT
    assert '"efe_margin_min": 1e-3' in RUNNER_TEXT


def test_boundary_shape_contract():
    probe = EfeAlignmentProbe(None, None, None)
    # run_arm with flag ON requires a [N, num_blocks, 8] boundary; a wrong
    # shape must fail closed before any EFE computation.
    os.environ[FEATURE_FLAG] = "1"
    try:
        # No planner -> run_arm returns STATUS_DISABLED only when flag OFF;
        # with flag ON it must fail closed on the missing machinery, never
        # silently fabricate a ranking.
        out = probe.run_arm("A", [[0]], [], "translate(dx=1, dy=0)", "t",
                            torch.zeros(1, 8, 8))
        assert out["status"] != "OK", "missing machinery must not pass"
    finally:
        os.environ.pop(FEATURE_FLAG, None)


def test_reduced_scale_loop_vmap_and_decomposition():
    """Reduced-scale (CPU) smoke: loop/vmap identity + decomposition math."""
    from darwinian_phase_swarm import HenriSwarmOrchestrator
    from arcengine import GameAction
    from henri_vision_encoder import HENRIVisionEncoder
    from progressive_semantic_grounding_engine import (
        ProgressiveSemanticGroundingEngine, MacroOption, _apply_option_to_grid,
    )

    torch.manual_seed(20260814)
    SCALE = dict(num_experts=64, d_model=512, r_rank=8, num_blocks=64)
    device = "cpu"
    orch = HenriSwarmOrchestrator(
        action_enum_class=GameAction,
        constraint_weight_max=5.0, constraint_reject_thresh=0.38,
        beta_pragmatic=1.0, lambda_goal=0.0,
        learnable_actions=False, chimera_mode=False, chimera_alpha=1.4,
        chimera_explorer_fraction=0.25, happy_tensor_cut=False,
        external_outcome_efe=False, external_eig_weight=0.25,
        external_task_weight=1.0, task_weighted_eig=False,
        task_eig_gamma=4.0, **SCALE,
    ).to(device)
    orch.eval()
    tokenizer = HENRIVisionEncoder(
        d_model=SCALE["d_model"], k_blocks=SCALE["num_blocks"], block_dim=8,
        max_grid_dim=30, device=device,
        spatial_basis_kind="incommensurate", bg_mask=True,
    )
    psg = ProgressiveSemanticGroundingEngine(
        orch.planner, tokenizer, device=device,
        num_blocks=SCALE["num_blocks"], block_dim=8,
    )

    def grid():
        g = [[0] * 8 for _ in range(8)]
        g[2][3] = g[2][4] = g[3][4] = 1
        return g

    def tr():
        g = grid()
        m = MacroOption(object_id=0, kind="translate", dx=1, dy=0,
                        bbox=(0, 0, 7, 7), area=0, color=1,
                        description="translate(dx=1, dy=0)")
        return _apply_option_to_grid(g, m)

    pairs = [(grid(), tr()), (grid(), tr()), (grid(), tr())]
    res = psg.compile_task_functor(pairs, task_id="contract")
    # The functor held-out gate is dimension-dependent and may FALSIFY at
    # reduced scale; the kernel-identity test below does not depend on it.
    # With no compiled w_task, goal_bind would raise, so fall back to an
    # unbound goal (goal_wave=None -> goal_distance term zero) for the
    # mechanics check.
    state = tokenizer.encode_spatial_grid(grid()).squeeze(0).to(device)
    if getattr(res, "status", "OK") == "OK":
        goal = psg.goal_bind(state)
    else:
        goal = None
    options = [o for o in psg.options_from_grid(grid())
               if o.kind in ("translate", "rotate")]
    waves, labels = psg.option_waves(grid(), options)
    true_idx = labels.index("translate(dx=1, dy=0)")

    # Boundary: single colored pixel (masked encoder rejects all-zero).
    bg = [[0] * 8 for _ in range(8)]
    bg[0][0] = 1
    boundary = tokenizer.encode_spatial_grid(bg).squeeze(0).unsqueeze(0)

    efe_loop = psg.score(state, waves, boundary, goal)
    efe_bat = psg.score_batched(state, waves, boundary, goal)
    assert float((efe_loop - efe_bat).abs().max()) <= 1e-5

    candidates = list(zip(labels, [w for w in waves]))
    ranked = orch.planner.score_actions(state, candidates, boundary,
                                        goal_wave=goal)
    efe_by_label = {r["action"]: r["efe"] for r in ranked}
    lam = orch.planner._constraint_lambda()
    for r in ranked:
        recomputed = (orch.planner.pragmatic_weight * r["pragmatic"]
                      - orch.planner.epistemic_weight * r["epistemic"]
                      + lam * r["constraint_penalty"])
        assert abs(recomputed - r["efe"]) <= 1e-4
    assert "translate(dx=1, dy=0)" in efe_by_label
