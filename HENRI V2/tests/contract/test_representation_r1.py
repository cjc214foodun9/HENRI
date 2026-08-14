"""Contract tests: Phase 8.3 R1 Representation Discrimination engine (default OFF).

Covers:
- flag OFF -> FEATURE_DISABLED, no functor allocation, no K1 harness run;
- K2 causal engagement: legacy byte identity (max diff 0.0) vs masked-ramp
  variant (diff != 0) on identical input;
- K3 masking fail-closed: all-color-0 grid raises (no zero wave);
- K4 discipline: score_eligible=false, diagnostic_only=true,
  rollout_authorized=false, no game.step, no engine.step;
- transform candidate set includes rotate/reflect labels;
- K1 harness schema (CPU reduced scale = plumbing check, NOT a K1 verdict).
"""

import os
import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from representation_discrimination_engine import (  # noqa: E402
    FEATURE_FLAG,
    RepresentationDiscriminationEngine,
    _legacy_encoder,
    _masked_ramp_encoder,
    K1_PASS,
    K1_FAIL,
    STATUS_DISABLED,
)

ENGINE_TEXT = Path(_REPO / "representation_discrimination_engine.py").read_text(
    encoding="utf-8")


def _make_engine(spatial_basis_kind="incommensurate", bg_mask=True):
    torch.manual_seed(20260814)
    from efe_planner import EFEPlanner
    from darwinian_phase_swarm import HenriSwarmOrchestrator
    from arcengine import GameAction

    SCALE = dict(num_experts=64, d_model=512, r_rank=8, num_blocks=64)
    device = "cpu"
    orch = HenriSwarmOrchestrator(
        action_enum_class=GameAction,
        constraint_weight_max=5.0, constraint_reject_thresh=0.38,
        beta_pragmatic=1.0, lambda_goal=0.5,
        learnable_actions=False, chimera_mode=False, chimera_alpha=1.4,
        chimera_explorer_fraction=0.25, happy_tensor_cut=False,
        external_outcome_efe=False, external_eig_weight=0.25,
        external_task_weight=1.0, task_weighted_eig=False,
        task_eig_gamma=4.0, **SCALE,
    ).to(device)
    orch.eval()
    tokenizer = _masked_ramp_encoder(
        d_model=SCALE["d_model"], num_blocks=SCALE["num_blocks"],
        block_dim=8, max_grid_dim=30, device=device,
        spatial_basis_kind=spatial_basis_kind, bg_mask=bg_mask,
    )
    eng = RepresentationDiscriminationEngine(
        orch.planner, tokenizer, device=device,
        num_blocks=SCALE["num_blocks"], block_dim=8,
    )
    return eng, orch


def test_flag_defaults_off():
    assert 'FEATURE_FLAG = "HENRI_ARC_REPRESENTATION_R1"' in ENGINE_TEXT
    assert 'os.environ.get(FEATURE_FLAG, "0") == "1"' in ENGINE_TEXT
    assert "HENRI_ARC_REPRESENTATION_R1" in ENGINE_TEXT
    assert "game.step(" not in ENGINE_TEXT
    assert "def step(" not in ENGINE_TEXT
    assert "score_eligible" in ENGINE_TEXT and "False" in ENGINE_TEXT


def test_status_disabled_and_no_allocation():
    eng, _ = _make_engine()
    s = eng.status()
    assert s["status"] == STATUS_DISABLED
    assert s["score_eligible"] is False
    assert s["diagnostic_only"] is True
    assert s["rollout_authorized"] is False
    # Flag OFF: harness refuses to run (no functor allocation).
    r = eng.k1_harness(
        [[1 if (r % 3 == 0 and c % 3 == 0) else 0 for c in range(9)]
         for r in range(9)],
        [([[0] * 6 for _ in range(6)], [[0] * 6 for _ in range(6)])],
        true_label="translate(dx=1, dy=0)", task_id="translation",
    )
    assert r["status"] == STATUS_DISABLED


def test_legacy_byte_identity():
    torch.manual_seed(7)
    grid = [[1 if (r * 7 + c) % 5 == 0 else 0 for c in range(12)]
            for r in range(12)]
    leg = _legacy_encoder(512, 64, 8, 30, "cpu")
    same = _masked_ramp_encoder(
        512, 64, 8, 30, "cpu",
        spatial_basis_kind="default", bg_mask=False,
    )
    a = leg.encode_spatial_grid(grid)
    b = same.encode_spatial_grid(grid)
    assert a.shape == b.shape == (1, 64, 8)
    assert float((a - b).abs().max().item()) == 0.0


def test_variant_causal_engagement():
    torch.manual_seed(7)
    grid = [[1 if (r * 7 + c) % 5 == 0 else 0 for c in range(12)]
            for r in range(12)]
    leg = _legacy_encoder(512, 64, 8, 30, "cpu")
    var = _masked_ramp_encoder(
        512, 64, 8, 30, "cpu",
        spatial_basis_kind="incommensurate", bg_mask=True,
    )
    a = leg.encode_spatial_grid(grid)
    b = var.encode_spatial_grid(grid)
    diff = float((a - b).abs().max().item())
    assert diff > 0.0, "masked-ramp variant must causally change output"


def test_masking_fail_closed():
    var = _masked_ramp_encoder(
        512, 64, 8, 30, "cpu",
        spatial_basis_kind="incommensurate", bg_mask=True,
    )
    with pytest.raises(ValueError):
        var.encode_spatial_grid([[0] * 6 for _ in range(6)])


def test_k1_harness_schema_cpu():
    """CPU reduced-scale plumbing check (NOT a K1 verdict; thresholds are
    production-scale CUDA only)."""
    eng, _ = _make_engine()
    os.environ[FEATURE_FLAG] = "1"
    try:
        g = [[1 if (r % 3 == 0 and c % 3 == 0) else 0 for c in range(9)]
             for r in range(9)]
        pairs = [(g, [[1 if (r % 3 == 1 and c % 3 == 1) else 0
                       for c in range(9)] for r in range(9)]),
                 (g, g)]
        out = eng.k1_harness(g, pairs, true_label="translate(dx=1, dy=0)",
                             task_id="translation")
        assert out["transform"] == "translation"
        assert out["score_eligible"] is False
        assert out["diagnostic_only"] is True
        assert out["authorizes_rollout"] is False
        if "k1" in out:
            assert out["k1"]["status"] in (K1_PASS, K1_FAIL)
            assert "true_rank" in out["k1"]
            assert "true_margin" in out["k1"]
            assert "num_options" in out
        else:
            # Functor may fail-closed at reduced scale (thresholds are
            # D=65,536-only); the schema must still carry the status +
            # discipline telemetry on the early-return path.
            assert out["status"] in ("OK", "FUNCTOR_FALSIFIED",
                                     "BLOCKED_EMPTY_FOREGROUND", K1_FAIL)
            assert "score_eligible" in out and out["score_eligible"] is False
            assert "diagnostic_only" in out and out["diagnostic_only"] is True
            assert "authorizes_rollout" in out and out["authorizes_rollout"] is False
    finally:
        del os.environ[FEATURE_FLAG]


def test_transform_candidate_labels():
    """The controlled candidate set must include rotate/reflect true labels."""
    from representation_discrimination_engine import (
        build_transform_options,
    )
    eng, _ = _make_engine()
    grid = [[1 if (r % 3 == 0 and c % 3 == 0) else 0 for c in range(9)]
            for r in range(9)]
    waves, labels = build_transform_options(
        grid, eng._psg.tokenizer, "cpu")
    assert "rotate(90)" in labels
    assert "reflect_h" in labels
    assert "reflect_v" in labels
    assert waves.shape[0] == len(labels)
    assert waves.shape[1:] == (64, 8)
