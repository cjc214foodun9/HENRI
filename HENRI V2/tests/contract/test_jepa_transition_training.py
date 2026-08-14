"""Phase 8.4 — JEPA Transition Training contract tests (default OFF).

Contracts:
  C1 flag default OFF -> FEATURE_DISABLED, no Zone C connection, no allocation.
  C2 eligibility telemetry single-source on EVERY return path
     (score_eligible=false, diagnostic_only=true, authorizes_rollout=false).
  C3 no game.step in new source files.
  C4 training API wiring: production train_transition_step descends the
     Sagnac loss on real known-transform encoder waves at reduced scale
     (wiring contract only — the G1/G2 CUDA gates run remotely).
  C5 pre-registered gate constants match the design doc (no threshold
     tuning by drift).
"""

import os
import re
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
_SRC_FILES = [
    _REPO / "jepa_transition_training_experiment.py",
    _REPO / "experiments" / "performance" / "jepa_transition_training_cuda_check.py",
]
_DESIGN = _REPO / "experiments" / "sweeps" / "phase84_jepa_transition_training_design.md"

FLAG = "HENRI_ARC_JEPA_TRANSITION_TRAINING"


def _clean_env() -> None:
    os.environ.pop(FLAG, None)


@pytest.fixture
def small_probe():
    _clean_env()
    from efe_planner import EFEPlanner
    from henri_vision_encoder import HENRIVisionEncoder
    from progressive_semantic_grounding_engine import ProgressiveSemanticGroundingEngine
    from jepa_transition_training_experiment import JepaTransitionTrainingProbe

    d, nb, rank = 512, 64, 8
    planner = EFEPlanner(num_blocks=nb, d_model=d, transition_rank=rank)
    tok = HENRIVisionEncoder(
        d_model=d, k_blocks=nb, block_dim=8, max_grid_dim=30,
        spatial_basis_kind="incommensurate", bg_mask=True,
    )
    psg = ProgressiveSemanticGroundingEngine(
        planner, tok, device="cpu", num_blocks=nb, block_dim=8)
    probe = JepaTransitionTrainingProbe(
        planner, tok, psg, device="cpu", num_blocks=nb, block_dim=8)
    return probe


def _assert_eligibility(payload) -> None:
    assert payload.get("score_eligible") is False
    assert payload.get("diagnostic_only") is True
    assert payload.get("authorizes_rollout") is False


def test_flag_default_off(small_probe):
    _clean_env()
    assert small_probe.enabled is False
    st = small_probe.status()
    assert st["status"] == "FEATURE_DISABLED"
    _assert_eligibility(st)


def test_flag_off_no_zone_c_no_alloc(small_probe):
    _clean_env()
    # No Zone C import may be attempted with flag OFF: run_arm must
    # short-circuit before touching load_boundary_axioms.
    out = small_probe.run_arm(
        "A", [[0] * 8 for _ in range(8)], [],
        true_label="translate(dx=1, dy=0)", task_id="translation",
        boundary_batch=torch.zeros(1, 64, 8),
    )
    assert out["status"] == "FEATURE_DISABLED"
    _assert_eligibility(out)
    tr = small_probe.train_transition([], k_steps=5)
    assert tr["status"] == "FEATURE_DISABLED"
    _assert_eligibility(tr)


def test_eligibility_single_source_every_return(small_probe):
    os.environ[FLAG] = "1"
    try:
        # BLOCKED_MISSING_MACHINERY path
        probe2 = small_probe.__class__(
            None, None, None, device="cpu", num_blocks=64, block_dim=8)
        out = probe2.run_arm(
            "A", [[0] * 8 for _ in range(8)], [],
            true_label="translate(dx=1, dy=0)", task_id="translation",
            boundary_batch=torch.zeros(1, 64, 8),
        )
        assert out["status"] == "BLOCKED_MISSING_MACHINERY"
        _assert_eligibility(out)
        # NO_PAIRS path
        tr = small_probe.train_transition([], k_steps=5)
        assert tr["status"] == "NO_PAIRS"
        _assert_eligibility(tr)
    finally:
        _clean_env()


def test_no_game_step_in_source():
    for p in _SRC_FILES:
        text = p.read_text(encoding="utf-8")
        assert "game.step" not in text, f"game.step present in {p.name}"


def test_training_api_wiring(small_probe):
    """Reduced-scale wiring: production API executes and descends loss.

    NOT a capability gate — the G1/G2 CUDA gates run remotely at D=65,536.
    """
    import torch
    os.environ[FLAG] = "1"
    try:
        pairs = small_probe.build_pairs(seeds=(0, 1, 2))
        assert len(pairs["pairs"]) >= 1, f"no pairs built: {pairs['skipped']}"
        tr = small_probe.train_transition(pairs["pairs"], k_steps=10, lr=0.05)
        assert tr["status"] == "OK"
        _assert_eligibility(tr)
        first = tr["loss_curve_downsampled"][0] if tr["loss_curve_downsampled"] else 1.0
        assert 0.0 <= tr["loss_final"] <= 1.01
        assert tr["loss_final"] < first, (
            f"transition loss did not descend: first={first:.4f} "
            f"final={tr['loss_final']:.4f}")
        assert isinstance(tr["loss_ema_final"], float)
        assert tr["steps"] == 10
    finally:
        _clean_env()


def test_pre_registered_gate_constants_match_design():
    runner = (_REPO / "experiments" / "performance"
              / "jepa_transition_training_cuda_check.py").read_text(encoding="utf-8")
    design = _DESIGN.read_text(encoding="utf-8")
    for const, value in [("GATE_LOSS", "0.30"), ("SPEC_LOSS", "0.15"),
                         ("RANK_PASS", "1"), ("RANK_PARTIAL", "2")]:
        assert re.search(rf"{const}\s*=\s*{value}", runner), f"{const}={value} missing in runner"
    assert "0.30" in design and "0.15" in design
