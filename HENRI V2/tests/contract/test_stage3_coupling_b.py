"""Contract tests — Stage 3 Mechanism B (AxiomaticDeficiencyError).

Carrier: carrier/stage3-coupling. Prereg:
experiments/verification/stage3_coupling_prereg.md (sealed 49f9969f).

Mechanism B: HenriTaskOperator.compile_from_demos raises
AxiomaticDeficiencyError when m < 2 or per-block cross-covariance rank < 2
(sigma_2/sigma_1 <= 1e-6). Runner catch is fail-closed and specific
(GOAL_ADAPTER_AXIOMATIC_DEFICIENCY, no identity assignment, no fallback).
"""

import sys
from pathlib import Path

import pytest
import torch

HENRI2 = Path(__file__).resolve().parents[2]  # <worktree>/HENRI V2
sys.path.insert(0, str(HENRI2))

RUNNER = HENRI2 / "production_arc_run.py"

from henri_goal_adapter import HenriGoalAdapter  # noqa: E402

NUM_BLOCKS_MIN = 8
BLOCK_DIM = 8


def _adapter() -> HenriGoalAdapter:
    return HenriGoalAdapter(num_blocks=NUM_BLOCKS_MIN, block_dim=BLOCK_DIM,
                            device="cpu")


def test_B_m1_raises():
    from henri_goal_adapter import AxiomaticDeficiencyError
    torch.manual_seed(1)
    adapter = _adapter()
    x = torch.randn(1, NUM_BLOCKS_MIN, BLOCK_DIM)
    y = torch.randn(1, NUM_BLOCKS_MIN, BLOCK_DIM)
    with pytest.raises(AxiomaticDeficiencyError):
        adapter.build_goal(x, y, torch.randn(NUM_BLOCKS_MIN, BLOCK_DIM))


def test_B_parallel_outputs_raises():
    from henri_goal_adapter import AxiomaticDeficiencyError
    torch.manual_seed(2)
    adapter = _adapter()
    x = torch.randn(2, NUM_BLOCKS_MIN, BLOCK_DIM)
    y = torch.stack([x[0].clone(), x[0].clone()])  # y_1 == y_2 -> rank-1 cross-cov
    with pytest.raises(AxiomaticDeficiencyError):
        adapter.build_goal(x, y, torch.randn(NUM_BLOCKS_MIN, BLOCK_DIM))


def test_B_generic_m2_passes_geometry():
    torch.manual_seed(3)
    adapter = _adapter()
    x = torch.randn(2, NUM_BLOCKS_MIN, BLOCK_DIM)
    y = torch.randn(2, NUM_BLOCKS_MIN, BLOCK_DIM)
    res = adapter.build_goal(x, y, torch.randn(NUM_BLOCKS_MIN, BLOCK_DIM))
    assert res["goal_wave"].shape == (NUM_BLOCKS_MIN, BLOCK_DIM)
    assert res["orthogonality_err"] <= 1e-4
    assert res["demo_recon_cos"] > 0.05


def test_B_runner_catch_is_fail_closed_and_specific():
    src = RUNNER.read_text(encoding="utf-8")
    assert "GOAL_ADAPTER_AXIOMATIC_DEFICIENCY" in src
    idx_spec = src.index("except AxiomaticDeficiencyError")
    idx_gen = src.index("except Exception as _adapter_exc")
    assert idx_spec < idx_gen
    # No identity fallback may exist in the catch region.
    seg = src[idx_gen - 400:idx_gen]
    assert "torch.eye" not in seg


def test_B_adapter_has_no_identity_fallback():
    src = (HENRI2 / "henri_goal_adapter.py").read_text(encoding="utf-8")
    region = src.split("compile_from_demos")[1].split("def apply")[0]
    assert "torch.eye" not in region
