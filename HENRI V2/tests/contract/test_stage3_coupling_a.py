"""Contract tests — Stage 3 Mechanism A (exteroceptive scorecard delta-gain).

Carrier: carrier/stage3-coupling. Prereg:
experiments/verification/stage3_coupling_prereg.md (sealed 49f9969f).

Mechanism A: EFEPlanner.train_transition_step accepts outcome_delta (default
None). When provided it overrides the motion-based valence: outcome_delta=1
crystallizes (lr/(1+nu)), outcome_delta=0 is neutral. Default path
(outcome_delta=None) is byte-identical. Runner wiring: HENRI_DELTA_GAIN=1
(with HENRI_ARC_SCORECARD_DELTA=1) forwards the scorecard delta into
train_ctx and the transition update, and emits delta_gain_valid/scorecard_delta_nu.
"""

import inspect
import os
import sys
from pathlib import Path

import torch

HENRI2 = Path(__file__).resolve().parents[2]  # <worktree>/HENRI V2
sys.path.insert(0, str(HENRI2))

RUNNER = HENRI2 / "production_arc_run.py"

from efe_planner import EFEPlanner  # noqa: E402

D_MODEL_MIN = 64
NUM_BLOCKS_MIN = 8
BLOCK_DIM = 8
R_MIN = 4


def _planner() -> EFEPlanner:
    return EFEPlanner(
        num_blocks=NUM_BLOCKS_MIN,
        d_model=D_MODEL_MIN,
        transition_rank=R_MIN,
    )


def test_A_signature_accepts_outcome_delta_default_none():
    sig = inspect.signature(EFEPlanner.train_transition_step)
    assert "outcome_delta" in sig.parameters
    assert sig.parameters["outcome_delta"].default is None


def test_A_outcome_delta_overrides_valence_identically():
    torch.manual_seed(7)
    s = torch.randn(NUM_BLOCKS_MIN, BLOCK_DIM)
    a = torch.randn(NUM_BLOCKS_MIN, BLOCK_DIM)
    n = torch.randn(NUM_BLOCKS_MIN, BLOCK_DIM)

    base = _planner()
    p1 = _planner()
    p1.load_state_dict(base.state_dict())
    loss_valence = p1.train_transition_step(s, a, n, lr=0.05, valence=0.7)

    p2 = _planner()
    p2.load_state_dict(base.state_dict())
    loss_delta = p2.train_transition_step(
        s, a, n, lr=0.05, valence=0.0, outcome_delta=0.7
    )

    assert abs(loss_valence - loss_delta) < 1e-9
    for (n1, w1), (n2, w2) in zip(p1.named_parameters(), p2.named_parameters()):
        assert n1 == n2
        assert torch.equal(w1.detach(), w2.detach())


def test_A_default_path_no_dense_allocation():
    p = _planner()
    for name, param in p.named_parameters():
        shape = tuple(param.shape)
        assert len(shape) <= 3, f"unexpected param rank {name} {shape}"
        if len(shape) == 2:
            assert shape[0] != shape[1] or shape[1] != D_MODEL_MIN, (
                f"dense [D,D] kernel detected: {name} {shape}"
            )


def test_A_runner_flag_and_delta_gain_wiring():
    src = RUNNER.read_text(encoding="utf-8")
    assert 'HENRI_DELTA_GAIN = os.environ.get("HENRI_DELTA_GAIN", "0") == "1"' in src
    assert "outcome_delta=" in src
    assert '"scorecard_delta":' in src
    assert "delta_gain_valid" in src
    assert "scorecard_delta_nu" in src
