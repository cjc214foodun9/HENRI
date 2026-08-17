"""Phase 8.22 D38 contract: fail-closed non-finite guard in train_transition_batch.

Regression target: 8.21 run2 crashed at env 19 with
`linalg.svd: input matrix contained non-finite values` at
efe_planner.py:1340 (CPU fallback tier). The guard must skip the fit
(return 0.0, record non_finite_skip) instead of raising, and the healthy
path must be byte-identical (no non_finite_skip key).
"""
import math

import pytest
import torch


@pytest.fixture
def planner_small():
    # Local contract tests run CPU-only (env-clean isolated interpreter).
    from efe_planner import EFEPlanner

    return EFEPlanner(num_blocks=64, d_model=512)


def _wave(shape, device, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(*shape, generator=g).to(device)
    return w / torch.norm(w, p=2, dim=-1, keepdim=True)


def test_d38_nonfinite_buffer_skips_fit(planner_small):
    p = planner_small
    dev = p.transition.field_V.device
    N = 8
    states = torch.stack([_wave((64, 8), dev, 100 + i) for i in range(N)])
    actions = torch.stack([_wave((64, 8), dev, 200 + i) for i in range(N)])
    # Poisoned targets: non-finite observed_nexts.
    nexts = torch.full_like(states, float("nan"))

    loss = p.train_transition_batch(states, actions, nexts, iters=1)
    diag = p.last_edmd_diagnostics
    assert loss == 0.0, "non-finite buffer must skip the fit, not raise"
    assert diag is not None
    assert diag.get("non_finite_skip") is True
    assert diag.get("cholesky_failed") is True


def test_d38_healthy_path_unchanged(planner_small):
    p = planner_small
    N = 8
    states = torch.stack([_wave((64, 8), p.transition.field_V.device, 300 + i) for i in range(N)])
    actions = torch.stack([_wave((64, 8), p.transition.field_V.device, 400 + i) for i in range(N)])
    nexts = torch.stack([p.transition(states[i], actions[i]) for i in range(N)])

    loss = p.train_transition_batch(states, actions, nexts, iters=1)
    diag = p.last_edmd_diagnostics
    assert math.isfinite(float(loss))
    assert diag is not None
    assert diag.get("non_finite_skip") is None, "healthy path must not mark skip"
    assert diag["cholesky_failed"] is False
