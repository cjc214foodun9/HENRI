"""Contract tests for the M2 horizon-8 open-loop coherence diagnostic
(SPEC-2026-08-28-M2SUCC, sealed #bb0be1c9).

Module contract (frozen):
- open_loop_rollout(transition, state_wave, action_wave, horizon=8) returns
  list of 8 [num_blocks, 8] predictions; None on any non-finite output.
- Rollout is CAUSAL: only the current (already-executed) action wave is used;
  no future action waves enter.
- sagnac_delta(pred, emp) = 1 - cos(pred, emp) in [0, 2] (full-wave cosine).
- Per-horizon mean over steps; gate: mean <= 0.15 for every k in 1..8.
- Engagement: m2_engaged = True iff at least one finite delta was emitted.
"""

import numpy as np
import torch
import torch.nn.functional as F

from henri_m2_coherence import (
    open_loop_rollout,
    sagnac_delta,
    per_horizon_means,
    M2_HORIZON,
)


class _IdentityTransition(torch.nn.Module):
    def forward(self, state_wave, action_wave):
        return state_wave


class _ScaleTransition(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, state_wave, action_wave):
        out = state_wave * self.scale
        return out / (torch.norm(out, p=2, dim=-1, keepdim=True) + 1e-9)


class _NaNTransition(torch.nn.Module):
    def forward(self, state_wave, action_wave):
        return torch.full_like(state_wave, float("nan"))


def _wave(num_blocks=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(num_blocks, 8, generator=g)
    return w / (torch.norm(w, p=2, dim=-1, keepdim=True) + 1e-9)


def test_rollout_shape_and_finite():
    t = _IdentityTransition()
    preds = open_loop_rollout(t, _wave(), _wave(), horizon=8)
    assert preds is not None
    assert len(preds) == 8
    for p in preds:
        assert p.shape == (8, 8)
        assert torch.isfinite(p).all()


def test_rollout_returns_none_on_nan():
    t = _NaNTransition()
    preds = open_loop_rollout(t, _wave(), _wave(), horizon=8)
    assert preds is None


def test_rollout_rejects_horizon_out_of_range():
    t = _IdentityTransition()
    try:
        open_loop_rollout(t, _wave(), _wave(), horizon=0)
        raise AssertionError("horizon=0 should raise")
    except ValueError:
        pass
    try:
        open_loop_rollout(t, _wave(), _wave(), horizon=9)
        raise AssertionError("horizon=9 should raise")
    except ValueError:
        pass


def test_sagnac_delta_bounds():
    w = _wave(seed=1)
    # identical -> 0
    assert abs(sagnac_delta(w, w.clone())) < 1e-6
    # EXACT per-row orthogonalization (rows of w are unit-norm; projecting
    # each randn row onto w's row and subtracting leaves full-wave dot = 0).
    # (The first version projected with the FLATTENED norm — ‖w_flat‖²=8 —
    # leaving a residual component along w: cos≈0.27, delta≈0.73. Test bug.)
    g = torch.Generator().manual_seed(2)
    w2 = torch.randn(8, 8, generator=g)
    w2 = w2 - (w2 * w).sum(dim=-1, keepdim=True) * w
    w2 = w2 / (torch.norm(w2, p=2, dim=-1, keepdim=True) + 1e-9)
    d = sagnac_delta(w, w2)
    assert 0.95 < d < 1.05, d
    # anti-parallel -> ~2
    d2 = sagnac_delta(w, -w)
    assert 1.95 < d2 <= 2.0 + 1e-6, d2
    # all in [0, 2]
    assert 0.0 <= d <= 2.0 and 0.0 <= d2 <= 2.0


def test_per_horizon_means_and_gate():
    rng = np.random.default_rng(3)
    deltas = {k: [] for k in range(1, M2_HORIZON + 1)}
    # identity transition with clean waves -> deltas ~ 0
    t = _IdentityTransition()
    for _ in range(30):
        s = _wave(seed=int(rng.integers(0, 1000)))
        preds = open_loop_rollout(t, s, _wave(seed=int(rng.integers(0, 1000))), horizon=8)
        for k, p in enumerate(preds, start=1):
            deltas[k].append(sagnac_delta(p, s))
    means = per_horizon_means(deltas)
    assert set(means.keys()) == set(range(1, M2_HORIZON + 1))
    for k, m in means.items():
        assert m <= 0.15, (k, m)  # gate passes for identity


def test_per_horizon_means_detects_drift():
    deltas = {k: [0.5] * 20 for k in range(1, M2_HORIZON + 1)}
    means = per_horizon_means(deltas)
    assert all(m > 0.15 for m in means.values())


def test_engagement_semantics():
    # m2_engaged is True iff at least one finite delta exists.
    from henri_m2_coherence import m2_engaged
    assert m2_engaged({1: [0.1]}) is True
    assert m2_engaged({k: [] for k in range(1, M2_HORIZON + 1)}) is False
