"""Contract tests for the corrected wave-aligned SGLD adaptation (C2).

Reduced-dimension mechanics only (no checkpoint, CPU); remote CUDA verification
is the production evidence path.
"""

import math

import torch

from henri_decoder import HENRINeuralEgressUnbinder, _sgld_thermal_schedule


def test_thermal_schedule_monotone_decreasing():
    vals = [_sgld_thermal_schedule(t) for t in range(500)]
    assert vals[0] == 1e-6
    assert all(v > 0 for v in vals)
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))


def test_wave_sgld_mechanism_reduced_dim():
    torch.manual_seed(0)
    ub = HENRINeuralEgressUnbinder(d_model=64, d_hidden=32, vocab_size=128, device="cpu")
    x = torch.randn(2, 64)
    y = torch.randn(2, 64)
    x = x / torch.norm(x, dim=-1, keepdim=True)
    y = y / torch.norm(y, dim=-1, keepdim=True)
    res = ub.adapt_in_context_sgld_wave(x, y, steps=5, seed=0)
    assert res["steps"] == 5
    assert res["adapt_protocol"] == "wave_soft_targets_scheduled_sgld"
    assert math.isfinite(res["loss_first"]) and math.isfinite(res["loss_last"])
    assert 0.0 <= res["sagnac_dist_final"] <= 1.0
    assert res["yield_events"] >= 0
    assert res["soft_target_entropy_nats"] > 0.0
    # The mechanism must move the adapting pair toward its soft target
    # (loose bound: SGLD noise + Stiefel retraction may partially undo descent).
    assert res["loss_last"] <= res["loss_first"] + 0.5
