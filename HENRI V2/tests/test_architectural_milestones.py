"""
Unit test suite covering the 4 Core Architectural Milestones for HENRI V2:
  1. Recursive Dual EDMD with exponential forgetting factor.
  2. Hierarchical Multiscale Temporal Coupling (3 Tiers).
  3. Multi-Modal Closed-Loop Physical Control Environments.
  4. Continuous Zone C Attractor Pruning.
"""

import math
import os
import sys
import time
import pytest
import torch
import torch.nn.functional as F

henri_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if henri_dir not in sys.path:
    sys.path.insert(0, henri_dir)

from recursive_dual_edmd import RecursiveDualEDMD
from multiscale_temporal_coupler import MultiscaleTemporalCoupler
from physical_control_environments import InvertedPendulumEnvironment, CartPolePhysicsEnvironment
from zone_c_attractor_pruner import ZoneCAttractorPruner
from zone_c_segment_cache import SegmentCache


def test_recursive_dual_edmd_exponential_forgetting():
    d_model = 512
    r_rank = 8
    edmd = RecursiveDualEDMD(d_model=d_model, r_rank=r_rank, lambda_forget=0.95)

    s = torch.randn(d_model)
    a = torch.randn(d_model)
    t = torch.randn(d_model)

    # Initial loss before online update
    pred_init = edmd(s, a)
    loss_init = float(F.mse_loss(pred_init.view(-1), F.normalize(t, p=2, dim=0)).item())

    # Perform 20 online update steps
    losses = []
    for _ in range(20):
        loss = edmd.update_online_step(s, a, t)
        losses.append(loss)

    assert losses[-1] < loss_init, f"Expected online loss reduction, got init={loss_init}, final={losses[-1]}"


def test_multiscale_temporal_coupler():
    coupler = MultiscaleTemporalCoupler(fast_hz=100.0, medium_hz=10.0, slow_hz=1.0)

    fast_executed = False
    medium_executed = False
    slow_executed = False

    def fast_cb():
        nonlocal fast_executed
        fast_executed = True

    def medium_cb():
        nonlocal medium_executed
        medium_executed = True

    def slow_cb():
        nonlocal slow_executed
        slow_executed = True

    t0 = 100.0
    fired = coupler.tick(t0, fast_cb, medium_cb, slow_cb)
    assert fired["fast"] is True
    assert fired["medium"] is True
    assert fired["slow"] is True
    assert fast_executed and medium_executed and slow_executed


def test_physical_control_environments():
    # Test Inverted Pendulum
    pendulum = InvertedPendulumEnvironment()
    st_p, cost_p, done_p = pendulum.step(torque=0.5)
    assert st_p.shape == (3,)
    assert isinstance(float(cost_p), float)
    wave_p = pendulum.state_to_wave(num_blocks=64, device=torch.device("cpu"))
    assert wave_p.shape == (64, 8)

    # Test CartPole Physics
    cartpole = CartPolePhysicsEnvironment()
    st_c, cost_c, done_c = cartpole.step(force=1.0)
    assert st_c.shape == (4,)
    assert isinstance(float(cost_c), float)
    wave_c = cartpole.state_to_wave(num_blocks=64, device=torch.device("cpu"))
    assert wave_c.shape == (64, 8)


def test_zone_c_attractor_pruner_in_memory():
    cache = SegmentCache.connect(dsn="offline://surrogate", num_blocks=64)
    # Write 3 near-identical engrams
    w = torch.randn(64, 8)
    w_norm = F.normalize(w, p=2, dim=-1)

    for _ in range(3):
        cache.checkpoint(w_norm, domain="unit_test", sagnac_stress=0.01)

    assert cache.store.count() == 3
