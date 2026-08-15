"""Phase 8.8 contract tests — CC-OS incommensurate spatial carrier ingress (8.8-A/B).

Guards (default-OFF diagnostic; production path untouched when OFF):
- carriers band-limited incommensurate, deterministic, mean cos >= 0.85;
- adjacent-translation continuity >= 0.85 (gate 1), decay curve monotone;
- identity ~1.0, distinct-grid discrimination, per-block unit norm;
- exact phase-rotation Lie structure (apply_translation == encode of translated);
- physics-env carrier path: flag OFF = legacy byte-identical; ON = same-adjacent continuity;
- fail-closed empty foreground.
"""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from henri_spatial_carrier_ingress import VectorizedIncommensurateSpatialIngress
from physical_control_environments import InvertedPendulumEnvironment, CartPolePhysicsEnvironment

D = 65536
NUM_BLOCKS = 8192


@pytest.fixture(scope="module")
def ingress():
    torch.manual_seed(0)
    return VectorizedIncommensurateSpatialIngress(dimension=D, carrier_scale=0.10, device=torch.device("cpu"))


def _grid_single(color: int, r: int, c: int, H: int = 16, W: int = 16):
    g = np.zeros((H, W), dtype=int)
    g[r, c] = color
    return g


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item())


# ------------------------------------------------------------------ carriers
def test_carrier_mean_cos_gate(ingress):
    m = ingress.mean_cos_carrier()
    assert m >= 0.85, f"band-limited carrier mean cos {m:.4f} < 0.85 (sketch-fix requirement)"


def test_carriers_incommensurate_and_deterministic(ingress):
    torch.manual_seed(1)
    i2 = VectorizedIncommensurateSpatialIngress(dimension=D, carrier_scale=0.10, device=torch.device("cpu"))
    assert torch.equal(ingress.omega_x, i2.omega_x), "carrier must be deterministic"
    assert torch.equal(ingress.theta_y, i2.theta_y)
    # Incommensurate: no small-integer ratio between x and y carriers.
    ratio_frac = torch.frac((ingress.theta_y + 1e-6) / (ingress.omega_x + 1e-6))
    assert ratio_frac.std().item() > 1e-3, "x/y carriers must not be commensurate (locked ratio)"


# ------------------------------------------------------------------ mechanism
def test_adjacent_translation_continuity(ingress):
    w0 = ingress.encode_grid(_grid_single(color=2, r=8, c=8))
    w1 = ingress.encode_grid(_grid_single(color=2, r=8, c=9))
    c = _cos(w0, w1)
    assert c >= 0.85, f"adjacent(x+1) cos {c:.4f} < 0.85"


def test_translation_decay_monotone(ingress):
    w0 = ingress.encode_grid(_grid_single(color=2, r=8, c=8))
    prev = 1.0
    for dx in range(1, 5):
        w = ingress.encode_grid(_grid_single(color=2, r=8, c=8 + dx))
        c = _cos(w0, w)
        assert c <= prev + 1e-6, f"cos must decay with dx (dx={dx}: {c:.4f} > {prev:.4f})"
        prev = c
    assert prev >= 0.2, f"long-distance decay should still be meaningful, got {prev:.4f}"


def test_identity_and_discrimination(ingress):
    w0 = ingress.encode_grid(_grid_single(color=2, r=8, c=8))
    assert _cos(w0, w0) > 0.9999, "identical grid must map ~identically"
    wfar = ingress.encode_grid(_grid_single(color=2, r=2, c=2))
    assert _cos(w0, wfar) < 0.85, "distant positions must be discriminable"
    wcol = ingress.encode_grid(_grid_single(color=7, r=8, c=8))
    assert _cos(w0, wcol) < 0.95, "different colors at same position must be discriminable"


def test_per_block_unit_norm(ingress):
    w = ingress.encode_grid(_grid_single(color=3, r=5, c=10))
    blocks = ingress.to_blocks(w, NUM_BLOCKS)
    assert blocks.shape == (NUM_BLOCKS, 8)
    norms = blocks.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "per-block unit norm"


def test_empty_foreground_fails_closed(ingress):
    with pytest.raises(ValueError):
        ingress.encode_grid(np.zeros((8, 8), dtype=int))


# ------------------------------------------------------------------ Lie structure
def test_exact_phase_rotation_operator(ingress):
    g0 = _grid_single(color=2, r=8, c=8)
    w0 = ingress.encode_grid(g0)
    w_t = ingress.encode_grid(_grid_single(color=2, r=8, c=9))  # dx=1
    w_rot = ingress.apply_translation(w0, 1.0, 0.0)
    assert _cos(w_rot, w_t) > 0.99, f"operator(translation) must equal encode(translated): {_cos(w_rot, w_t):.4f}"


# ------------------------------------------------------------------ physics env path
def test_physics_env_flag_off_byte_identical():
    a = InvertedPendulumEnvironment()
    b = InvertedPendulumEnvironment(use_carrier_ingress=True)
    # same deterministic state
    a.reset(theta_init=0.3, dtheta_init=0.1)
    b.reset(theta_init=0.3, dtheta_init=0.1)
    wa = a.state_to_wave(64, torch.device("cpu"))
    wa = a.state_to_wave(64, torch.device("cpu"))  # legacy path deterministic
    # re-seed legacy env to same state (carrier env must not change legacy output)
    a2 = InvertedPendulumEnvironment()
    a2.reset(theta_init=0.3, dtheta_init=0.1)
    wa2 = a2.state_to_wave(64, torch.device("cpu"))
    assert torch.equal(wa, wa2), "legacy path must be deterministic across instances"
    # carrier path engages and stays on-manifold
    wb = b.state_to_wave(64, torch.device("cpu"))
    assert not torch.equal(wa, wb), "carrier flag must change the output"
    norms = wb.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_physics_env_carrier_continuity():
    e = InvertedPendulumEnvironment(use_carrier_ingress=True)
    e.reset(theta_init=0.1, dtheta_init=0.0)
    w0 = e.state_to_wave(8192, torch.device("cpu"))
    e.theta = 0.1 + 0.02  # small continuous step
    w1 = e.state_to_wave(8192, torch.device("cpu"))
    assert _cos(w0, w1) > 0.85, f"small physical step must be continuous, cos {_cos(w0, w1):.4f}"


def test_cartpole_carrier_flag_default_off_unchanged():
    c = CartPolePhysicsEnvironment()
    assert c.use_carrier_ingress is False
    w = c.state_to_wave(64, torch.device("cpu"))
    assert w.shape == (64, 8)
    assert torch.allclose(w.norm(p=2, dim=-1), torch.ones(64), atol=1e-5)
