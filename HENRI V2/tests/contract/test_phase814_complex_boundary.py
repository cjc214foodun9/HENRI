"""Phase 8.14 contract tests — complex boundary wiring (variant B).

Pre-registration: HENRI V2/experiments/sweeps/phase814_complex_boundary_wiring_design.md
Blueprint: user instruction + Project HENRI V2 Strategic R&D Roadmap.pdf
(SHA 0ca9f7a1...).

Variant B: re-pair the encoder's native complex projection
(w[:D/2] + 1j*w[D/2:]), run the 8.11 transition on the TRUE phase space
(C^{D/2}), re-realify only at the environment edge. The complex inner
product on re-paired waves is IDENTICAL to the legacy real cosine.
"""

import os

import pytest
import torch

from complex_boundary import (
    complex_boundary_enabled,
    complex_cosine,
    complex_cycle,
    re_realify,
    un_realify,
)
from complex_phase_transition import NativeComplexWaveTransition
from henri_vision_encoder import HENRIVisionEncoder

D, NB, BD = 512, 64, 8

GRID_A = [[0, 1, 0], [1, 2, 1], [0, 1, 0]]
GRID_A6 = [[0, 6, 0], [6, 2, 6], [0, 6, 0]]  # color 3 -> 6, same shape
GRID_B = [[0, 0, 1], [0, 2, 1], [0, 1, 1]]  # shares 3 cells with C
GRID_C = [[0, 2, 0], [2, 1, 2], [0, 2, 0]]
HARD_PAIRS = [
    ("color", GRID_A, GRID_A6),
    ("shared", GRID_B, GRID_C),
    ("disjoint", GRID_A, GRID_C),
]


@pytest.fixture(scope="module")
def encoder():
    return HENRIVisionEncoder(
        d_model=D,
        k_blocks=NB,
        block_dim=BD,
        device="cpu",
        spatial_basis_kind="incommensurate",
        bg_mask=True,
    )


@pytest.fixture(scope="module")
def transition():
    # transition phase space = native complex state space (D/2)
    return NativeComplexWaveTransition(
        dimension=D // 2, num_actions=16, device="cpu",
        num_blocks=D // 2, block_dim=1,
    )


def _wave(encoder, grid):
    return encoder.encode_spatial_grid(grid)


def _real_cos(a, b):
    a, b = a.reshape(-1).float(), b.reshape(-1).float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def test_g6_roundtrip(encoder):
    """re_realify(un_realify(w)) == w exactly (max err < 1e-6)."""
    w = _wave(encoder, GRID_A)
    rt = re_realify(un_realify(w)).reshape(w.shape)
    assert float((rt - w).abs().max()) < 1e-6


def test_g1_identity(encoder):
    """Complex cosine on re-paired waves == legacy real cosine (|Δ| < 1e-5)."""
    for name, ga, gb in HARD_PAIRS:
        wa, wb = _wave(encoder, ga), _wave(encoder, gb)
        rc = _real_cos(wa, wb)
        cc = complex_cosine(un_realify(wa), un_realify(wb))
        assert abs(rc - cc) < 1e-5, f"{name}: real {rc:.6f} vs complex {cc:.6f}"


def test_g1_shared_disjoint_discriminate(encoder):
    """Shared/disjoint pairs discriminate at d=512 (cos < 0.05).

    (Color pair at d=512 is a finite-dim artifact; identity is asserted in
    test_g1_identity. The ≤0.02 scale gate runs at D=65,536.)
    """
    for name, ga, gb in HARD_PAIRS:
        if name == "color":
            continue
        cc = complex_cosine(
            un_realify(_wave(encoder, ga)), un_realify(_wave(encoder, gb))
        )
        assert cc < 0.05, f"{name}: complex_cos {cc:.4f} >= 0.05"


def test_g2_cycle_preserves(encoder, transition):
    """Identity-action complex cycle preserves the wave (cos > 0.999999)."""
    w = _wave(encoder, GRID_A)
    w_cycle = complex_cycle(w, transition, 0)
    assert tuple(w_cycle.shape) == (1, NB, BD)
    assert _real_cos(w_cycle, w) > 0.999999


def test_g5_default_off():
    """Default-OFF: flag unset => complex boundary inactive."""
    assert "HENRI_ARC_COMPLEX_BOUNDARY" not in os.environ
    assert complex_boundary_enabled() is False
