"""Phase 8.9 contract tests — FrequencyDomainDiagonalTransition (8.9-A/B) + analytic carriers.

Guards (small-D CPU, deterministic):
- identity init -> forward == state (normalized)
- unit modulus per element
- complex64 dtype
- phasor deterministic
- analytic translation EXACT (encode(r+dx,c+dy) == encode(r,c) . rotator(dx,dy))
- band-limited incommensurate frequencies
- Wirtinger update recovers Theta_true in ONE step at lr=1.0 (G3 < 1e-4 in <=10)
- sagnac floor after recovery < 1e-4
- no [D,D] materialization in forward
"""

import math

import pytest
import torch

from henri_frequency_domain_transition import (
    AnalyticSpatialCarriers,
    FrequencyDomainDiagonalTransition,
    TAU,
)

D = 2048
NUM_ACTIONS = 4
DEV = "cpu"


@pytest.fixture()
def model():
    return FrequencyDomainDiagonalTransition(dimension=D, num_actions=NUM_ACTIONS, device=DEV)


@pytest.fixture()
def carriers():
    return AnalyticSpatialCarriers(dimension=D, carrier_scale=0.10, device=DEV)


def test_identity_init_forward_passthrough(model):
    state = torch.randn(D, dtype=torch.complex64) + 1j * torch.randn(D, dtype=torch.complex64)
    state = state / torch.norm(state)
    out = model.forward(state, torch.tensor(0))
    assert torch.allclose(out, state, atol=1e-6)


def test_unit_modulus(model):
    # FHRR phasor convention: per-element unit modulus preserved under rotation
    state = torch.polar(torch.ones(D, dtype=torch.float32), torch.randn(D) * 2.0 * math.pi)
    out = model.forward(state, torch.tensor(1))
    mod = torch.abs(out)
    assert torch.allclose(mod, torch.ones_like(mod), atol=1e-6)


def test_complex64_output(model):
    state = torch.randn(D, dtype=torch.complex64)
    out = model.forward(state, torch.tensor(2))
    assert out.dtype == torch.complex64


def test_phasor_deterministic(model):
    p1 = model.phasor(torch.tensor(3))
    p2 = model.phasor(torch.tensor(3))
    assert torch.equal(p1, p2)
    assert p1.dtype == torch.complex64


def test_action_phases_initialized_zero(model):
    assert torch.all(model.action_phases == 0.0)


def test_analytic_translation_exact(carriers):
    r, c = 5.0, 7.0
    dx, dy = 3.0, -2.0
    psi = carriers.encode(r, c)
    psi_shift = carriers.encode(r + dx, c + dy)
    rot = carriers.rotator(dx, dy)
    pred = psi * rot
    cos = torch.abs(torch.sum(pred * torch.conj(psi_shift))) / D
    assert cos > 0.999999, f"translation exactness failed: cos={cos}"


def test_band_limited_frequencies(carriers):
    s = 0.10
    assert torch.max(torch.abs(carriers.omega)) <= 2.0 * math.pi * s + 1e-6
    assert torch.max(torch.abs(carriers.theta)) <= 2.0 * math.pi * s + 1e-6


def test_incommensurate_frequency_vectors(carriers):
    # omega/theta from irrational seeds (tau, sqrt(3)) must not be identical
    assert not torch.allclose(carriers.omega, carriers.theta, atol=1e-3)


def test_wirtinger_one_step_recovery(carriers, model):
    # true action 0 phase = dx*Omega + dy*Theta
    dx, dy = 2.0, 3.0
    theta_true = dx * carriers.omega + dy * carriers.theta  # [D]
    r, c = 4.0, 6.0
    st = carriers.encode(r, c)
    stp1 = carriers.encode(r + dx, c + dy)
    model.update_online_wirtinger(st, stp1, action_idx=0, lr=1.0)
    theta_learned = model.action_phases[0]
    err = torch.max(torch.abs(theta_learned - theta_true))
    assert err < 1e-4, f"G3 phase recovery failed: err={err}"


def test_sagnac_floor_after_recovery(carriers, model):
    dx, dy = 1.0, -1.0
    theta_true = dx * carriers.omega + dy * carriers.theta
    with torch.no_grad():
        model.action_phases[0].copy_(theta_true)
    st = carriers.encode(2.0, 3.0)
    stp1 = carriers.encode(2.0 + dx, 3.0 + dy)
    pred = model.forward(st, torch.tensor(0))
    sag = 1.0 - torch.abs(torch.sum(pred * torch.conj(stp1))) / D
    assert sag < 1e-4, f"sagnac floor failed: {sag}"


def test_no_dd_materialization(model):
    # forward must not allocate a [D, D] tensor — guard via the phasor being [D]
    state = torch.randn(D, dtype=torch.complex64)
    state = state / torch.norm(state)
    phi = model.phasor(torch.tensor(0))
    assert phi.shape == (D,)
    out = model.forward(state, torch.tensor(0))
    assert out.shape == (D,)
