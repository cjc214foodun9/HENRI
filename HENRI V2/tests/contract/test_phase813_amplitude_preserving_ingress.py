"""Phase 8.13 contract tests — Amplitude-Preserving Complex Ingress (KILL).

Blueprint: docs-HENRI_V2_PHASE_8_12_POSTMORTEM_AND_PHASE_8_13....pdf.pdf
(SHA 5e435cd9...). Pre-registration:
HENRI V2/experiments/sweeps/phase813_amplitude_preserving_ingress_design.md

PRE-REGISTERED G1 KILL (established d=512, exact math, D-independent):
1. COLOR-BLIND: amplitude-only color weighting is scale-invariant under
   cosine similarity: cos(3z, 6z) = 1.0 exactly for same-position grids.
   ARC color transformations are core object properties -> hard kill.
2. SHARED-SUPPORT COHERENCE: position-carrier superposition adds shared
   cells coherently: B-ring vs C-line share 3 cells -> cos = 6/sqrt(96)
   = 0.6113 EXACT (D cancels; same at D=65,536).
3. LEGACY CONTROL DOMINANCE: verified 7.3/7.4 encoder (incommensurate +
   bg_mask) phase-encodes color (pc term) -> 0.0000 (color pair),
   0.0033 (shared pair): strictly better on every pair. The discriminative
   channel is COLOR PHASE, not amplitude.

Surviving mechanism properties (recorded, not promoted):
- G1X translation exactness (angle residual ~1e-7) — holds for ANY
  superposition of position carriers, amplitude-weighted or not.
- G2 fit: NativeComplexWaveTransition is amplitude-invariant, fits the
  translation operator exactly on amplitude waves.
"""

import pytest
import torch

from amplitude_preserving_ingress import AmplitudePreservingComplexIngress
from complex_phase_transition import NativeComplexWaveTransition
from henri_vision_encoder import HENRIVisionEncoder

NB, BD = 64, 8
D = NB * BD


@pytest.fixture(scope="module")
def ingress():
    return AmplitudePreservingComplexIngress(
        dimension=D, num_blocks=NB, block_size=BD, device="cpu"
    )


@pytest.fixture(scope="module")
def legacy():
    return HENRIVisionEncoder(
        d_model=D,
        k_blocks=NB,
        block_dim=BD,
        device="cpu",
        spatial_basis_kind="incommensurate",
        bg_mask=True,
    )


GRID_A = [[0, 0, 0], [0, 3, 0], [0, 0, 0]]
GRID_A6 = [[0, 0, 0], [0, 6, 0], [0, 0, 0]]
GRID_B = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
GRID_C = [[0, 0, 2], [0, 0, 2], [0, 0, 2]]
GRID_D = [[5, 0, 0], [0, 0, 0], [0, 0, 5]]


def _cos(fn, a, b):
    return fn.distinct_cosine(torch.tensor(a), torch.tensor(b))


def _legacy_cos(enc, a, b):
    za = enc.encode_spatial_grid(torch.tensor(a, dtype=torch.long)).reshape(-1)
    zb = enc.encode_spatial_grid(torch.tensor(b, dtype=torch.long)).reshape(-1)
    return float(torch.abs(torch.dot(za, zb)) / (torch.norm(za) * torch.norm(zb) + 1e-12))


# ---- G1 KILL: pre-registered kill criteria, asserted as evidence ----

def test_g1_kill_color_blind(ingress):
    """Color 3 vs 6 at same position -> cos = 1.0 exactly (scale-invariant)."""
    c = _cos(ingress, GRID_A, GRID_A6)
    assert c > 0.999, f"expected color-blind cos ~1.0, got {c:.6f}"
    assert c >= 0.0100  # pre-registered kill threshold


def test_g1_kill_shared_support(ingress):
    """B-ring vs C-line share 3 cells -> cos ~= 6/sqrt(96) = 0.6113.

    Exact value is the D->inf carrier-orthogonality limit; at d=512 the
    finite-D cross-carrier correlation shifts it by ~1/sqrt(d) ~ 1e-3.
    """
    c = _cos(ingress, GRID_B, GRID_C)
    assert abs(c - 6.0 / (8 ** 0.5 * 12 ** 0.5)) < 5e-3, f"got {c:.6f}"
    assert c > 0.5  # kill magnitude at any D
    assert c >= 0.0100  # pre-registered kill threshold


def test_g1_occupancy_fix_works(ingress):
    """Disjoint-support grids ARE near-orthogonal (the 8.12 fix is real)."""
    for a, b in [(GRID_A, GRID_B), (GRID_A, GRID_C), (GRID_A, GRID_D)]:
        assert _cos(ingress, a, b) < 0.150, f"disjoint cos {_cos(ingress, a, b):.4f}"


# ---- G3 legacy control dominance (decisive comparator) ----

def test_g3_legacy_color_discrimination(legacy):
    """Legacy phase-encoded color: 3 vs 6 -> cos 0.0000 (vs 1.0000)."""
    c = _legacy_cos(legacy, GRID_A, GRID_A6)
    assert c < 0.010, f"legacy color cos {c:.6f}"


def test_g3_legacy_shared_support_discrimination(legacy):
    """Legacy phase-encoded positions: shared cells -> cos 0.0033 (vs 0.6113)."""
    c = _legacy_cos(legacy, GRID_B, GRID_C)
    assert c < 0.010, f"legacy shared-support cos {c:.6f}"


def test_g3_legacy_dominates_amplitude_on_hard_pairs(ingress, legacy):
    """Legacy < amplitude ingress on BOTH hard pairs (color, shared)."""
    c_amp_color = _cos(ingress, GRID_A, GRID_A6)
    c_leg_color = _legacy_cos(legacy, GRID_A, GRID_A6)
    c_amp_shared = _cos(ingress, GRID_B, GRID_C)
    c_leg_shared = _legacy_cos(legacy, GRID_B, GRID_C)
    assert c_leg_color < c_amp_color
    assert c_leg_shared < c_amp_shared


# ---- Surviving mechanism properties (recorded, NOT promoted) ----

def test_g1x_translation_exactness_single_pixel(ingress):
    """+x shift of a single pixel = diagonal rotation by omega_x (1e-7)."""
    z0 = ingress.forward(torch.tensor(GRID_A)).reshape(-1)
    z1 = ingress.forward(torch.tensor([[0, 0, 0], [0, 0, 3], [0, 0, 0]])).reshape(-1)
    delta = torch.angle(z1 * torch.conj(z0))
    tgt = torch.angle(torch.exp(1j * ingress.omega_x))
    err = float((delta - tgt).abs().mean())
    assert err < 1e-4, f"single-pixel translation phase err {err:.2e}"


def test_g2_transition_fits_translation_on_amplitude_waves(ingress):
    """8.11 transition fits the translation operator on amplitude waves."""
    tr = NativeComplexWaveTransition(dimension=D, num_blocks=NB, block_dim=BD)
    z0 = ingress.forward(torch.tensor(GRID_A)).reshape(-1)
    z1 = ingress.forward(torch.tensor([[0, 0, 0], [0, 0, 3], [0, 0, 0]])).reshape(-1)
    tr.update_phase_complex(z0, z1, action_idx=0, lr=1.0)
    g_ho = torch.tensor([[0, 4, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.long)
    g_ho_s = torch.tensor([[0, 0, 4], [0, 0, 0], [0, 0, 0]], dtype=torch.long)
    z_ho = ingress.forward(g_ho).reshape(-1)
    z_ho_s = ingress.forward(g_ho_s).reshape(-1)
    z_pred = tr.forward_complex(z_ho, action_idx=0)
    loss = float(torch.angle(z_ho_s * torch.conj(z_pred)).detach().abs().mean())
    assert loss <= 0.05, f"held-out translation fit loss {loss:.4f} > 0.05"


# ---- G5 default-OFF (additive module, production untouched) ----

def test_g5_default_off_no_planner_change():
    from efe_planner import EFEPlanner

    p = EFEPlanner(num_blocks=NB, d_model=D)
    assert type(p.transition).__name__ == "LowRankCoupledTransition"
