"""Phase 8.12 contract tests — Complex-Native Ingress (KILL, pre-registered).

Blueprint: HENRI V2 Structural Analysis & Phase 8.12 Architecture.pdf.
Pre-registration: HENRI V2/experiments/sweeps/phase812_complex_native_ingress_design.md.

G1 (8.12-A): adjacent spatial-state complex cosine >= 0.85 AND distinct-state
cosine < 0.95. OBSERVED FALSIFIED at d=512: no bandwidth s satisfies both
(amplitude-normalized per-element superposition collapses all grids onto the
shared carrier subspace). Legacy control (incommensurate + bg_mask)
discriminates (cos ~ 0.005) — the blueprint's "legacy encoder corrupt"
premise is FALSIFIED for that configuration.
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "HENRI V2"))

from complex_native_ingress import ComplexNativeIngress  # noqa: E402

D = 64 * 8  # d=512 local (contract scale)
NB, BD = 64, 8

GRID_A = [[0, 1, 0], [1, 2, 1], [0, 1, 0]]   # cross
GRID_B = [[0, 0, 1], [0, 2, 1], [0, 1, 1]]   # A translated right by 1
GRID_C = [[0, 2, 0], [2, 1, 2], [0, 2, 0]]   # ring color swap
GRID_D = [[3, 3, 3], [3, 0, 3], [3, 3, 3]]   # hollow square
GRID_E = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]   # plus


@pytest.fixture()
def enc():
    return ComplexNativeIngress(dimension=D, num_blocks=NB, block_dim=BD,
                                device="cpu", band_s=0.10)


def _cos_pair(enc, ga, gb):
    return float(enc.complex_cosine(enc.encode_grid(ga), enc.encode_grid(gb)))


def test_g1_kill_no_bandwidth_satisfies_both_gates():
    """Kill criterion G1: assert NO s has adj>=0.85 AND distinct<0.95.

    The sweep across s shows the reachable set is empty: narrow band keeps
    adjacency high but makes ALL distinct grids collinear (cos ~ 0.9999);
    wide band discriminates but destroys adjacency (cos < 0.85).
    """
    ok_any = False
    for s in (0.05, 0.10, 0.20, 0.40, 0.60, 0.85):
        e = ComplexNativeIngress(dimension=D, num_blocks=NB, block_dim=BD,
                                 device="cpu", band_s=s)
        adj = _cos_pair(e, GRID_A, GRID_B)
        distinct = max(_cos_pair(e, GRID_A, g) for g in (GRID_C, GRID_D))
        if adj >= 0.85 and distinct < 0.95:
            ok_any = True
            break
    assert not ok_any, "G1 pair (adj>=0.85 AND distinct<0.95) is REACHABLE — kill not fired"


def test_g2_kill_unit_modulus_discards_pattern_amplitude():
    """Mechanistic root cause: unit-modulus renormalization destroys the
    amplitude information that distinguishes patterns.

    Distinct grids (A vs C, A vs D) must NOT be near-collinear. OBSERVED
    cos ~ 0.9999 -> assert > 0.95 (the failure is the evidence).
    """
    e = ComplexNativeIngress(dimension=D, num_blocks=NB, block_dim=BD,
                             device="cpu", band_s=0.10)
    c_ac = _cos_pair(e, GRID_A, GRID_C)
    c_ad = _cos_pair(e, GRID_A, GRID_D)
    # Falsification evidence: distinct grids are indistinguishable.
    assert c_ac > 0.95 and c_ad > 0.95, (
        f"expected collapse evidence, got A,C={c_ac:.4f} A,D={c_ad:.4f}")


def test_g3_legacy_control_discriminates():
    """Blueprint-premise control: the LEGACY production encoder with
    incommensurate basis + bg_mask DISCRIMINATES these grids (cos ~ 0.005).
    This falsifies the blueprint's blanket 'legacy real encoder corrupt'
    claim and shows the real degeneracy is the DEFAULT collinear basis.
    """
    from henri_vision_encoder import HENRIVisionEncoder
    enc = HENRIVisionEncoder(d_model=D, k_blocks=NB, block_dim=BD, device="cpu",
                             spatial_basis_kind="incommensurate", bg_mask=True)
    wa = enc.encode_spatial_grid(GRID_A).reshape(-1)
    wc = enc.encode_spatial_grid(GRID_C).reshape(-1)
    wd = enc.encode_spatial_grid(GRID_D).reshape(-1)

    def cos(a, b):
        return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))

    c_ac, c_ad = cos(wa, wc), cos(wa, wd)
    assert c_ac < 0.10 and c_ad < 0.10, (
        f"legacy control expected <0.10, got A,C={c_ac:.4f} A,D={c_ad:.4f}")


def test_g4_unit_modulus_invariant_holds():
    """Module invariant: per-element unit modulus preserved (FHRR)."""
    e = ComplexNativeIngress(dimension=D, num_blocks=NB, block_dim=BD,
                             device="cpu", band_s=0.10)
    z = e.encode_grid(GRID_A)
    assert torch.allclose(z.abs(), torch.ones_like(z.abs()), atol=1e-5)
