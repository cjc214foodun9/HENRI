"""Phase 8.16.1 contract tests — Shared-Projection Calibration & Gauge Realignment.

Spec: HENRI-SPEC-2026-08-PHASE8.16.1-REFORM (SHA bdf4602b...).
Gates (pre-registered in phase8161_shared_projection_design.md):
  G1-8.16.1: L_valid < 1e-4
  G2-8.16.1: ratio L_mism / max(L_valid, 1e-12) >= 10.0
  G3-8.16.1: Triton LUT <= 50 us (remote CUDA runner; local CPU skip)
  G4-8.16.1: top-1 codebook recall >= 0.99 (remote CUDA runner; local CPU path)
Spec claimed constants L_valid 2.7e-5 / L_mism 3.2e-4 are recorded metrics,
NOT gate thresholds (deviation D16 — gates unchanged).
"""
import os
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from henri_functor_flow import DiagrammaticSharedEgressEvaluator  # noqa: E402
from qfhrr_kernels import wave_to_phase_codes, phase_codes_to_wave  # noqa: E402

D, NB, LAT = 4096, 512, 128
S = 5.0e-6
K = 256


def _rand_codes(n, seed):
    return torch.randint(
        0, 256, (n, NB, 4),
        generator=torch.Generator().manual_seed(seed),
    ).to(torch.uint8)


def _codes_to_wave(q):
    n, nb, _ = q.shape
    return phase_codes_to_wave(q.reshape(n * nb, 4)).reshape(n, nb, 8)


def _flat(w):
    return w.reshape(w.shape[0], -1)


@pytest.fixture()
def evaluator():
    return DiagrammaticSharedEgressEvaluator(dim=D, latent_dim=LAT, scale=S)


@pytest.fixture()
def data():
    q1 = _rand_codes(K, 1)
    w1 = _codes_to_wave(q1)
    q1_rt = wave_to_phase_codes(w1.reshape(K * NB, 8)).reshape(K, NB, 4)
    w1_rt = _codes_to_wave(q1_rt)
    w2 = _codes_to_wave(_rand_codes(K, 2))
    return w1, w1_rt, w2


def test_scale_pinned_by_construction(evaluator):
    """Spectral scale pinned via orthogonal init * scale (spec 2.1)."""
    with torch.no_grad():
        sv = torch.linalg.svdvals(evaluator.eta_shared.weight)
        s_obs = sv[0].item()
    assert abs(s_obs - S) < S * 1e-5  # float32 SVD tolerance


def test_g1_valid_obstruction_below_gate(evaluator, data):
    """G1-8.16.1: valid aligned pairs < 1e-4 (codebook round-trip pairs)."""
    w1, w1_rt, _ = data
    with torch.no_grad():
        l_valid = evaluator(_flat(w1), _flat(w1_rt)).item()
    assert l_valid < 1e-4


def test_g2_discrimination_ratio(evaluator, data):
    """G2-8.16.1: ratio >= 10 (denominator clamp max(L_valid, 1e-12))."""
    w1, w1_rt, w2 = data
    with torch.no_grad():
        l_valid = evaluator(_flat(w1), _flat(w1_rt)).item()
        l_mism = evaluator(_flat(w1), _flat(w2)).item()
    ratio = l_mism / max(l_valid, 1e-12)
    assert ratio >= 10.0


def test_spec_claimed_constants_recorded(evaluator, data):
    """Recorded metrics (deviation D16): actual values at s=5e-6, not spec claims."""
    w1, w1_rt, w2 = data
    with torch.no_grad():
        l_valid = evaluator(_flat(w1), _flat(w1_rt)).item()
        l_mism = evaluator(_flat(w1), _flat(w2)).item()
    # Spec claims 2.7e-5 / 3.2e-4 — actual are far smaller (s^2*k form).
    assert l_valid <= 1e-5
    assert abs(l_mism - (S * S * LAT)) < 1e-10


def test_codebook_roundtrip_exact(data):
    """Phase-ring codec round-trip is lossless for codebook waves (d_rt = 0)."""
    w1, w1_rt, _ = data
    d = (w1 - w1_rt).norm(dim=(-1, -2)).square().mean().item()
    assert d < 1e-12


def test_default_off_no_production_wiring():
    """Evaluator is additive/diagnostic-only: no production caller references it."""
    prods = [
        REPO_ROOT / "HENRI V2" / "production_arc_run.py",
        REPO_ROOT / "HENRI V2" / "efe_planner.py",
        REPO_ROOT / "HENRI V2" / "henri_decoder.py",
    ]
    for p in prods:
        assert "DiagrammaticSharedEgressEvaluator" not in p.read_text(encoding="utf-8")
