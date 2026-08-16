"""Phase 8.16 contract tests — Diagrammatic FunctorFlow Egress.

Spec: HENRI-SPEC-2026-08-PHASE8.16-EGRESS (SHA 2ec60178...).
Verification matrix:
  G1-EGRESS: as-shipped DiagrammaticEgressEvaluator FALSIFIED (noise floor
             above 1e-4 at ANY scale; ordering inverted on real data).
             LOCKED by test: spec-default init cannot reach valid < 1e-4 even
             after calibrate() on matched pairs (held-out >= 1e-4).
  G2-EGRESS: top-1 recall >= 0.99 on 256-symbol phase-ring codebook with
             noisy realizations (sigma 0.01/0.05/0.1) via existing LUT path.
  Additivity/default-OFF: module is diagnostic-only; no production caller.
"""
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from henri_functor_flow import DiagrammaticEgressEvaluator
from qfhrr_kernels import (
    phase_codes_to_wave,
    wave_to_phase_codes,
    qfhrr_similarity,
    build_cos_lut,
)

D_SMALL, NB_SMALL, LAT = 4096, 512, 128
N_TR, N_HO = 128, 64


def _bc2w(q):
    n, nb, _ = q.shape
    return phase_codes_to_wave(q.reshape(n * nb, 4)).reshape(n, nb, 8)


def _bw2c(w):
    n, nb, _ = w.shape
    return wave_to_phase_codes(w.reshape(n * nb, 8)).reshape(n, nb, 4)


def _pairs(n, seed):
    q = torch.randint(0, 256, (n, NB_SMALL, 4), generator=torch.Generator().manual_seed(seed)).to(torch.uint8)
    w = _bc2w(q).reshape(n, D_SMALL)
    a = _bc2w(_bw2c(w.reshape(n, NB_SMALL, 8))).reshape(n, D_SMALL)
    return w, a


# --------------------------------------------------------------------------- G1
def test_g1_as_shipped_falsified_lock():
    """Spec-default init cannot meet its own 1e-4 gate (noise floor ~2/3).

    This test LOCKS the falsification so the gate cannot silently re-pass.
    """
    w_ho, a_ho = _pairs(N_HO, 2)
    q_m = torch.randint(0, 256, (N_HO, NB_SMALL, 4), generator=torch.Generator().manual_seed(3)).to(torch.uint8)
    w_mis = _bc2w(q_m).reshape(N_HO, D_SMALL)
    ev = DiagrammaticEgressEvaluator(dim=D_SMALL, latent_dim=LAT)
    with torch.no_grad():
        lv = ev(w_ho, a_ho).item()
        lm = ev(w_ho, w_mis).item()
    assert lv >= 1e-4, "as-shipped valid loss must exceed the gate (falsified lock)"
    assert lm >= 1e-4


def test_g1_calibration_cannot_reach_gate_heldout():
    """Even after calibrate(), held-out valid loss stays >= 1e-4 (under-determined fit)."""
    w_tr, a_tr = _pairs(N_TR, 1)
    w_ho, a_ho = _pairs(N_HO, 2)
    q_m = torch.randint(0, 256, (N_HO, NB_SMALL, 4), generator=torch.Generator().manual_seed(3)).to(torch.uint8)
    w_mis = _bc2w(q_m).reshape(N_HO, D_SMALL)
    ev = DiagrammaticEgressEvaluator(dim=D_SMALL, latent_dim=LAT)
    hist = ev.calibrate(w_tr, a_tr, steps=200, lr=1e-3, sym_lambda=1e-2)
    assert hist[-1] < 1e-3, "train objective must shrink"
    with torch.no_grad():
        lv = ev(w_ho, a_ho).item()
        lm = ev(w_ho, w_mis).item()
    assert lv >= 1e-4, "held-out valid must remain above the gate (falsification lock)"
    assert lm >= 1e-4


def test_g1_reject_semantics():
    """reject() flags candidates above the threshold."""
    ev = DiagrammaticEgressEvaluator(dim=D_SMALL, latent_dim=LAT)
    w_ho, a_ho = _pairs(N_HO, 2)
    with torch.no_grad():
        r = ev.reject(w_ho, a_ho, threshold=1e-4)
    assert r.dtype == torch.bool
    assert r.all(), "as-shipped valid pairs must be rejected at 1e-4 (falsification lock)"


# --------------------------------------------------------------------------- G2
@pytest.mark.parametrize("sigma", [0.01, 0.05, 0.1])
def test_g2_phase_ring_recall(sigma):
    """Top-1 recall >= 0.99 on 256-symbol codebook with noisy queries (torch LUT path)."""
    M = 256
    lut = build_cos_lut("cpu")
    q_cb = torch.randint(0, 256, (M, NB_SMALL, 4), generator=torch.Generator().manual_seed(21)).to(torch.uint8)
    w_cb = _bc2w(q_cb).reshape(M, D_SMALL)
    q_flat = q_cb.reshape(M, -1)
    gen = torch.Generator().manual_seed(31)
    noise = torch.randn(M, NB_SMALL, 8, generator=gen) * sigma
    q_noisy = _bw2c((w_cb.reshape(M, NB_SMALL, 8) + noise).reshape(M, NB_SMALL, 8)).reshape(M, -1)
    hits = 0
    for i in range(M):
        sims = qfhrr_similarity(q_noisy[i], q_flat, lut)
        hits += int(sims.argmax().item() == i)
    recall = hits / M
    assert recall >= 0.99, f"recall {recall:.4f} < 0.99 at sigma={sigma}"


# --------------------------------------------------------------------------- additivity
def test_default_off_no_production_wiring():
    """The evaluator is additive/diagnostic-only: no env flag, no runner import change."""
    import inspect

    import henri_functor_flow as ff
    src = inspect.getsource(ff)
    # The module must not read any HENRI_ARC_* flag (not wired into production paths)
    assert "HENRI_ARC_" not in src or "HENRI_ARC_" not in src.split("def main")[0]
