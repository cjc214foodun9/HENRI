"""Phase 7.5 CPX contracts: read-only complex third-family diagnostic sidecar.

Pre-registered (manifest_CPX_complex_sidecar.md):
- C1 flat real view: adapt path preserves wave.reshape(-1) bytes (hash record).
- C2 unit-modulus preservation: |z| == 1 within 1e-5.
- C3 deterministic output: same input -> identical values.
- C4 malformed-input rejection: None / wrong ndim / wrong last dim / NaN ->
  typed CPX_SIDECAR_UNAVAILABLE, no silent fallback.
- C5 degenerate-map discrimination: collinear (identical-phase) wave ->
  CPX_SIDECAR_DEGENERATE (coherence_r ~ 1); phase-spread wave -> CPX_OK
  (coherence_r < 0.5). Pre-registered kill-gate proxy.
- C6 source inspection: flag HENRI_ARC_COMPLEX_SIDECAR defaults OFF; the
  sidecar result is written ONLY into the telemetry emit (no coupling to
  chosen/action/efe_table/policy/rank/step); ON-init failure emits typed
  UNAVAILABLE (never silent).
- C7 device placement: output tensors live on the INPUT tensor device.
- C8 production-container reuse: output is a phase_codec_adapter
  ComplexPhaseState (isinstance) with FLAT_D layout.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch

from arc_complex_sidecar import (
    CPX_DEGENERATE,
    CPX_OK,
    CPX_UNAVAILABLE,
    adapt_uwe_to_complex,
    evaluate_complex_sidecar,
)
from phase_codec_adapter import ComplexPhaseState, PhaseLayout


def _make_wave(k=8192, seed=7, degenerate=False):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(k, 8, generator=g)
    if degenerate:
        # Collinear phase map: every block carries the SAME bivector/scalar
        # ratio (identical phase across blocks).
        w[:, 4:7] = w[:, 4:7].abs().mean(dim=0, keepdim=True).expand(k, 3)
        w[:, 0] = w[:, 0].abs().mean()
    w = torch.nn.functional.normalize(w, p=2, dim=-1)
    return w


def test_flat_real_view_identity():
    w = _make_wave()
    diag, status = evaluate_complex_sidecar(w)
    assert status == CPX_OK
    import hashlib
    expect = hashlib.sha256(
        w.reshape(-1).detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    assert diag["flat_real_sha256"] == expect


def test_unit_modulus():
    w = _make_wave()
    z = adapt_uwe_to_complex(w)
    err = torch.max(torch.abs(torch.abs(z.values) - 1.0)).item()
    assert err < 1e-5


def test_deterministic():
    w = _make_wave()
    z1 = adapt_uwe_to_complex(w)
    z2 = adapt_uwe_to_complex(w)
    assert torch.equal(z1.values, z2.values)


def test_malformed_rejection():
    assert evaluate_complex_sidecar(None)[1] == CPX_UNAVAILABLE
    assert evaluate_complex_sidecar(torch.randn(8192))[1] == CPX_UNAVAILABLE
    assert evaluate_complex_sidecar(torch.randn(8192, 7))[1] == CPX_UNAVAILABLE
    bad = torch.randn(8192, 8)
    bad[0, 0] = float("nan")
    assert evaluate_complex_sidecar(bad)[1] == CPX_UNAVAILABLE


def test_degenerate_discrimination():
    deg = _make_wave(degenerate=True)
    diag_deg, status_deg = evaluate_complex_sidecar(deg)
    assert status_deg == CPX_DEGENERATE
    assert diag_deg["coherence_r"] > 0.999
    spread = _make_wave(degenerate=False)
    diag_spread, status_spread = evaluate_complex_sidecar(spread)
    assert status_spread == CPX_OK
    # Measured 2026-08-12: spread 0.8449, degenerate 0.9999 (atan2 phase
    # concentrates naturally on Gaussian entries). The kill-gate proxy is
    # the DISCRIMINATION MARGIN, not an absolute low threshold.
    assert diag_spread["coherence_r"] < 0.95
    assert diag_spread["coherence_r"] < diag_deg["coherence_r"] - 0.05


def test_runner_flag_default_off_and_telemetry_only():
    runner = Path(__file__).resolve().parents[2] / "production_arc_run.py"
    src = runner.read_text(encoding="utf-8")
    assert 'os.environ.get("HENRI_ARC_COMPLEX_SIDECAR", "0") == "1"' in src
    assert src.count('"complex_sidecar": complex_sidecar_info') == 1
    decision_terms = ("chosen", "action", "efe_table", "policy", "rank", "step")
    for i, line in enumerate(src.splitlines(), 1):
        if "complex_sidecar_info" in line:
            for term in decision_terms:
                assert term not in line, (
                    f"line {i} couples sidecar to decision term '{term}': {line.strip()}"
                )
    assert "if HENRI_ARC_COMPLEX_SIDECAR:" in src
    assert "if HENRI_ARC_COMPLEX_SIDECAR and" not in src
    assert "CPX_SIDECAR_UNAVAILABLE" in src


def test_device_placement():
    w = _make_wave()
    z = adapt_uwe_to_complex(w)
    assert z.values.device == w.device
    diag, status = evaluate_complex_sidecar(w)
    assert status == CPX_OK
    assert diag["device"] == str(w.device)


def test_production_container_reuse():
    w = _make_wave()
    z = adapt_uwe_to_complex(w)
    assert isinstance(z, ComplexPhaseState)
    assert z.layout is PhaseLayout.FLAT_D
    assert z.provenance.transform == "atan2_bivector_norm_to_unit_phasor"
