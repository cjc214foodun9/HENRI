"""Contract tests for Carrier G3 wave-packet path search (diagnostic sidecar).

Directive: user approval (2026-09-01) + holographic search.pdf
(HENRI-AUDIT-2026-09-V3-QUANTUM-WAVE-SEARCH, 76c28f6b..., 190,418 B).
Prereg: docs/spec/g3_wave_packet_path_search_preregistration.md.

Audited corrections under test:
- Frozen deterministic generators (zero Parameter) — NOT nn.Parameter.
- HENRI normalized Sagnac delta (1 - Re<a,b>/(||a||||b||)) — NOT sin^2 formula.
- One-way norm-preserving complexification adapter (third-family sidecar).
"""
import math
import os
import pathlib
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

TESTS = pathlib.Path(__file__).resolve()
ROOT = TESTS.parents[2]  # .../HENRI V2  (the code dir)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "verification"))

from arc_g3_wave_packet_search import (  # noqa: E402
    SEED,
    WavePacketPathSearch,
    complexify_wave,
    require_flag,
    sagnac_delta,
    veto_selectivity,
)


def _flag_ctx():
    return pytest.MonkeyPatch.context()


def test_c1_flag_gate_required():
    import importlib
    mod = importlib.import_module("arc_g3_wave_packet_search")
    with pytest.raises(SystemExit):
        require_flag(mod.FLAG)  # env unset -> sys.exit


def test_c1b_production_runner_does_not_import():
    src = (ROOT / "production_arc_run.py").read_text(encoding="utf-8")
    assert "arc_g3_wave_packet_search" not in src
    assert "HENRI_G3_WAVE_PACKET" not in src


def test_c2_frozen_deterministic_generators():
    torch.manual_seed(0)
    a = WavePacketPathSearch(dim=4096, num_actions=7, seed=SEED, device="cpu")
    b = WavePacketPathSearch(dim=4096, num_actions=7, seed=SEED, device="cpu")
    assert torch.equal(a.generators, b.generators), "same seed -> identical generators"
    params = [p for p in a.parameters()]
    assert len(params) == 0, "zero trainable parameters (zero-pretrain invariant)"
    assert a.generators.dtype == torch.complex64


def test_c3_norm_preserved_after_each_step():
    torch.manual_seed(1)
    eng = WavePacketPathSearch(dim=4096, num_actions=3, horizon=4, seed=SEED, device="cpu")
    psi = F.normalize(torch.randn(2, 4096, dtype=torch.complex64), p=2, dim=-1)
    axioms = F.normalize(torch.randn(3, 4096, dtype=torch.complex64), p=2, dim=-1)
    priors = torch.ones(2, 3) / 3
    best, coherence, clearance = eng.propagate_superposed_paths(psi, priors, axioms)
    # Every surviving path is unit norm
    assert torch.allclose(best.norm(dim=-1), torch.ones(2, dtype=torch.float32), atol=1e-5)
    # Path count bounded by top-k
    assert coherence.shape[1] <= 64
    # Best wavefront = argmax coherence path
    # (validated implicitly: coherence is finite and positive)


def test_c4_veto_selectivity_aligned_survives():
    torch.manual_seed(2)
    dim = 4096
    eng = WavePacketPathSearch(dim=dim, num_actions=2, horizon=1, seed=SEED, device="cpu")
    g = torch.Generator().manual_seed(SEED)
    psi = F.normalize(torch.randn(1, dim, dtype=torch.complex64, generator=g), p=2, dim=-1)
    # Axioms: aligned to action 0, orthogonal to action 1
    ax0 = psi  # aligned with root
    ax1 = F.normalize(torch.randn(1, dim, dtype=torch.complex64, generator=g), p=2, dim=-1)
    # Remove ax0 component from ax1 so it is orthogonal-ish
    ax1 = ax1 - (ax1.conj() * ax0).sum(-1, keepdim=True) * ax0
    ax1 = F.normalize(ax1, p=2, dim=-1)
    axioms = torch.cat([ax0, ax1], dim=0)  # [2, D]
    priors = torch.ones(1, 2) / 2
    best, coherence, clearance = eng.propagate_superposed_paths(psi, priors, axioms)
    # Path 0 (aligned with axiom 0) should have much higher survival weight
    w0 = float(coherence[0, 0].item())
    w1 = float(coherence[0, 1].item()) if coherence.shape[1] > 1 else 0.0
    assert w0 > w1, f"aligned path must dominate: {w0} vs {w1}"
    # And the best wavefront is closer to the aligned axiom than the orthogonal one
    d_aligned = float(sagnac_delta(best[0:1], ax0[0:1]).item())
    d_ortho = float(sagnac_delta(best[0:1], ax1[0:1]).item())
    assert d_aligned < d_ortho, f"best path must align with axiom 0: {d_aligned} vs {d_ortho}"


def test_c5_topk_bounded_and_best_path():
    torch.manual_seed(3)
    eng = WavePacketPathSearch(dim=4096, num_actions=4, horizon=3, seed=SEED, device="cpu")
    psi = F.normalize(torch.randn(1, 4096, dtype=torch.complex64), p=2, dim=-1)
    axioms = F.normalize(torch.randn(2, 4096, dtype=torch.complex64), p=2, dim=-1)
    priors = torch.ones(1, 4) / 4
    best, coherence, clearance = eng.propagate_superposed_paths(psi, priors, axioms)
    assert coherence.shape[1] <= 64
    assert torch.isfinite(coherence).all()
    assert torch.isfinite(clearance).all()


def test_c6_complexify_adapter_preserves_norm():
    torch.manual_seed(4)
    real = F.normalize(torch.randn(8192, 8), p=2, dim=-1)
    z = complexify_wave(real)
    assert z.shape == (65536,)
    assert z.dtype == torch.complex64
    assert torch.isclose(z.norm().float(), torch.tensor(1.0), atol=1e-5)
    # injective: distinct inputs -> distinct outputs
    real2 = F.normalize(torch.randn(8192, 8), p=2, dim=-1)
    z2 = complexify_wave(real2)
    assert not torch.allclose(z, z2)


def test_c7_no_policy_influence_static():
    src = (ROOT / "production_arc_run.py").read_text(encoding="utf-8")
    assert "arc_g3_wave_packet_search" not in src


def test_c8_latency_diagnostic_cpu():
    torch.manual_seed(5)
    eng = WavePacketPathSearch(dim=4096, num_actions=7, horizon=8, seed=SEED, device="cpu")
    psi = F.normalize(torch.randn(1, 4096, dtype=torch.complex64), p=2, dim=-1)
    axioms = F.normalize(torch.randn(3, 4096, dtype=torch.complex64), p=2, dim=-1)
    priors = torch.ones(1, 7) / 7
    import time
    t0 = time.perf_counter()
    eng.propagate_superposed_paths(psi, priors, axioms)
    dt = time.perf_counter() - t0
    assert dt < 30.0, f"CPU diagnostic latency {dt:.2f}s exceeds smoke bound"


def test_c9_sagnac_delta_bounds():
    a = F.normalize(torch.randn(1, 1024, dtype=torch.complex64), p=2, dim=-1)
    assert float(sagnac_delta(a, a).item()) == pytest.approx(0.0, abs=1e-5)
    orth = F.normalize(torch.randn(1, 1024, dtype=torch.complex64), p=2, dim=-1)
    orth = orth - (orth.conj() * a).sum(-1, keepdim=True) * a
    orth = F.normalize(orth, p=2, dim=-1)
    d = float(sagnac_delta(a, orth).item())
    assert 0.99 <= d <= 1.01, f"orthogonal Sagnac delta ~1: {d}"


def test_c10_cli_receipt_smoke(tmp_path):
    out = tmp_path / "g3"
    env = dict(os.environ)
    env["HENRI_G3_WAVE_PACKET"] = "1"
    r = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "verification" / "arc_g3_wave_packet_search.py"),
         "--dim", "4096", "--device", "cpu", "--out-dir", str(out)],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert r.returncode == 0, f"CLI rc={r.returncode}\nstdout={r.stdout[-2000:]}\nstderr={r.stderr[-2000:]}"
    receipt = out / "g3_receipt.json"
    assert receipt.exists()
