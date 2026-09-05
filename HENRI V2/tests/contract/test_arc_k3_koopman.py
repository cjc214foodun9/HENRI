"""Contract tests — Carrier K3 empirical block-Koopman transition generator.

Prereg: docs/spec/carrier_k3_empirical_koopman_preregistration.md (sealed
2026-09-03). Supplied prereg + kernel SHAs are asserted byte-identical from
their staged copies (docs/spec/carrier_k3_supplied_prereg.md,
HENRI V2/experiments/verification/carrier_k3_supplied_kernel.py).

Covers: fail-closed flag, supplied-artifact byte identity, ring accumulator
causality (held-out rows are the newest and never enter the fit), ridge fit
recovery of a planted block map, KG1 held-out error on the recovery fixture,
KG3 operator separation, KG4 contractive projection engagement (a planted
expansive block is scaled so post sigma <= 1.0), the disclosed screen-quality
assumption (est >= 0.95 * exact sigma outside the exact top set), counted
pinv fallback, NaN fail-closed abort, launcher seam wiring (default G7 path
preserved; K3 takes precedence over the sealed-FALSIFIED C1), and
runner-independence (no production_arc_run import).
Core math tests run on CPU; full-scale CUDA equivalence is gated by CUDA
availability (remote gate owns the verdict).
"""

import hashlib
import os
import pathlib
import sys

import numpy as np
import pytest
import torch

TESTS = pathlib.Path(__file__).resolve()
ROOT = TESTS.parents[2]  # <repo>/HENRI V2
VERIF = ROOT / "experiments" / "verification"
for _p in (str(ROOT), str(VERIF)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from arc_k3_koopman_generator import (  # noqa: E402
    K3_FLAG,
    K3_M,
    K3_D,
    K3_ALPHA,
    K3_RING_CAP,
    KFIT_MIN_N,
    BlockRidgeKoopmanFit,
    K3NumericalAbort,
    K3RingAccumulator,
    _screen_sigma_max,
    require_k3_flag,
)

SEED = 20260930
N_BLOCKS = K3_M          # full scale: 8192 blocks (CPU-capable)
D_BLOCK = K3_D           # 8
SUPPLIED_PREREG_SHA = (
    "841ac58159935f8a27cc19f602c357bf4c612dba934ed01dedd462c060543ecc")
SUPPLIED_KERNEL_SHA = (
    "bff0174955e5eea7d22be222c4a8056f2e04d02ad077de272776a7bf8ce66e4e")


def _sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _unit_rows(n_blocks: int, d_block: int, seed: int, n: int = 1):
    """[n, n_blocks, d_block] unit rows."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, n_blocks, d_block, generator=g)
    return x / x.norm(p=2, dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Fail-closed flag + supplied-artifact byte identity
# ---------------------------------------------------------------------------

def test_flag_fail_closed():
    os.environ.pop(K3_FLAG, None)
    with pytest.raises(RuntimeError):
        require_k3_flag()
    os.environ[K3_FLAG] = "1"
    try:
        require_k3_flag()  # must not raise
    finally:
        os.environ.pop(K3_FLAG, None)


def test_flag_string_pinned():
    assert K3_FLAG == "HENRI_K3_KOOPMAN"


def test_supplied_prereg_byte_identity():
    staged = ROOT / ".." / "docs" / "spec" / "carrier_k3_supplied_prereg.md"
    assert staged.is_file(), "staged supplied prereg missing"
    assert _sha256(staged) == SUPPLIED_PREREG_SHA


def test_supplied_kernel_byte_identity():
    staged = VERIF / "carrier_k3_supplied_kernel.py"
    assert staged.is_file(), "staged supplied kernel missing"
    assert _sha256(staged) == SUPPLIED_KERNEL_SHA


# ---------------------------------------------------------------------------
# Ring accumulator: capacity, ordering, causal split
# ---------------------------------------------------------------------------

def test_ring_push_ordered_and_causal_split():
    ring = K3RingAccumulator(K3_RING_CAP, 4, D_BLOCK, torch.device("cpu"))
    for i in range(10):
        x = torch.randn(4, D_BLOCK)
        y = torch.randn(4, D_BLOCK)
        ring.push(x, y)
    X, Y = ring.ordered()
    assert X.shape == (10, 4, D_BLOCK)
    n_fit, w = ring.fit_eval_split()
    assert n_fit == 8 and w == 2       # newest 2 rows held out
    # Causality: the split is the LAST w rows of arrival order.
    assert n_fit + w == ring.count


def test_ring_cap_wraps_and_preserves_newest():
    ring = K3RingAccumulator(16, 4, D_BLOCK, torch.device("cpu"))
    for i in range(40):
        x = torch.full((4, D_BLOCK), float(i))
        ring.push(x, x + 1.0)
    X, _ = ring.ordered()
    assert ring.count == 16
    # The newest 16 arrivals are rows 24..39 (arrival order preserved).
    assert float(X[0, 0, 0].item()) == 24.0
    assert float(X[-1, 0, 0].item()) == 39.0


def test_ring_rejects_nonfinite():
    ring = K3RingAccumulator(8, 4, D_BLOCK, torch.device("cpu"))
    x = torch.randn(4, D_BLOCK)
    y = torch.randn(4, D_BLOCK)
    y[0, 0] = float("nan")
    with pytest.raises(K3NumericalAbort):
        ring.push(x, y)


# ---------------------------------------------------------------------------
# Ridge fit: planted-map recovery, KG1 held-out error, KG3 separation
# ---------------------------------------------------------------------------

def _planted_data(n_rows, seed, n_blocks=64, d_block=D_BLOCK,
                  expand_scale=1.0, contractive=True):
    """X rows; Y = K_true X + small noise. K_true per-block: I + scale*A_mix.
    With contractive=True the map is renormalized to max sigma 0.9 so ridge
    identification is not distorted by the KG4 projection (the
    projection-fires case is tested separately with contractive=False)."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n_blocks, d_block, d_block, generator=g) * 0.03
    K_true = torch.eye(d_block).unsqueeze(0) + A * expand_scale
    if contractive:
        sig = torch.linalg.svdvals(K_true).max(dim=-1).values.max()
        K_true = K_true * (0.9 / float(sig))
    X = torch.randn(n_rows, n_blocks, d_block, generator=g)
    noise = torch.randn(n_rows, n_blocks, d_block, generator=g) * 1e-3
    Y = torch.einsum("mij,nmj->nmi", K_true, X) + noise
    return X, Y, K_true


def test_ridge_fit_recovers_planted_map_and_kg1():
    X, Y, _ = _planted_data(48, seed=SEED, n_blocks=64)
    fit = BlockRidgeKoopmanFit(alpha=K3_ALPHA)
    n_fit = 40
    res = fit.fit(X, Y, n_fit)
    assert res["K"].shape == (64, D_BLOCK, D_BLOCK)
    assert res["n_fit"] == 40
    assert not res["pinv_fallback"]
    # KG1 held-out relative error over the trailing rows (never in fit).
    err = BlockRidgeKoopmanFit.heldout_error(X[n_fit:], Y[n_fit:], res["K"])
    assert err <= 0.05, f"held-out err {err} > 0.05 on a planted linear map"


def test_kg3_operator_separation_two_actions():
    X, Y1, _ = _planted_data(40, seed=SEED + 1, n_blocks=32)
    _, Y2, _ = _planted_data(40, seed=SEED + 2, n_blocks=32)
    fit = BlockRidgeKoopmanFit(alpha=K3_ALPHA)
    r1 = fit.fit(X, Y1, 32)
    r2 = fit.fit(X, Y2, 32)
    sep = (r1["K"] - r2["K"]).norm(p="fro", dim=(-2, -1)).mean().item()
    assert sep >= 0.05, f"KG3 separation {sep} < 0.05"


# ---------------------------------------------------------------------------
# KG4 contractive projection ENGAGEMENT (non-vacuous)
# ---------------------------------------------------------------------------

def test_projection_fires_and_enforces_bound_on_expansive_map():
    X, Y, _ = _planted_data(48, seed=SEED + 3, n_blocks=64,
                            expand_scale=6.0, contractive=False)
    fit = BlockRidgeKoopmanFit(alpha=K3_ALPHA)
    res = fit.fit(X, Y, 40)
    assert res["fired_blocks"] > 0, (
        "expansive planted map must fire the contractive projection")
    # KG4: post-scale spectral max over fired blocks <= 1.0 + tol.
    assert res["sigma_post_max"] <= 1.0 + 1e-3, res["sigma_post_max"]
    # Reported raw max exceeds 1.0 (projection actually had work to do).
    assert res["sigma_max"] > 1.0


def test_projection_not_required_for_contractive_map():
    X, Y, _ = _planted_data(48, seed=SEED + 4, n_blocks=64, expand_scale=0.0)
    fit = BlockRidgeKoopmanFit(alpha=K3_ALPHA)
    res = fit.fit(X, Y, 40)
    # K_true = I + tiny noise -> sigma ~ 1.0; projection may or may not fire.
    # The invariant is the post bound, and the raw value is reported.
    assert res["sigma_post_max"] <= 1.0 + 1e-3


def test_screen_estimate_quality_disclosed_assumption():
    """est >= 0.95 * sigma_max outside the exact top set (16 power steps)."""
    g = torch.Generator().manual_seed(SEED + 5)
    K = torch.randn(512, D_BLOCK, D_BLOCK, generator=g)
    # Inflate blocks to various scales to stress the screen (deterministic).
    scales = torch.linspace(0.5, 3.0, 512).unsqueeze(-1).unsqueeze(-1)
    K = K * scales
    est, exact_mask = _screen_sigma_max(K)
    exact = torch.linalg.svdvals(K).max(dim=-1).values
    assert bool((~exact_mask).any().item())
    ratio = (est[~exact_mask] / exact[~exact_mask].clamp(min=1e-12))
    assert float(ratio.min().item()) >= 0.95, (
        f"screen estimate fell below 0.95*sigma: {float(ratio.min().item())}")
    # Exact top set carries exact values.
    assert bool(exact_mask.any().item())
    assert torch.allclose(est[exact_mask], exact[exact_mask], atol=1e-4)


# ---------------------------------------------------------------------------
# Counted defensive pinv fallback (monkeypatched cholesky failure)
# ---------------------------------------------------------------------------

def test_pinv_fallback_counted(monkeypatch):
    X, Y, _ = _planted_data(24, seed=SEED + 6, n_blocks=8)
    fit = BlockRidgeKoopmanFit(alpha=K3_ALPHA)

    def _boom(*a, **k):
        raise torch.linalg.LinAlgError("forced")

    monkeypatch.setattr(torch.linalg, "cholesky", _boom)
    res = fit.fit(X, Y, 16)
    assert res["pinv_fallback"]
    assert fit.pinv_fallbacks == 1
    assert res["K"].shape == (8, D_BLOCK, D_BLOCK)
    assert torch.isfinite(res["K"]).all()


# ---------------------------------------------------------------------------
# Full-scale (8192-block) CPU smoke of the fit path
# ---------------------------------------------------------------------------

def test_full_scale_cpu_fit_smoke():
    g = torch.Generator().manual_seed(SEED + 7)
    n = 16
    X = torch.randn(n, N_BLOCKS, D_BLOCK, generator=g)
    X = X / X.norm(p=2, dim=-1, keepdim=True)
    Y = torch.roll(X, shifts=1, dims=-1)  # deterministic non-trivial map
    fit = BlockRidgeKoopmanFit(alpha=K3_ALPHA)
    res = fit.fit(X, Y, 12)
    assert res["K"].shape == (N_BLOCKS, D_BLOCK, D_BLOCK)
    assert torch.isfinite(res["K"]).all()
    assert res["sigma_post_max"] <= 1.0 + 1e-3


# ---------------------------------------------------------------------------
# Launcher seam wiring + runner independence (C1-mirror source audit)
# ---------------------------------------------------------------------------

LAUNCHER = VERIF / "arc_g7_calibrated_engine.py"


def test_launcher_flag_wiring_present():
    src = LAUNCHER.read_text(encoding="utf-8", errors="replace")
    assert K3_FLAG in src
    assert "K3KoopmanSteeringEngine" in src
    assert "require_k3_flag" in src
    assert 'use_k3 = os.environ.get("HENRI_K3_KOOPMAN") == "1"' in src


def test_launcher_default_path_is_g7():
    src = LAUNCHER.read_text(encoding="utf-8", errors="replace")
    assert "engine_cls = G7CalibratedAffordanceEngine" in src
    assert "use_k3" in src
    # K3 takes precedence over the sealed-FALSIFIED C1.
    k3_idx = src.index("use_k3")
    c1_idx = src.index("elif use_c1:")
    assert k3_idx < c1_idx


def test_engines_never_import_runner():
    for name in ("arc_k3_koopman_generator.py", "arc_k3_steering_engine.py"):
        src = (VERIF / name).read_text(encoding="utf-8", errors="replace")
        assert "production_arc_run" not in src


def test_efe_planner_has_no_k3_seam():
    src = (ROOT / "efe_planner.py").read_text(encoding="utf-8", errors="replace")
    assert "k3_koopman" not in src and "K3Koopman" not in src


# ---------------------------------------------------------------------------
# Full-scale CUDA equivalence + kernel smoke (remote gate; skipped on CPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_full_scale_fit_and_supplied_kernel():
    import importlib.util
    if importlib.util.find_spec("triton") is None:
        pytest.skip("triton not available")
    # Reference fit at full scale on CUDA.
    # CPU generator + device="cuda" raises RuntimeError; generate on CPU with
    # the deterministic generator, then move to CUDA (canonical device pattern).
    g = torch.Generator().manual_seed(SEED + 8)
    n = 32
    X = torch.randn(n, N_BLOCKS, D_BLOCK, generator=g, device="cpu").to("cuda")
    X = X / X.norm(p=2, dim=-1, keepdim=True)
    Y = torch.roll(X, shifts=1, dims=-1)
    fit = BlockRidgeKoopmanFit(alpha=K3_ALPHA)
    res = fit.fit(X, Y, 24)
    assert torch.isfinite(res["K"]).all().item()
    assert res["sigma_post_max"] <= 1.0 + 1e-3
    # Supplied kernel equivalence at a FIXED N (constexpr-N; small M slice is
    # not supported by the hardcoded 8192-stride kernel, so run full M).
    import carrier_k3_supplied_kernel as supplied
    Xc = X.contiguous().view(n, N_BLOCKS, D_BLOCK)
    A = torch.empty((N_BLOCKS, D_BLOCK, D_BLOCK), device="cuda",
                    dtype=torch.float32)
    B = torch.empty((N_BLOCKS, D_BLOCK, D_BLOCK), device="cuda",
                    dtype=torch.float32)
    supplied._block_covariance_accum_kernel[(N_BLOCKS,)](
        Xc.float(), Y.float().contiguous(), A, B,
        N_transitions=n, alpha=K3_ALPHA, BLOCK_D=D_BLOCK)
    A_ref = torch.einsum("nmi,nmj->mij", Xc.float(), Xc.float()) + \
        K3_ALPHA * torch.eye(D_BLOCK, device="cuda")
    B_ref = torch.einsum("nmi,nmj->mij", Y.float(), Xc.float())
    assert torch.allclose(A, A_ref, atol=1e-2), "kernel A != reference A"
    assert torch.allclose(B, B_ref, atol=1e-2), "kernel B != reference B"
