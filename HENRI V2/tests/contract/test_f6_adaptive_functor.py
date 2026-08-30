"""Carrier F6 contract tests (RED first): per-task adaptive functor compilation.

Spec: HENRI-SPEC-2026-08-F6-ADAPTIVE-FUNCTOR sections 2, 3, 6.
Kernels: f6_adaptive_functor.py (default-OFF, HENRI_F6_FUNCTOR=1).
"""
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

from f6_adaptive_functor import (
    AdaptiveFunctorCompiler,
    compile_adaptive_functor,
    deocclusion_mask,
    spectral_newton_schulz,
)
from arc_task_functor import compile_task_functor


def _rand_unit_modulus(D: int, seed: int, n: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    ph = torch.rand(n, D, generator=g) * 2.0 * np.pi
    return torch.exp(1j * ph)


def _rand_unitary_operator(D: int, seed: int) -> torch.Tensor:
    """A genuinely unitary circulant operator: unit-modulus DFT spectrum.

    W_true = ifft(exp(i*phi_b)) with iid random phases => |W_hat_b| = 1 for all
    bins, so the NS polar factor recovers W_true exactly (fidelity 1.0).
    A random-phase TIME-domain wave is NOT a unitary operator (spread spectrum,
    polar factor reconstructs only ~0.886) and is the wrong fixture for G2.
    """
    g = torch.Generator().manual_seed(seed)
    ph = torch.rand(D, generator=g) * 2.0 * np.pi
    return torch.fft.ifft(torch.exp(1j * ph))


def _ns_err(w: torch.Tensor) -> float:
    lam = torch.fft.fft(w)
    return float(torch.sqrt(((torch.abs(lam) ** 2 - 1.0).square().sum())).item())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    num = float(torch.abs(torch.vdot(a, b)).item())
    den = float(a.norm().item()) * float(b.norm().item()) + 1e-12
    return num / den


class TestSpectralNewtonSchulz:
    def test_c1_convergence_gate_g1(self):
        """C1: 5+ NS iterations converge to unitary polar factor (Gate G1 <= 1e-5)."""
        w = _rand_unit_modulus(512, seed=11)[0]
        w_out, err, iters = spectral_newton_schulz(w, max_iters=32, tol=1e-5)
        assert iters <= 32
        assert err <= 1e-5, f"spectral err {err} > 1e-5"
        assert _ns_err(w_out) <= 1e-5

    def test_c2_dense_vs_spectral_equivalence(self):
        """C2: spectral NS == dense circulant NS at toy D (disclosed correction).

        Both must start from the SAME spectral-norm normalization
        (W_0 = K / ||K||_2), otherwise the raw DFT bins |lambda| ~ sqrt(D)
        diverge under the scalar map.
        """
        D = 64
        w = torch.randn(D, dtype=torch.complex64)
        w = w / (w.norm() + 1e-12)
        K = torch.zeros(D, D, dtype=torch.complex64)
        for i in range(D):
            for j in range(D):
                K[i, j] = w[(i - j) % D]
        spec_norm = float(torch.fft.fft(w).abs().max().item())
        Wd = K / spec_norm
        for _ in range(5):
            Wd = 1.5 * Wd - 0.5 * (Wd @ Wd.conj().T @ Wd)
        w_dense = Wd[:, 0]
        w_spec, _, _ = spectral_newton_schulz(w, max_iters=5, tol=0.0)
        rel = float((w_dense - w_spec).norm() / (w_dense.norm() + 1e-12))
        assert rel <= 1e-4, f"dense vs spectral rel err {rel} > 1e-4"


class TestCompileAndDeocclusion:
    def test_c3_reconstruction_fidelity(self):
        """C3: Gate G2 — demo reconstruction fidelity >= 0.90 (synthetic
        Y = X (x) W_true with W_true a UNITARY circulant operator)."""
        D = 512
        w_true = _rand_unitary_operator(D, seed=22)
        X = _rand_unit_modulus(D, seed=23, n=6)
        Y = X * w_true
        W, mask, err, iters, fid = compile_adaptive_functor(
            X, Y, max_iters=32, tol=1e-5, eps_floor=1e-3)
        assert err <= 1e-5
        assert fid >= 0.90, f"recon fidelity {fid} < 0.90"
        # held-out style retrieval on the first demo pair
        assert _cos(W * X[0], Y[0]) >= 0.90

    def test_c4_deocclusion_mask(self):
        """C4: constant-phase bins shunted, informative bins kept."""
        D, M = 256, 10
        ph = torch.zeros(M, D)
        ph[:, :128] = 1.0  # constant phase (uninformative background)
        ph[:, 128:] = torch.rand(M, 128) * 2.0 * np.pi
        X = torch.exp(1j * ph)
        mask = deocclusion_mask(X, eps_floor=1e-3)
        assert not mask[:128].any(), "constant-phase bins must be masked"
        assert mask[128:].all(), "informative bins must be kept"

    def test_c5_per_task_discrimination(self):
        """C5: the occlusion claim — per-task functor discriminates; pooled global
        functor collapses. Per-task margin > 0.3; pooled margin <= 0.1."""
        D = 512
        wA = _rand_unitary_operator(D, seed=31)
        wB = _rand_unitary_operator(D, seed=32)
        XA = _rand_unit_modulus(D, seed=33, n=4)
        YA = XA * wA
        XB = _rand_unit_modulus(D, seed=34, n=4)
        YB = XB * wB
        WA, _, _, _, _ = compile_adaptive_functor(XA, YA, max_iters=32, tol=1e-5)
        WB, _, _, _, _ = compile_adaptive_functor(XB, YB, max_iters=32, tol=1e-5)
        qA, qB = XA[0], XB[0]
        sAA, sAB = _cos(WA * qA, YA[0]), _cos(WA * qB, YB[0])
        sBB, sBA = _cos(WB * qB, YB[0]), _cos(WB * qA, YA[0])
        margin_per_task = min(sAA - sAB, sBB - sBA)
        Xp = torch.cat([XA, XB])
        Yp = torch.cat([YA, YB])
        Wg, _, _, _, _ = compile_adaptive_functor(Xp, Yp, max_iters=32, tol=1e-5)
        sGA, sGB = _cos(Wg * qA, YA[0]), _cos(Wg * qB, YB[0])
        margin_global = abs(sGA - sGB)
        assert margin_per_task > 0.3, f"per-task margin {margin_per_task} <= 0.3"
        assert margin_global <= 0.1, f"pooled margin {margin_global} > 0.1 (no occlusion collapse)"

    def test_c6_g4_arithmetic(self):
        """C6: Gate G4 arithmetic — F6 threshold = F5 sealed 0.4352 + 0.3000 = 0.7352."""
        assert round(0.4352 + 0.3000, 4) == 0.7352


class TestDefaultOff:
    def test_c7_differential(self):
        """C7: default path byte-identical (flag unset reproduces the captured
        pre-wiring baseline constants); flag set engages a DIFFERENT operator
        (dead-flag guard: the flag reaches a computational consumer)."""
        D = 128

        class MockTok:
            def __init__(self, D: int):
                self.D = D

            def encode_spatial_grid(self, grid):
                g = torch.Generator().manual_seed(99)
                return torch.rand(1, self.D, generator=g)

        tok = MockTok(D)
        pairs = [
            (np.zeros((2, 2), dtype=np.int64), np.ones((2, 2), dtype=np.int64)),
            (np.ones((2, 2), dtype=np.int64), np.zeros((2, 2), dtype=np.int64)),
            (np.full((2, 2), 2, dtype=np.int64), np.full((2, 2), 3, dtype=np.int64)),
            (np.full((2, 2), 3, dtype=np.int64), np.full((2, 2), 2, dtype=np.int64)),
        ]
        os.environ.pop("HENRI_F6_FUNCTOR", None)
        r1 = compile_task_functor(pairs, tok, device="cpu", task_id="t")
        # Baseline captured pre-wiring (f6_capture_legacy_baseline.py, 4 pairs).
        assert r1.w_task_sha256 == "76972f15fdc5520a81087892aa4c95edd96955d8d262e16a2024f54a5e310d08", \
            "default path drifted from captured pre-wiring baseline"
        assert r1.status == "FUNCTOR_FALSIFIED"
        assert abs(r1.held_out_cos - 0.9297219514846802) < 1e-9
        assert abs(r1.identity_cos - 0.9999999403953552) < 1e-9
        os.environ["HENRI_F6_FUNCTOR"] = "1"
        try:
            r2 = compile_task_functor(pairs, tok, device="cpu", task_id="t")
        finally:
            os.environ.pop("HENRI_F6_FUNCTOR", None)
        assert r2.w_task_sha256 != r1.w_task_sha256, \
            "F6 flag did not change the operator (dead flag)"


class TestSplitGuard:
    def test_c8_consumed_guard(self, tmp_path):
        """C8: F6 split loader refuses consumed f5-split-seal.v1 receipts."""
        verif = str(Path(_root) / "experiments" / "verification")
        if verif not in sys.path:
            sys.path.insert(0, verif)
        from f6_adaptive_functor_gates import load_sealed_folds
        fake = {
            "schema_id": "f5-split-seal.v1",
            "single_use": True,
            "n_folds": 4,
            "seed": 20260831,
            "envs": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"],
            "folds": {
                f"fold{i}": {"heldout_envs": [], "train_envs": [], "n_heldout": 0, "n_train": 0}
                for i in range(4)
            },
            "split_rule": "x",
            "fold_manifest_sha256": "0" * 64,
        }
        p = tmp_path / "consumed.json"
        p.write_text(json.dumps(fake))
        with pytest.raises(AssertionError):
            load_sealed_folds(str(p))
