"""Phase 8.18 contract tests — Field-to-Wave Isomorphic Transducer.

Spec: HENRI-SPEC-2026-08-PHASE8.18-TRANSDUCER (SHA 158c02c7...).
Pre-registered gates (experiments/sweeps/phase818_transducer_design.md):
G1 round-trip < 1e-5; G2 non-commutativity > 0.5 (+ commuting control);
C2 default-OFF fail-closed; C3 Triton kernel presence (CUDA-gated).
Deviations D19 (eig log), D20/D21 (einsum corrections) applied in the module.
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "HENRI V2"))

from universal_data_transducer import SU3FieldWaveTransducer  # noqa: E402
import qfhrr_kernels  # noqa: E402


def gell_mann_basis(dtype=torch.complex64):
    b = []
    e = lambda i, j: (lambda m: (m.__setitem__((i, j), torch.tensor(1.0, dtype=dtype)), m)[1])(
        torch.zeros(3, 3, dtype=dtype)
    )
    b.append(e(0, 1) + e(1, 0))
    b.append(-1j * e(0, 1) + 1j * e(1, 0))
    b.append(torch.diag(torch.tensor([1.0, -1.0, 0.0], dtype=dtype)))
    b.append(e(0, 2) + e(2, 0))
    b.append(-1j * e(0, 2) + 1j * e(2, 0))
    b.append(e(1, 2) + e(2, 1))
    b.append(-1j * e(1, 2) + 1j * e(2, 1))
    b.append(torch.diag(torch.tensor([1.0, 1.0, -2.0], dtype=dtype)) / math.sqrt(3.0))
    return torch.stack(b)


def rand_su3(n, basis, theta_scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    theta = (torch.rand(n, 8, generator=g) * 2 - 1) * theta_scale
    alg = 1j * torch.einsum("na,abc->nbc", theta.to(basis.dtype), basis)
    return torch.matrix_exp(alg)


@pytest.fixture(scope="module")
def trans():
    return SU3FieldWaveTransducer(gell_mann_basis())


def test_gell_mann_orthonormality():
    basis = gell_mann_basis()
    prod = torch.einsum("aij,bji->ab", basis, basis)
    expected = 2.0 * torch.eye(8, dtype=torch.complex64)
    assert torch.allclose(prod, expected, atol=1e-5), f"Tr(la lb) != 2 delta: {prod.diag()}"


def test_g1_round_trip_complex128(trans):
    U = rand_su3(512, trans.basis, seed=11)
    w = trans.field_to_wave(U.unsqueeze(0))
    rec = trans.wave_to_field(w)[0]
    err = float((U - rec).norm(dim=(-2, -1)).mean().item())
    assert err < 1e-5, f"G1 c128 round-trip {err:.3e} >= 1e-5"
    assert torch.allclose(torch.abs(w), torch.ones_like(torch.abs(w)), atol=1e-5)


def test_g1_round_trip_complex64(trans):
    U = rand_su3(512, trans.basis, seed=12).to(torch.complex64)
    w = trans.field_to_wave(U.unsqueeze(0))
    rec = trans.wave_to_field(w)[0]
    err = float((U - rec).norm(dim=(-2, -1)).mean().item())
    assert err < 1e-5, f"G1 c64 round-trip {err:.3e} >= 1e-5"


def test_g2_non_commutativity(trans):
    UA = rand_su3(512, trans.basis, seed=21)
    UB = rand_su3(512, trans.basis, seed=22)
    d_nc = float(
        (trans.field_to_wave((UA @ UB).unsqueeze(0))
         - trans.field_to_wave((UB @ UA).unsqueeze(0))).norm().item()
    )
    assert d_nc > 0.5, f"G2 non-comm dist {d_nc:.3f} <= 0.5"


def test_g2_commuting_control(trans):
    # same-axis generators -> commute -> distance ~ 0 (discrimination control)
    g = torch.Generator().manual_seed(31)
    theta = (torch.rand(512, 8, generator=g) * 2 - 1) * 0.5
    alg1 = 1j * torch.einsum("na,abc->nbc", theta.to(trans.basis.dtype), trans.basis)
    U = torch.matrix_exp(alg1)
    U2 = U @ U
    d_com = float(
        (trans.field_to_wave((U @ U2).unsqueeze(0))
         - trans.field_to_wave((U2 @ U).unsqueeze(0))).norm().item()
    )
    assert d_com < 0.05, f"commuting control dist {d_com:.3e} not near zero"


def test_c2_default_off_fail_closed():
    # HENRI_ARC_IN_CONTEXT_ALIGN must default to "0" (no goal bridging without
    # an explicit opt-in).
    assert os.environ.get("HENRI_ARC_IN_CONTEXT_ALIGN", "0") == "0"
    src = (REPO_ROOT / "HENRI V2" / "production_arc_run.py").read_text(encoding="utf-8")
    assert "bridged_goal_wave = None  # Phase 8.18 C2 transducer bridge (default OFF)" in src
    assert "W_TASK_GOAL_BRIDGED" in src


def test_c3_triton_kernel_present():
    if not getattr(qfhrr_kernels, "_HAS_TRITON", False):
        pytest.skip("triton unavailable on this host")
    assert hasattr(qfhrr_kernels, "su3_matrix_log_triton"), "C3 kernel missing"
    assert hasattr(qfhrr_kernels, "_su3_log_kernel"), "C3 triton.jit kernel missing"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_c3_triton_kernel_cuda_correctness():
    basis = gell_mann_basis().to("cuda")
    U = rand_su3(1024, basis, seed=41).to("cuda")
    lg = qfhrr_kernels.su3_matrix_log_triton(U)
    # torch reference via eig on cuda
    evals, evecs = torch.linalg.eig(U)
    ref = evecs @ torch.diag_embed(torch.log(evals)) @ evecs.conj().transpose(-2, -1)
    err = float((lg - ref).abs().norm(dim=(-2, -1)).mean().item())
    assert err < 1e-3, f"Triton log vs eig-log mean err {err:.3e} >= 1e-3"
