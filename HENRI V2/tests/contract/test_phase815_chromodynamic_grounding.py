"""Phase 8.15 contract tests — SU(3) chromodynamic grounding (default-OFF, additive)."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import chromodynamic_grounding as cg  # noqa: E402

torch.manual_seed(20260816)


def _su3_from_theta(theta: torch.Tensor) -> torch.Tensor:
    alg = 1j * torch.einsum("na,abc->nbc", theta.to(torch.complex64), cg.GELL_MANN_BASIS)
    return torch.matrix_exp(alg)


class TestGellMannAlgebra:
    def test_trace_orthonormality(self):
        for a in range(8):
            for b in range(8):
                tr = torch.trace(cg.GELL_MANN_BASIS[a] @ cg.GELL_MANN_BASIS[b])
                assert abs(tr.item() - (2.0 if a == b else 0.0)) < 1e-4

    def test_hermitian_traceless(self):
        for a in range(8):
            m = cg.GELL_MANN_BASIS[a]
            assert (m - m.conj().T).abs().max().item() < 1e-6
            assert abs(torch.trace(m).item()) < 1e-6

    def test_commutation_structure_constants(self):
        f, max_res = cg.structure_constants()
        assert max_res < 1e-3
        assert f.imag.abs().max().item() < 1e-3  # f real
        assert f[0, 1, 2].real.item() == pytest.approx(1.0, abs=1e-3)  # [l1,l2] = 2i l3
        assert (f + f.transpose(0, 1)).abs().max().item() < 1e-3  # antisymmetric


class TestColorBinding:
    def test_encode_shape_and_unitarity(self):
        grid = torch.tensor([[[0, 1], [2, 3]]])
        u = cg.encode_su3_color_field(grid)
        assert u.shape == (1, 2, 2, 3, 3)
        err = (
            u @ u.conj().transpose(-1, -2) - torch.eye(3, dtype=torch.complex64)
        ).abs().max().item()
        assert err < 1e-4

    def test_g1_all_distinct_colors_noncommute(self):
        grid = torch.arange(10).reshape(1, 1, 10)
        u = cg.encode_su3_color_field(grid)[0, 0]  # [10,3,3]
        min_dist = float("inf")
        for a in range(10):
            for b in range(a + 1, 10):
                d = torch.linalg.matrix_norm(u[a] @ u[b] - u[b] @ u[a]).item()
                min_dist = min(min_dist, d)
        assert min_dist > 0.5000


class TestConfinement:
    def test_singlet_projection_idempotent_and_free(self):
        psi = torch.randn(8, 3, 3, dtype=torch.complex64)
        p = cg.singlet_projection(psi)
        assert (p - cg.singlet_projection(p)).abs().max().item() < 1e-5
        assert cg.confinement_penalty(p).abs().max().item() < 1e-5

    def test_g2_veto_rates(self):
        n = 256
        a = torch.randn(n, dtype=torch.complex64)
        singlet = torch.einsum("n,jk->njk", a, torch.eye(3, dtype=torch.complex64))
        veto, _ = cg.confinement_veto(singlet)
        assert veto.float().mean().item() == 0.0
        nonsinglet = torch.randn(n, 3, 3, dtype=torch.complex64)
        veto, _ = cg.confinement_veto(nonsinglet)
        assert veto.float().mean().item() == 1.0


class TestSU3Transport:
    def test_fit_recovers_gauge_heldout(self):
        n, t = 32, 20
        theta = (torch.rand(n, 8) * 2 - 1) * 1.0
        psi0 = _su3_from_theta(theta)
        u_true = _su3_from_theta(torch.randn(1, 8) * 0.5)[0]
        traj = [psi0]
        for _ in range(t - 1):
            traj.append(cg.su3_transport(traj[-1], u_true))
        traj = torch.stack(traj)
        u_hat = cg.fit_su3_gauge(traj[:10])
        x = traj[10:-1].reshape(-1, 3, 3)
        y = traj[11:].reshape(-1, 3, 3)
        loss = (torch.matmul(x, u_hat) - y).abs().pow(2).mean().item()
        assert loss < 1e-3  # far below the 0.1500 gate at small scale


class TestDefaultOff:
    def test_flag_off_by_default(self):
        assert "HENRI_ARC_CHROMODYNAMIC" not in os.environ or os.environ.get(
            "HENRI_ARC_CHROMODYNAMIC"
        ) != "1"
        assert cg.ENABLED is False

    def test_triton_fallback_equivalence(self):
        a = torch.randn(16, 3, 3, dtype=torch.complex64)
        b = torch.randn(16, 3, 3, dtype=torch.complex64)
        ref = cg.su3_matmul_torch(a, b)
        got = cg.su3_matmul_triton(a, b)
        assert got.shape == ref.shape
        # fallback path (no CUDA in local CI) must be identical to reference
        assert (got - ref).abs().max().item() < 1e-6
