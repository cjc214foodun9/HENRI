"""Phase 8.15 — Non-Abelian Chromodynamic Grounding (default-OFF, additive).

PDF: HENRI-SPEC-2026-08-PHASE8.15-QCD (SHA 621d2456...).

Maps ARC colors to su(3) gauge fields via the 8 Gell-Mann matrices, enforces
color-singlet confinement with a string-tension veto, provides SU(3) gauge
transport for transition prediction, and ships a Triton 3x3 complex matmul
kernel (torch fallback).

Production default path is UNTOUCHED unless HENRI_ARC_CHROMODYNAMIC=1.
"""

from __future__ import annotations

import math
import os

import torch

ENABLED = os.environ.get("HENRI_ARC_CHROMODYNAMIC", "0") == "1"

# --- 8 Gell-Mann matrices, C^{3x3}, Hermitian traceless, Tr(la lb) = 2 delta_ab ---
GELL_MANN_BASIS = torch.tensor(
    [
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
        [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
        [[1 / math.sqrt(3), 0, 0], [0, 1 / math.sqrt(3), 0], [0, 0, -2 / math.sqrt(3)]],
    ],
    dtype=torch.complex64,
)

# Fixed deterministic color -> su(3) angle projection [10 colors, 8 generators].
# All columns populated so distinct colors generically fail to commute (G1-QCD).
DEFAULT_COLOR_PROJECTION = torch.tensor(
    [
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [1, 0, 0, 1, 1, 0, 0, 1],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0, 1],
        [0, 1, 0, 0, 1, 0, 1, 1],
    ],
    dtype=torch.float32,
)

SIGMA_CONFINEMENT = 0.18  # string tension (PDF 1.2)
EPS_CONFINEMENT = 1e-3

TRITON_AVAILABLE = False
try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    @triton.jit
    def _su3_cmatmul_kernel(
        a_ptr, b_ptr, c_ptr, n, BLOCK_N: tl.constexpr
    ):
        """Batched 3x3 complex matmul. Layout [n, 18]: c*9 + i*3 + j, c=0 re, c=1 im."""
        pid = tl.program_id(0)
        offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < n
        for i in tl.static_range(3):
            for j in tl.static_range(3):
                acc_re = tl.zeros([BLOCK_N], dtype=tl.float32)
                acc_im = tl.zeros([BLOCK_N], dtype=tl.float32)
                for k in tl.static_range(3):
                    a_re = tl.load(a_ptr + offs * 18 + 0 * 9 + i * 3 + k, mask=mask, other=0.0)
                    a_im = tl.load(a_ptr + offs * 18 + 1 * 9 + i * 3 + k, mask=mask, other=0.0)
                    b_re = tl.load(b_ptr + offs * 18 + 0 * 9 + k * 3 + j, mask=mask, other=0.0)
                    b_im = tl.load(b_ptr + offs * 18 + 1 * 9 + k * 3 + j, mask=mask, other=0.0)
                    acc_re += a_re * b_re - a_im * b_im
                    acc_im += a_re * b_im + a_im * b_re
                tl.store(c_ptr + offs * 18 + 0 * 9 + i * 3 + j, acc_re, mask=mask)
                tl.store(c_ptr + offs * 18 + 1 * 9 + i * 3 + j, acc_im, mask=mask)

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    pass


def encode_su3_color_field(
    grid_colors: torch.Tensor, color_projection: torch.Tensor | None = None
) -> torch.Tensor:
    """Map ARC colors [B,H,W] int 0..9 -> [B,H,W,3,3] complex SU(3) gauge fields.

    theta = one_hot(grid) @ projection  (PDF 2.1)
    U = matrix_exp(i * sum_a theta_a lambda_a)
    """
    if color_projection is None:
        color_projection = DEFAULT_COLOR_PROJECTION
    theta = torch.matmul(
        torch.nn.functional.one_hot(grid_colors, 10).float(),
        color_projection.to(grid_colors.device),
    )  # [B,H,W,8]
    su3_algebra = 1j * torch.einsum(
        "bhwa,arc->bhwrc",
        theta.to(torch.complex64),
        GELL_MANN_BASIS.to(theta.device),
    )
    return torch.matrix_exp(su3_algebra)


def singlet_projection(psi: torch.Tensor) -> torch.Tensor:
    """P_singlet(Psi) = (1/3) Tr(Psi) I3 (PDF 1.2)."""
    tr = psi.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # [...,]
    eye = torch.eye(3, dtype=psi.dtype, device=psi.device)
    return (tr / 3.0).unsqueeze(-1).unsqueeze(-1) * eye


def confinement_penalty(psi: torch.Tensor, sigma: float = SIGMA_CONFINEMENT) -> torch.Tensor:
    """F_conf = sigma * ||Psi - P_singlet(Psi)||_F^2 per channel."""
    diff = psi - singlet_projection(psi)
    return sigma * (diff.abs() ** 2).sum(dim=(-1, -2))


def confinement_veto(
    psi: torch.Tensor,
    sigma: float = SIGMA_CONFINEMENT,
    eps: float = EPS_CONFINEMENT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(veto, penalty): Delta_Sagnac = 1.0 iff any channel violates singlet confinement."""
    pen = confinement_penalty(psi, sigma)
    return pen > eps, pen


def su3_transport(field: torch.Tensor, gauge: torch.Tensor) -> torch.Tensor:
    """Psi' = Psi * U_gauge per channel (right action)."""
    return torch.matmul(field, gauge)


def fit_su3_gauge(traj: torch.Tensor) -> torch.Tensor:
    """Fit U minimizing sum_k ||Psi_{k+1} - Psi_k U||_F^2 over steps of traj.

    traj: [T, N, 3, 3] complex -> [3,3] complex gauge.
    """
    x = traj[:-1].reshape(-1, 3, 3)
    y = traj[1:].reshape(-1, 3, 3)
    a = torch.einsum("nki,nkj->ij", x.conj().transpose(-1, -2), x)
    b = torch.einsum("nki,nkj->ij", x.conj().transpose(-1, -2), y)
    return torch.linalg.solve(a, b)


def su3_matmul_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reference batched 3x3 complex matmul for [N,3,3]."""
    return torch.matmul(a, b)


def su3_matmul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[N,3,3] complex x [N,3,3] complex via Triton; torch fallback if unavailable."""
    if not TRITON_AVAILABLE:
        return su3_matmul_torch(a, b)
    n = a.shape[0]
    ab = torch.view_as_real(a).permute(0, 3, 1, 2).reshape(n, 18).contiguous()
    bb = torch.view_as_real(b).permute(0, 3, 1, 2).reshape(n, 18).contiguous()
    cb = torch.empty_like(ab)
    grid = (triton.cdiv(n, 64),)
    _su3_cmatmul_kernel[grid](ab, bb, cb, n, BLOCK_N=64)
    return torch.view_as_complex(cb.reshape(n, 2, 3, 3).permute(0, 2, 3, 1).contiguous())


def structure_constants() -> tuple[torch.Tensor, float]:
    """Numeric f_abc from [la, lb] = 2i f_abc lc; returns (f[8,8,8], max residual)."""
    f = torch.zeros(8, 8, 8, dtype=torch.complex64)
    max_res = 0.0
    for a in range(8):
        for b in range(8):
            comm = GELL_MANN_BASIS[a] @ GELL_MANN_BASIS[b] - GELL_MANN_BASIS[b] @ GELL_MANN_BASIS[a]
            for c in range(8):
                f[a, b, c] = torch.trace(comm @ GELL_MANN_BASIS[c]) / (4j)
            rhs = 2j * torch.einsum("c,cij->ij", f[a, b], GELL_MANN_BASIS)
            max_res = max(max_res, (comm - rhs).abs().max().item())
    return f, max_res
