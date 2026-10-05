"""Shared wave geometry for the HENRI tri-model core.

One term per meaning:
  wave        unit-norm complex vector Psi in C^D, D = 65,536
  block       one of 8,192 Cl(3,0) blocks of 8 real blades
  blade       one of the 8 Clifford basis elements (1, e1, e2, e3, e12, e13, e23, e123)
  slot        one of 4 structural subspaces of 2,048 blocks (16,384 real dims)

Document anchors:
  doc p5  : 8,192 Cl(3,0) blocks; 4-Slot rule
  doc p6  : Slot 0 Entity 0..2047, Slot 1 Action 2048..4095, Slot 2 Object 4096..6143,
            Slot 3 Context 6144..8191
  doc p20 : Stiefel manifold S^{D-1}; A_signal = const rather than 1/sqrt(L)
  doc p25 : Sagnac  Delta = 1.0 - Re(<Psi_cand, Psi_axiom>)
"""
from __future__ import annotations

import math

import torch

DEFAULT_DIM = 65536
BLADES = 8                      # Cl(3,0) basis-blade count
N_BLOCKS = DEFAULT_DIM // BLADES
N_SLOTS = 4
SLOT_BLOCKS = N_BLOCKS // N_SLOTS          # 2048
SLOT_DIM = DEFAULT_DIM // N_SLOTS          # 16384
SLOT_NAMES = ("entity", "action", "object", "context")
SAGNAC_THRESHOLD = 0.35         # doc p25: Delta <= 0.35 clears the port


def slot_block_range(i: int, n_blocks: int = N_BLOCKS, n_slots: int = N_SLOTS):
    """Return (start, stop) block indices for structural slot `i`."""
    if not 0 <= i < n_slots:
        raise ValueError(f"slot {i} out of range 0..{n_slots - 1}")
    per = n_blocks // n_slots
    return i * per, (i + 1) * per


def slot_flat_range(i: int, dim: int = DEFAULT_DIM, n_slots: int = N_SLOTS):
    """Return (start, stop) FLAT component indices for structural slot `i`."""
    if not 0 <= i < n_slots:
        raise ValueError(f"slot {i} out of range 0..{n_slots - 1}")
    per = dim // n_slots
    return i * per, (i + 1) * per


def unit_norm(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Project to the Stiefel manifold S^{D-1}: one step, exact.

    A single complex row-vector is a point on S^{D-1}; normalizing divides by
    the 2-norm. This is the closed-form Stiefel retraction for k=1.
    """
    n = z.norm(dim=-1, keepdim=True)
    return z / n.clamp_min(eps)


def stiefel_retract(mat: torch.Tensor, iters: int = 5, eps: float = 1e-12,
                    method: str = "qr") -> torch.Tensor:
    """Orthonormalize the rows of `mat` (shape [K, D]) onto the Stiefel manifold.

    D75 (self-caught): the first draft used Newton-Schulz with an interleaved
    row renormalization. That is NOT a convergent polar iteration, and it
    diverged for K=16 on highly correlated rows (gate G-U2 measured
    offdiag_max = 1.0000002, i.e. two rows became identical). It happened to
    work at K=5, which is how the defect survived a passing test.

    Primary path: dual thin QR. For X [K, D], factor X^H [D, K] = Q R with Q
    orthonormal columns; then Q^H [K, D] has orthonormal rows and spans the same
    row space. This is the closed-form Stiefel retraction the document names
    (doc p4). Cost is O(D K^2): no D^2 allocation, per the architecture rule.

    method="newton_schulz" keeps the iterative polar path, used only when QR is
    unavailable or degenerate.
    """
    if mat.dim() != 2:
        raise ValueError("stiefel_retract expects a 2-D matrix [K, D]")
    k, d = mat.shape
    if k > d:
        raise ValueError("K > D: rows cannot be orthonormal")
    x = mat / mat.norm(dim=-1, keepdim=True).clamp_min(eps)

    if method == "qr":
        try:
            q, _ = torch.linalg.qr(x.conj().transpose(-1, -2), mode="reduced")
            return q.conj().transpose(-1, -2)
        except Exception:                      # noqa: BLE001 - fall through
            method = "newton_schulz"

    # iterative polar fallback: X <- 1.5 X - 0.5 X (X^H X), no row rescaling
    for _ in range(max(1, iters)):
        gram = x.conj() @ x.transpose(-1, -2)      # [K, K], dual Gram
        x = 1.5 * x - 0.5 * (gram @ x)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def sagnac_margin(psi_cand: torch.Tensor, psi_axiom: torch.Tensor) -> torch.Tensor:
    """Delta_Sagnac = 0.5 ||Psi_cand - Psi_axiom||^2 = 1 - Re(<cand, axiom>).

    doc p25. Both waves must be unit norm. Returns a real tensor.
    """
    inner = (psi_cand.conj() * psi_axiom).sum(dim=-1)
    return (1.0 - inner.real).clamp_min(0.0)


def koopman_apply(
    z: torch.Tensor,
    diag: torch.Tensor,
    a_mat: torch.Tensor,
    s_mat: torch.Tensor,
    b_mat: torch.Tensor,
) -> torch.Tensor:
    """Apply  K = diag(m) + A S B^H  to a wave batch.  doc p12 / p17.

    z      [B, D] complex
    diag   [D]    complex  (m)
    a_mat  [D, r], b_mat [D, r], s_mat [r, r] complex, r << D
    Returns [B, D]. Cost is B*r*D, never D^2.
    """
    bh_z = z @ b_mat.conj()                 # [B, r]
    core = bh_z @ s_mat.transpose(0, 1)     # [B, r]
    low_rank = core @ a_mat.conj().transpose(0, 1)   # [B, D]
    return diag.unsqueeze(0) * z + low_rank


def hopfield_energy(z: torch.Tensor, patterns: torch.Tensor, beta: float) -> torch.Tensor:
    """Continuous modern Hopfield Lyapunov energy.  doc p19.

    E(xi) = -beta^-1 ln sum_i exp(beta <x_i, xi>) + 0.5 ||xi||^2
    z        [B, D] one or more query states
    patterns [N, D] stored archetypes (each unit norm)
    Returns [B] real energies. Lower is more converged.
    """
    logits = beta * (z @ patterns.conj().transpose(0, 1)).real      # [B, N]
    lse = torch.logsumexp(logits, dim=-1)
    quad = 0.5 * (z.abs() ** 2).sum(dim=-1)
    return -lse / beta + quad


def kuramoto_order(phase: torch.Tensor) -> torch.Tensor:
    """Kuramoto order parameter r = |mean(exp(i*phase))| over the batch.  doc p14."""
    return torch.exp(1j * phase).mean().abs()


def phase_of(z: torch.Tensor) -> torch.Tensor:
    return torch.angle(z)
