"""OBSERVED: Phase 10.1 Koopman-subspace task-functor estimator.

    W* = sum_{j=1..K} alpha_j L_j,     alpha = (G + lambda I)^-1 b
    G[j,k] = sum_m <L_j X_m, L_k X_m>          (K x K, Hermitian)
    b[j]   = sum_m <L_j X_m, Y_m>              (K)

Replaces the D=32768-parameter diagonal regression with K<=8 parameters, which is
the directive's sample-efficiency argument (section 3.2). The bank is a list of
callables from `koopman_generator_bank.build_generator_bank`; a dense [K,D,D] bank
is impossible (68.7 GB at K=8, D=32768).

Falsifiable kill experiment, pre-registered BEFORE running:
    ACCEPT iff held_out_mean > 0.5000 on the 60-task split AND
            gap <= 0.2490 (directive section 3, p.8, recovered from the PDF --
            the value was lost to a math-mode failure in the request text).
    else FALSIFIED. A K=8 restriction that cannot beat the 0.4215 diagonal
    baseline is a negative result and is reported as one.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from koopman_generator_bank import build_generator_bank


def compute_koopman_subspace_functor(
    X_demos: torch.Tensor,
    Y_demos: torch.Tensor,
    generators: Sequence,
    reg_lambda: float = 1e-2,
    ridge_relative: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Solve for alpha in the K-dimensional generator subspace.

    X_demos, Y_demos: [M, D] complex (flat; the flat index IS (block, slot)).
    generators: sequence of callables L(X) -> X', each [M, D] -> [M, D] complex.
    Returns (alpha [K], apply_fn, diagnostics).
    """
    if not isinstance(X_demos, torch.Tensor) or not torch.is_complex(X_demos):
        raise TypeError(f"X_demos must be a complex tensor, got {type(X_demos)}")
    if X_demos.shape != Y_demos.shape:
        raise ValueError(f"X/Y shape mismatch: {tuple(X_demos.shape)} "
                         f"vs {tuple(Y_demos.shape)}")
    K = len(generators)
    if K == 0:
        raise ValueError("empty generator bank")

    LX = [L(X_demos) for L in generators]                 # each [M, D]
    D = X_demos.shape[-1]
    G = torch.zeros(K, K, dtype=torch.complex64, device=X_demos.device)
    b = torch.zeros(K, dtype=torch.complex64, device=X_demos.device)
    for j in range(K):
        for k in range(K):
            G[j, k] = (LX[j].conj() * LX[k]).sum()
        b[j] = (LX[j].conj() * Y_demos).sum()
    # RIDGE SCALE (defect fixed). With reg_lambda read as an ABSOLUTE value the
    # ridge is numerically absent here: measured mean Gram diagonal mass is
    # ~2.5e+07, so lambda=1e-2 is a relative 4e-10, and lambda 1e-2 and 1e-1 gave
    # bit-identical results. Solving an unregularised system against a rank-5.75
    # of-8 Gram is the exact degeneracy the ridge is supposed to remove, so the
    # first pass did NOT fairly test the subspace hypothesis.
    # Default is now RELATIVE: lambda_eff = reg_lambda * mean(|diag(G)|).
    gram_scale = float(G.diagonal().abs().mean().item()) if K else 1.0
    scale = gram_scale if ridge_relative else 1.0
    lam = torch.as_tensor(reg_lambda * scale, dtype=G.real.dtype, device=G.device)
    G_reg = G + lam * torch.eye(K, dtype=G.dtype, device=G.device)
    try:
        alpha = torch.linalg.solve(G_reg, b)
    except Exception:
        alpha = torch.linalg.lstsq(G_reg, b.unsqueeze(-1)).solution.squeeze(-1)

    G_shift = G.detach().clone()
    G_shift[range(K), range(K)] = 0
    diag_mass = float(G.diagonal().abs().sum())
    off_mass = float(G_shift.abs().sum())
    diag_only = bool(off_mass < 1e-6 * max(diag_mass, 1e-30))

    def apply_fn(X: torch.Tensor, alpha=alpha, LX_=None) -> torch.Tensor:
        out = torch.zeros_like(X)
        for j in range(K):
            out = out + alpha[j] * generators[j](X)
        return out

    diag = {"K": K, "reg_lambda": reg_lambda,
            "ridge_relative": ridge_relative, "gram_scale": gram_scale,
            "lambda_effective": float(lam.item()),
            "rank_G": int(torch.linalg.matrix_rank(G).item()),
            "gram_diag_mass": diag_mass, "gram_offdiag_mass": off_mass,
            "gram_is_diagonal": diag_only,
            "cond_G_reg": float(torch.linalg.cond(G_reg).item())
            if G_reg.abs().max().item() > 0 else float("inf"),
            "alpha_abs": [float(a.abs().item()) for a in alpha],
            "alpha_argmax": int(alpha.abs().argmax().item())}
    return alpha, apply_fn, diag
