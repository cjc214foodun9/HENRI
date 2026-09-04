"""Carrier F7: per-task non-unitary affine operator & family-conditioned supervised egress.

Spec: HENRI-SPEC-2026-08-F7-AFFINE-EGRESS (docs/spec/f7_affine_egress_preregistration.md).
Selected by HENRI_F7_AFFINE=1; never imported on the default path.

Mechanism (directive 3.1 / spec Appendix A/B/C):
  z^(e) = A^(e) Psi_X + b^(e),   A in R^{K x D}, b in R^K
  A = Yc.T (Xc Xc.T + lam I)^{-1} Xc          (dual ridge, [M,M] solve)
  b = ybar - A xbar                           (affine centering)
  z_cal = z_raw + Sigma_family z_raw          (Tier 2, family covariance)
  a* = argmax_k z_cal
Memory contract (C8): no [D,D] tensor is ever formed. The solve keeps only
the [M,D] factor GinvX and [M,K] Yc; A is materialized ONLY when K*D <= 4e6
(toy-scale contract tests); predict() always uses the implicit form
  z = Yc.T (GinvX (x - xbar).T) + ybar.
"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Dict, List, Optional

import torch

_LAMBDA_DEFAULT = 1e-3
_SIGMA_EPS = 1e-3
_A_MATERIALIZE_ELEMS = 4_000_000  # C8: only toy-scale A materialization


class AffineEgress:
    """Closed-form dual ridge affine readout (zero trainable params)."""

    def __init__(self, lam: float = _LAMBDA_DEFAULT, sigma_eps: float = _SIGMA_EPS):
        self.lam = float(lam)
        self.sigma_eps = float(sigma_eps)
        self.A: Optional[torch.Tensor] = None       # [K, D], toy-scale only
        self.b: Optional[torch.Tensor] = None       # [K]
        self._factors: List[torch.Tensor] = []      # C8 audit: materialized tensors
        self._GinvX: Optional[torch.Tensor] = None  # [M, D]
        self._Yc: Optional[torch.Tensor] = None     # [M, K]
        self._xbar: Optional[torch.Tensor] = None   # [1, D]
        self._ybar: Optional[torch.Tensor] = None   # [1, K]
        self._M: int = 0
        self._K: int = 0
        self._Sigma: Optional[torch.Tensor] = None  # [K, K] family covariance
        self._fit_ms: Optional[float] = None

    def fit(self, X: torch.Tensor, Y: torch.Tensor,
            family_z: Optional[torch.Tensor] = None) -> "AffineEgress":
        """X: [M, D] real demo waves; Y: [M, K] targets (one-hot actions or waves).

        family_z: [N, K] demo readout vectors from TRAIN envs (Tier 2 prior);
        if given, Sigma_family = cov(family_z) + eps*I is fitted.
        """
        X = X.to(torch.float32)
        Y = Y.to(torch.float32)
        M, D = X.shape
        K = Y.shape[1]
        assert M >= 1 and K >= 1

        xbar = X.mean(0, keepdim=True)                 # [1, D]
        ybar = Y.mean(0, keepdim=True)                 # [1, K]
        Xc = X - xbar                                  # [M, D]
        Yc = Y - ybar                                  # [M, K]

        G = Xc @ Xc.T                                  # [M, M]
        G = G + self.lam * torch.eye(M, dtype=G.dtype, device=G.device)
        GinvX = torch.linalg.solve(G, Xc)              # [M, D]  (the only large factor)
        b = (ybar - (Yc.T @ (GinvX @ xbar.T)).T).squeeze(0)  # [K]

        self._GinvX = GinvX
        self._Yc = Yc
        self._xbar = xbar
        self._ybar = ybar
        self.b = b
        self._M = M
        self._K = K
        self._factors = [Xc, Yc, G, GinvX]
        if K * D <= _A_MATERIALIZE_ELEMS:              # C8: toy-scale only
            self.A = Yc.T @ GinvX                      # [K, D]
            self._factors.append(self.A)
        else:
            self.A = None

        if family_z is not None and family_z.shape[0] >= 2:
            Z = family_z.to(torch.float32)
            zbar = Z.mean(0, keepdim=True)
            Zc = Z - zbar
            Sigma = (Zc.T @ Zc) / max(Z.shape[0] - 1, 1)   # [K, K]
            Sigma = Sigma + self.sigma_eps * torch.eye(K, dtype=Sigma.dtype,
                                                       device=Sigma.device)
            self._Sigma = Sigma
        return self

    def predict(self, X: torch.Tensor, use_family: bool = True) -> torch.Tensor:
        """Implicit affine: z = Yc.T (GinvX (x - xbar).T) + ybar; then Tier 2."""
        assert self._GinvX is not None and self.b is not None, "fit() before predict()"
        X = X.to(self._GinvX.dtype)
        z = (self._Yc.T @ (self._GinvX @ (X - self._xbar).T)).T + self.b  # [N, K]
        if use_family and self._Sigma is not None:
            z = z + z @ self._Sigma.T
        return z

    def rank(self) -> int:
        # rank(A) <= min(rank(Yc), rank(GinvX)) <= min(M, K)
        return min(self._M, self._K)

    def to(self, device: str) -> "AffineEgress":
        for name in ("_GinvX", "_Yc", "_xbar", "_ybar", "b", "A", "_Sigma"):
            t = getattr(self, name)
            if t is not None:
                setattr(self, name, t.to(device))
        return self


def compile_affine_egress(
    X_demo: torch.Tensor,
    Y_demo: torch.Tensor,
    family_z: Optional[torch.Tensor] = None,
    lam: float = _LAMBDA_DEFAULT,
    device: str = "cpu",
) -> Dict:
    """Fit an AffineEgress and return the compact operator bundle."""
    t0 = time.perf_counter()
    eg = AffineEgress(lam=lam).fit(X_demo, Y_demo, family_z=family_z)
    eg.to(device)
    fit_ms = (time.perf_counter() - t0) * 1e3
    A_sha = hashlib.sha256(eg._GinvX.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    b_sha = hashlib.sha256(eg.b.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    sigma_trace = float(eg._Sigma.trace().item()) if eg._Sigma is not None else None
    return {
        "egress": eg,
        "schema_id": "f7-affine-egress.v1",
        "A_sha256": A_sha,
        "b_sha256": b_sha,
        "rank": eg.rank(),
        "fit_ms": round(fit_ms, 4),
        "sigma_trace": sigma_trace,
        "lam": lam,
    }


def _f7_enabled() -> bool:
    return os.environ.get("HENRI_F7_AFFINE") == "1"
