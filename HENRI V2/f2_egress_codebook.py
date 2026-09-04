"""F2-M3 Calibrated Hopfield Lexical Egress Codebook (default-OFF).

Spec ID: SPEC-2026-08-29-F2-EGRESS
Algorithm: F2HopfieldLexicalEgressCalibration

Mechanism:
    M = A B ,  A = Y^T (X X^T + lambda I)^{-1} ,  B = X     (dual ridge)
    z_clean = Softmax(beta * Re(Psi M^dagger)) M               (Hopfield snap, beta=8.0)

Gate: HENRI_F2_EGRESS=1 to enable. OFF => factory returns None and the legacy
pipeline is byte-identical (differential, not flag-read).

Dense-ban: the calibration solve NEVER forms a [D,D] tensor. It uses the thin
dual form:  X = U S V^T (thin SVD, [N,r]x[r]x[r,D])  =>  A = Y^T U S^{-1} V^T.
Accumulation dtype is float32 throughout. Zero trainable parameters (no-BPTT).

Eligibility boundary: this codebook alone NEVER grants score eligibility.
"""
from __future__ import annotations

import os
from typing import Optional

import torch


def _egress_enabled() -> bool:
    return os.environ.get("HENRI_F2_EGRESS", "0") == "1"


class F2HopfieldEgressCodebook:
    """Factorized Hopfield lexical codebook with dual-ridge calibration."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        beta: float = 8.0,
        ridge_lambda: float = 1e-3,
    ) -> None:
        if d_model <= 0 or vocab_size <= 0:
            raise ValueError("d_model and vocab_size must be positive")
        if beta <= 0.0:
            raise ValueError("beta must be positive")
        if not (ridge_lambda > 0.0):
            raise ValueError("ridge_lambda must be positive")
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.beta = beta
        self.ridge_lambda = ridge_lambda
        self.M: Optional[torch.Tensor] = None
        self._calibrated = False

    # -- calibration (dual ridge, dense-ban compliant) ---------------------
    def calibrate(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """Calibrate the codebook from (X, Y) pairs.

        X: [N, D] float32 wave stack (provenance-pinned calibration split).
        Y: [N, V] float32 one-hot token/action labels.

        Solves M = Y^T (X X^T + lambda I)^{-1} X  without forming [D,D].
        """
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be 2-D")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must share the row count")
        if X.dtype != torch.float32 or Y.dtype != torch.float32:
            X = X.to(torch.float32)
            Y = Y.to(torch.float32)

        N, D = X.shape
        V = Y.shape[1]
        if V != self.vocab_size:
            raise ValueError(f"Y vocab {V} != codebook vocab {self.vocab_size}")

        device = X.device
        # Thin SVD: X [N,D] -> U [N,r], S [r], Vt [r,D], r = min(N,D)
        U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        r = S.numel()
        # regularized inverse of singular values
        S_inv = S / (S * S + self.ridge_lambda)
        # A = Y^T U S^{-1}  ->  [V, r]
        A = (Y.transpose(0, 1) @ U) * S_inv.unsqueeze(0)  # [V, r]
        # M = A Vt  ->  [V, r] @ [r, D] = [V, D]
        self.M = (A @ Vt).to(device=device, dtype=torch.float32).contiguous()
        self._calibrated = True

    # -- snap (Hopfield retrieval) -----------------------------------------
    def snap(
        self, Psi: torch.Tensor, return_logits: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Softmax-Hopfield snap of wave states.

        Psi: [N, D] float32.
        Returns (z_clean [N, D], logits [N, V]) when return_logits, else (z_clean, None).
        """
        if self.M is None:
            raise RuntimeError("codebook not calibrated; call calibrate() first")
        if Psi.ndim != 2 or Psi.shape[1] != self.d_model:
            raise ValueError(f"Psi must be [N, {self.d_model}]")

        logits = Psi @ self.M.transpose(0, 1)  # [N, V]  (real inner products)
        logits = logits * self.beta
        weights = torch.softmax(logits, dim=-1)  # [N, V]
        z_clean = weights @ self.M  # [N, D]
        if return_logits:
            return z_clean, logits
        return z_clean, None

    # -- telemetry ---------------------------------------------------------
    def codebook_bytes(self) -> int:
        if self.M is None:
            return 0
        return self.M.numel() * self.M.element_size()

    def telemetry(self) -> dict:
        return {
            "f2_egress_status": "ENGAGED" if self._calibrated else "BLOCKED",
            "f2_p1_heldout": None,
            "f2_beta": self.beta,
            "f2_ridge_lambda": self.ridge_lambda,
            "f2_codebook_bytes": self.codebook_bytes(),
        }


def get_f2_egress(
    d_model: int = 65536,
    vocab_size: int = 32000,
    beta: float = 8.0,
    ridge_lambda: float = 1e-3,
) -> Optional[F2HopfieldEgressCodebook]:
    """Factory. Returns None unless HENRI_F2_EGRESS=1 (default-OFF)."""
    if not _egress_enabled():
        return None
    return F2HopfieldEgressCodebook(
        d_model=d_model, vocab_size=vocab_size, beta=beta, ridge_lambda=ridge_lambda
    )
