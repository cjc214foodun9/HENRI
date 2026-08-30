"""Carrier F6: Per-Task Adaptive Functor Compilation (default-OFF).

Spec: HENRI-SPEC-2026-08-F6-ADAPTIVE-FUNCTOR (docs/spec/f6_adaptive_functor_preregistration.md).
Selected by HENRI_F6_FUNCTOR=1; never imported by the production runner.

Tier 2 - in-situ Procrustes functor synthesis with Newton-Schulz retraction.
K_raw = (1/M) sum_i Psi_Y,i (x) Psi_X,i^dag  (FHRR binding sum; circulant).
The dense [D,D] NS iteration is infeasible at D=65536 (17.2 GB fp32), but a
circulant operator is diagonalized by the DFT, so the identical iteration runs
as D independent complex-scalar updates in the Fourier domain:
    w_b <- 1.5 w_b - 0.5 w_b |w_b|^2
same operator, same fixed point (unit modulus), same Gate G1.

Tier 3 - task subspace de-occlusion masking:
    m_active = I( Var_demo(Arg(Psi_X)) > eps_floor )
    Psi_goal = normalize( m_active (x) (W (x) Psi_X_query) )

Tier 4 - calibrated Hopfield lexical snapping:
    a* = argmax_k Re( <Psi_goal, M_k^(e)> ),  M_k^(e) = per-action prototype waves.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tier 2 kernels
# ---------------------------------------------------------------------------

def spectral_newton_schulz(
    w: torch.Tensor,
    max_iters: int = 5,
    tol: float = 1e-5,
) -> Tuple[torch.Tensor, float, int]:
    """Newton-Schulz unitary retraction in the Fourier domain.

    w: complex [D] circulant generator (the FHRR functor K_raw).
    Returns (w_out, err, iters) with
        err = ||W^dag W - I||_F computed spectrally as sqrt(sum_b (|w_b|^2 - 1)^2).
    """
    if w.ndim != 1 or not torch.is_complex(w):
        raise ValueError(f"spectral_newton_schulz expects complex 1-D, got {tuple(w.shape)}")
    W = torch.fft.fft(w)
    # Directive eq. 2: W_0 = K_raw / ||K_raw||_2 (spectral norm = max |lambda_b|).
    # The scalar map x <- x(1.5 - 0.5|x|^2) converges to the unit circle only
    # from |x| < sqrt(3); spectral normalization guarantees |lambda_b| <= 1 and
    # monotone convergence to the polar factor (unit-modulus on the support,
    # zero on the null space).
    W = W / (W.abs().max() + 1e-12)
    err = float("inf")
    k = 0
    for k in range(1, max_iters + 1):
        W = 1.5 * W - 0.5 * W * (W.abs() ** 2)
        err = float(torch.sqrt(((W.abs() ** 2 - 1.0).square().sum())).item())
        if err <= tol:
            break
    return torch.fft.ifft(W), err, k


# ---------------------------------------------------------------------------
# Tier 3 kernel
# ---------------------------------------------------------------------------

def deocclusion_mask(
    X: torch.Tensor,
    eps_floor: float = 1e-3,
) -> torch.Tensor:
    """Task subspace de-occlusion mask over frequency bins.

    X: complex [M, D] demo waves. Bins whose phase variance across demos is
    at or below eps_floor are uninformative background modes and are masked.
    Returns bool [D].
    """
    if X.ndim != 2 or not torch.is_complex(X):
        raise ValueError(f"deocclusion_mask expects complex [M, D], got {tuple(X.shape)}")
    phase_var = X.angle().var(dim=0)
    return phase_var > eps_floor


# ---------------------------------------------------------------------------
# Tier 2 + Tier 3 composition
# ---------------------------------------------------------------------------

def compile_adaptive_functor(
    X: torch.Tensor,
    Y: torch.Tensor,
    max_iters: int = 8,
    tol: float = 1e-5,
    eps_floor: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, float, int, float]:
    """Compile the per-task adaptive functor from demo waves.

    X, Y: complex [M, D] demo input/output waves (unit-ish modulus).
    Returns (W, mask, ns_err, ns_iters, recon_fidelity) with
        recon_fidelity = (1/M) sum_i cos(W (x) X_i, Y_i)   (Gate G2 >= 0.90).
    """
    if X.shape != Y.shape or X.ndim != 2:
        raise ValueError(f"X/Y must be complex [M, D] with equal shape, got {tuple(X.shape)}/{tuple(Y.shape)}")
    M = X.shape[0]
    if M == 0:
        raise ValueError("compile_adaptive_functor requires >= 1 demo pair")

    k_raw = torch.zeros_like(X[0])
    for i in range(M):
        k_raw = k_raw + Y[i] * torch.conj(X[i])
    k_raw = k_raw / M
    w0 = k_raw / (k_raw.norm() + 1e-12)

    W, err, iters = spectral_newton_schulz(w0, max_iters=max_iters, tol=tol)
    mask = deocclusion_mask(X, eps_floor=eps_floor)

    fid = 0.0
    with torch.no_grad():
        for i in range(M):
            pred = W * X[i]
            num = float(torch.abs(torch.vdot(pred, Y[i])).item())
            den = float(pred.norm().item()) * float(Y[i].norm().item()) + 1e-12
            fid += num / den
    fid /= M
    return W, mask, err, iters, float(fid)


# ---------------------------------------------------------------------------
# Tier 2 + Tier 3 + Tier 4 class
# ---------------------------------------------------------------------------

class AdaptiveFunctorCompiler:
    """Per-task functor compiler with de-occlusion masking + lexical snapping.

    Default-OFF: instantiate only under HENRI_F6_FUNCTOR=1 (harness or runner
    flag branch); never imported by the production runner's default path.
    """

    compiler_name = "adaptive_functor"
    compiler_version = "f6-v1"

    def __init__(
        self,
        device: Optional[str] = None,
        max_iters: int = 8,
        tol: float = 1e-5,
        eps_floor: float = 1e-3,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_iters = int(max_iters)
        self.tol = float(tol)
        self.eps_floor = float(eps_floor)
        self._w: Optional[torch.Tensor] = None
        self._mask: Optional[torch.Tensor] = None
        self._prototypes: Dict[str, torch.Tensor] = {}
        self._ns_err: Optional[float] = None
        self._ns_iters: Optional[int] = None
        self._recon_fidelity: Optional[float] = None

    def compile_demo(
        self,
        x_waves: Sequence[torch.Tensor],
        y_waves: Sequence[torch.Tensor],
        action_names: Sequence[str],
    ) -> None:
        """Compile W^(e), mask, and per-action prototype waves from demo rows."""
        if len(x_waves) == 0:
            raise ValueError("compile_demo requires >= 1 demo pair")
        if len(x_waves) != len(y_waves) or len(x_waves) != len(action_names):
            raise ValueError("x_waves/y_waves/action_names length mismatch")
        X = torch.stack([w.to(self.device) for w in x_waves])
        Y = torch.stack([w.to(self.device) for w in y_waves])
        W, mask, err, iters, fid = compile_adaptive_functor(
            X, Y, max_iters=self.max_iters, tol=self.tol, eps_floor=self.eps_floor)
        self._w = W
        self._mask = mask
        self._ns_err = float(err)
        self._ns_iters = int(iters)
        self._recon_fidelity = fid
        # Tier 4: per-action prototype waves (mean of demo Y waves per action).
        protos: Dict[str, List[torch.Tensor]] = {}
        for name, yw in zip(action_names, y_waves):
            protos.setdefault(name, []).append(yw.to(self.device))
        self._prototypes = {}
        for name, ws in protos.items():
            acc = torch.zeros_like(ws[0])
            for wv in ws:
                acc = acc + wv
            self._prototypes[name] = F.normalize(acc, p=2, dim=-1)

    def retrieve(self, x_wave: torch.Tensor) -> Tuple[str, Dict[str, float]]:
        """Goal wave -> lexical snap: a* = argmax_k Re(<Psi_goal, M_k>)."""
        if self._w is None or self._mask is None or not self._prototypes:
            raise RuntimeError("retrieve() before compile_demo()")
        x = x_wave.to(self.device)
        goal = self._mask.to(self._w.dtype) * (self._w * x)
        goal = F.normalize(goal, p=2, dim=-1)
        scores: Dict[str, float] = {}
        for name, proto in self._prototypes.items():
            scores[name] = float(torch.real(torch.vdot(goal, proto)).item())
        best = max(scores, key=scores.get)
        return best, scores

    @property
    def ns_err(self) -> Optional[float]:
        return self._ns_err

    @property
    def ns_iters(self) -> Optional[int]:
        return self._ns_iters

    @property
    def recon_fidelity(self) -> Optional[float]:
        return self._recon_fidelity

    @property
    def active_mask_fraction(self) -> Optional[float]:
        if self._mask is None:
            return None
        return float(self._mask.float().mean().item())

    def geometry_metadata(self) -> Dict[str, object]:
        return {
            "compiler_name": self.compiler_name,
            "compiler_version": self.compiler_version,
            "max_iters": self.max_iters,
            "tol": self.tol,
            "eps_floor": self.eps_floor,
            "ns_err": self.ns_err,
            "ns_iters": self.ns_iters,
            "recon_fidelity": self.recon_fidelity,
            "active_mask_fraction": self.active_mask_fraction,
            "tiers": "T2_ns_retraction T3_deocclusion_mask T4_hopfield_snap",
        }
