"""hops_vsa_core.py — HOPS-VLA reference core (Class 4.5, default-OFF).

Holographic Orthogonal Phase-Swarm module per
HENRI_Ground-Up_VSA_Holographic_Model_Specification.md (SHA c304d30d...).
Python reference implementation of the three load-bearing mechanisms:

A. Invariant-subspace decoupling: P_null = I - V V^dagger where V is an
   orthonormal skeleton basis (common AST node types). Removes the shared
   carrier so lookalike skeletons stop dominating cosine ranking.
B. Diagonal complex Clifford rotor: per-phasor rotation exp(i b), exactly
   isometric (Gram < 1e-6 by construction; retraction enforces the bound).
C. Dual-channel Sagnac veto over the skeleton-free channel (delta > 0.35
   vetoes the proposal before scoring).

Representation contract: continuous float32 waves [D] (interleaved cos/sin
pairs), D = 65,536. uint8 Z_256 ring input raises RepresentationBoundaryError.
No dense [D, D] allocation (banned, ~34 GiB); projector is [D, k] with k <= 8.

Default-OFF experimental component. Not imported by any production path unless
the runner flag --hops-vsa-rank is explicitly set.
"""
from __future__ import annotations

import hashlib
import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

SKELETON_NODE_TYPES = (
    "Module", "FunctionDef", "Return", "Name", "arg", "Call", "arguments", "Load",
)
VETO_TAU = 0.35


def ring_to_real_wave(ring: torch.Tensor) -> torch.Tensor:
    """Explicit representation boundary: Z_256 uint8 ring -> continuous float32
    S^{D-1} wave via ((c/255)-1)*2 then unit normalize (representation-core-audit)."""
    if ring.dtype != torch.uint8:
        raise RepresentationBoundaryError(
            f"ring_to_real_wave requires uint8 Z_256 ring, got {ring.dtype}"
        )
    real = (ring.to(torch.float32) / 255.0 - 1.0) * 2.0
    return torch.nn.functional.normalize(real, p=2, dim=0)


class RepresentationBoundaryError(TypeError):
    """Raised when a Z_256 uint8 ring vector crosses into the continuous module."""


def _validate_wave(x: torch.Tensor, d_model: int) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(x).__name__}")
    if x.dtype == torch.uint8:
        raise RepresentationBoundaryError(
            "uint8 Z_256 ring vector crossed into the continuous HOPS-VSA module; "
            "map through ring_to_real ((c/255)-1)*2 then normalize"
        )
    if x.dtype != torch.float32:
        raise RepresentationBoundaryError(f"HOPS-VSA requires float32 waves, got {x.dtype}")
    if x.shape != (d_model,):
        raise ValueError(f"expected [{d_model}] waves, got {tuple(x.shape)}")


def _phasor_wave(seed_str: str, d_model: int, device: str = "cpu") -> torch.Tensor:
    """Deterministic unit-norm phasor wave from a seed string (no dataset content)."""
    if d_model % 2 != 0:
        raise ValueError("d_model must be even (interleaved cos/sin pairs)")
    n_phasors = d_model // 2
    raw = hashlib.sha256(seed_str.encode("utf-8")).digest()
    gen = bytearray()
    ctr = 0
    # float32 = 4 bytes per value; need exactly n_phasors values.
    while len(gen) < n_phasors * 4:
        gen.extend(hashlib.sha256(raw + str(ctr).encode("utf-8")).digest())
        ctr += 1
    # uint32 = 4 bytes per value; mod 65536 gives a well-defined angle.
    # (Interpreting raw bytes as float32 would produce ~50% NaN/Inf.)
    vals = torch.frombuffer(bytes(gen[: n_phasors * 4]), dtype=torch.uint32).to(torch.float32)
    angles = (vals % 65536) / 65536.0 * (2.0 * math.pi)
    wave = torch.empty(d_model, dtype=torch.float32)
    wave[0::2] = torch.cos(angles)
    wave[1::2] = torch.sin(angles)
    return wave.to(device) / math.sqrt(float(n_phasors))


class HopsVSASkeletonProjector(nn.Module):
    """P_null = I - V V^dagger over an orthonormal skeleton basis."""

    def __init__(self, d_model: int = 65536, skeleton_types: Sequence[str] = SKELETON_NODE_TYPES,
                 device: str = "cpu") -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.device = device
        cols = [_phasor_wave(f"HOPS_SKEL_{t}", d_model, device) for t in skeleton_types]
        V = torch.stack(cols, dim=1)  # [D, k]
        # Orthonormalize columns via Cholesky on the [k, k] Gram (rank-feasible).
        G = V.T @ V + 1e-8 * torch.eye(V.shape[1], device=device)
        L = torch.linalg.cholesky(G)
        V = torch.linalg.solve_triangular(L, V.T, upper=False).T
        self.register_buffer("V", V)  # [D, k]

    def project_null(self, psi: torch.Tensor) -> torch.Tensor:
        """Skeleton-free channel: psi - V (V^dagger psi)."""
        _validate_wave(psi, self.d_model)
        coef = self.V.T @ psi  # [k]
        return psi - self.V @ coef

    def project_skeleton(self, psi: torch.Tensor) -> torch.Tensor:
        _validate_wave(psi, self.d_model)
        return self.V @ (self.V.T @ psi)

    def gram_error(self) -> float:
        V = self.V
        return float((V.T @ V - torch.eye(V.shape[1], device=V.device)).abs().max().item())


class HopsVSACliffordRotor(nn.Module):
    """Diagonal complex Clifford rotor: per-phasor rotation exp(i b).

    Apply maps (cos, sin) pairs by the angle b; exactly orthogonal per phasor.
    retract() clamps the bivector to enforce Gram < 1e-6 (spec E_Gram).
    """

    def __init__(self, d_model: int = 65536, device: str = "cpu", seed: int = 7) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even")
        self.d_model = int(d_model)
        self.num_phasors = d_model // 2
        g = torch.Generator()
        g.manual_seed(seed)
        self.b = nn.Parameter(torch.randn(self.num_phasors, generator=g, device="cpu") * 0.01)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        _validate_wave(psi, self.d_model)
        b = self.b.to(psi.device)
        cb = torch.cos(b)
        sb = torch.sin(b)
        out = torch.empty_like(psi)
        c = psi[0::2]
        s = psi[1::2]
        out[0::2] = c * cb - s * sb
        out[1::2] = c * sb + s * cb
        return out

    def retract(self) -> float:
        """Enforce per-phasor isometry (Gram = |cos^2+sin^2 - 1| max)."""
        with torch.no_grad():
            self.b.copy_(self.b % (2.0 * math.pi))
            cb = torch.cos(self.b)
            sb = torch.sin(self.b)
            gram = float((cb * cb + sb * sb - 1.0).abs().max().item())
        return gram


class HopsVSASagnacGate:
    """Dual-channel Sagnac veto over the skeleton-free channel (delta > 0.35 veto)."""

    def __init__(self, tau: float = VETO_TAU) -> None:
        self.tau = float(tau)

    def delta(self, pred: torch.Tensor, emp: torch.Tensor) -> torch.Tensor:
        """1 - Re(<pred, emp>)/(||pred|| ||emp||) over skeleton-free channel."""
        _validate_wave(pred, pred.shape[0])
        _validate_wave(emp, pred.shape[0])
        pn = torch.nn.functional.normalize(pred, p=2, dim=0)
        en = torch.nn.functional.normalize(emp, p=2, dim=0)
        return 1.0 - torch.dot(pn, en).clamp(-1.0, 1.0)

    def veto(self, pred: torch.Tensor, emp: torch.Tensor) -> bool:
        return bool(self.delta(pred, emp).item() > self.tau)


class HopsVSACandidateScorer:
    """Candidate ranking over the skeleton-free channel (P_null residual)."""

    def __init__(self, projector: HopsVSASkeletonProjector,
                 gate: Optional[HopsVSASagnacGate] = None) -> None:
        self.projector = projector
        self.gate = gate or HopsVSASagnacGate()

    def score(self, goal: torch.Tensor, candidate: torch.Tensor) -> tuple[float, bool]:
        g_null = self.projector.project_null(goal)
        c_null = self.projector.project_null(candidate)
        g_null = torch.nn.functional.normalize(g_null, p=2, dim=0)
        c_null = torch.nn.functional.normalize(c_null, p=2, dim=0)
        cos = float(torch.dot(g_null, c_null).item())
        veto = self.gate.veto(candidate, goal)
        return cos, veto
