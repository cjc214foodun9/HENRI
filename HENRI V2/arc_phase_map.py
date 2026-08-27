"""ARC Spatial Phase-Map Verifier and Fractional Unbinding (Phase 7.2 Step 3).

Fail-closed module for the ACTION6 coordinate premise. It verifies whether
the live encoder's spatial basis is invertible enough to localize (x, y)
from a wave, and implements the correct fractional-unbinding protocol
(color-signature spectral division + position-carrier LUT argmax).

OBSERVED (2026-08-11, probe phase72_probe_basis.py): the production
HENRIVisionEncoder builds spatial_phases_x == spatial_phases_y (both
linspace(0, 2*pi*127, D/2)), so the position carrier is exp(i*(x+y)*w) —
the map is rank-deficient: single-pixel grids at (1,2) and (2,1) produce
IDENTICAL waves (cos 1.000000). Fractional unbinding cannot recover (x,y)
from the current phase field. Pre-registered status:
    BLOCKED_PHASE_MAP_NONINVERTIBLE.

The correct protocol (validated on a synthetic separable basis where the
ramps differ, omega_x != omega_y) is:

1. unbind the object color signature: carrier = wave / codebook(v)  (complex
   spectral division; ring-domain equivalent is qFHRR modular subtraction);
2. correlate the residual against position carriers P^x ⊛ P^y over the grid
   manifold (LUT_cos);
3. (x*, y*) = argmax of the cosine response surface.

This module is default-off infrastructure: it never modifies the encoder.
An encoder-basis change (distinct incommensurate ramps) is a load-bearing
representation change and requires explicit approval before the block is
lifted.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

STATUS_NONINVERTIBLE = "BLOCKED_PHASE_MAP_NONINVERTIBLE"
STATUS_INVERTIBLE = "PHASE_MAP_INVERTIBLE"
STATUS_NO_BASIS = "BLOCKED_PHASE_BASIS_UNAVAILABLE"


@dataclass
class PhaseMapVerdict:
    status: str = ""
    reason: str = ""
    same_sum_cos: Optional[float] = None
    diff_sum_cos: Optional[float] = None
    degeneracy_detected: bool = False
    grid_dim: int = 0


def verify_phase_map_invertibility(
    encoder: object,
    grid_dim: int = 4,
    color: int = 5,
    device: str = "cpu",
    degenerate_cos_threshold: float = 0.9,
) -> PhaseMapVerdict:
    """Detect collinear x/y ramps in a live HENRIVisionEncoder-style basis.

    Encodes single-pixel grids at (1,2) and (2,1) (same x+y) and (1,1)
    (different sum). If the same-sum waves are near-identical (cos above
    threshold) the basis is non-invertible for 2D localization.
    """
    verdict = PhaseMapVerdict(grid_dim=grid_dim)
    encode = getattr(encoder, "encode_spatial_grid", None)
    if encode is None:
        verdict.status = STATUS_NO_BASIS
        verdict.reason = "encoder lacks encode_spatial_grid"
        return verdict

    def wave_at(r: int, c: int) -> torch.Tensor:
        g = [[0] * grid_dim for _ in range(grid_dim)]
        g[r][c] = color
        w = encode(g).squeeze(0).reshape(-1).to(device).to(torch.float32)
        return F.normalize(w, p=2, dim=-1)

    with torch.no_grad():
        w12 = wave_at(1, 2)
        w21 = wave_at(2, 1)
        w11 = wave_at(1, 1)
        same_sum_cos = float(torch.dot(w12, w21).item())
        diff_sum_cos = float(torch.dot(w12, w11).item())

    verdict.same_sum_cos = same_sum_cos
    verdict.diff_sum_cos = diff_sum_cos
    if same_sum_cos > degenerate_cos_threshold:
        verdict.degeneracy_detected = True
        verdict.status = STATUS_NONINVERTIBLE
        verdict.reason = (
            f"x/y spatial ramps collinear: same-sum waves cos="
            f"{same_sum_cos:.6f} (threshold {degenerate_cos_threshold}); "
            f"(1,2) == (2,1) in wave space; fractional unbinding cannot "
            f"recover (x,y)"
        )
    else:
        verdict.status = STATUS_INVERTIBLE
        verdict.reason = (
            f"basis separates same-sum positions: cos={same_sum_cos:.6f}, "
            f"diff-sum cos={diff_sum_cos:.6f}"
        )
    return verdict


def _color_signature(color: int, half_d: int, device: str = "cpu") -> torch.Tensor:
    """Replicates HENRIVisionEncoder.color_codebook row for one color.

    color_angles = linspace(0, 2*pi*15/16, 16) -> entry `color` at
    theta = 2*pi*color/16; codebook = exp(1j*(theta * freqs)).
    """
    theta = 2 * math.pi * color / 16
    freqs = torch.arange(1, half_d + 1, dtype=torch.float32, device=device)
    return torch.exp(1j * (theta * freqs))


def fractional_unbind_coordinate(
    wave_real: torch.Tensor,
    encoder: object,
    color: int,
    grid_dim: int,
    device: str = "cpu",
) -> Tuple[int, int, float]:
    """Fractional position unbinding on a SEPARABLE basis (synthetic control).

    wave_real: [D] real wave of a single-pixel object of `color`.
    encoder: object exposing spatial_basis_x, spatial_basis_y (complex
    [max_grid_dim, D/2]) — used ONLY when the ramps differ.

    Returns (x*, y*, response_cos). Raises ValueError on a degenerate
    (collinear) basis — the protocol is undefined there.
    """
    basis_x = getattr(encoder, "spatial_basis_x", None)
    basis_y = getattr(encoder, "spatial_basis_y", None)
    if basis_x is None or basis_y is None:
        raise ValueError("encoder lacks spatial_basis_x/y")
    basis_x = basis_x.to(device)
    basis_y = basis_y.to(device)
    if basis_x.shape[0] < 2 or basis_y.shape[0] < 2:
        raise ValueError("undersized spatial basis")
    # Compare non-zero-coordinate carriers: coordinate 0 is always angle 0.
    if torch.allclose(
        torch.angle(basis_x[1]), torch.angle(basis_y[1]), atol=1e-4
    ) or basis_x.shape[0] < grid_dim or basis_y.shape[0] < grid_dim:
        raise ValueError("degenerate or undersized spatial basis (collinear ramps)")

    half = wave_real.numel() // 2
    z = torch.complex(wave_real[:half].to(torch.float32), wave_real[half:].to(torch.float32))
    z = F.normalize(z, p=2, dim=-1)
    sig = _color_signature(color, half, device)
    with torch.no_grad():
        residual = F.normalize(z / sig, p=2, dim=-1)  # color unbinding
        best_cos = -1.0
        best_rc = (0, 0)
        for r in range(grid_dim):
            for c in range(grid_dim):
                carrier = F.normalize(basis_x[c] * basis_y[r], p=2, dim=-1)
                s = float(torch.real(torch.vdot(carrier, residual)).item())
                if s > best_cos:
                    best_cos = s
                    best_rc = (r, c)
    return best_rc[0], best_rc[1], best_cos
