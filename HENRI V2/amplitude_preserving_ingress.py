"""Phase 8.13 — Amplitude-Preserving Complex Ingress (default-OFF, additive).

Implements the pre-registered Phase 8.13 mechanism (blueprint SHA
5e435cd9...): maps raw ARC-style integer grids directly to UN-NORMALIZED
complex waves in C^D. Background pixels (0) are masked to zero amplitude;
foreground occupancy is preserved in the complex MAGNITUDE channel
(Phase 8.12 kill lesson: amplitude IS the discriminative channel).

Composes with NativeComplexWaveTransition (8.11, amplitude-invariant:
forward_complex = Hadamard by unit phasor) — no acos lift.

Deviation D4 (documented in design doc): carriers are deterministic
incommensurate frequencies omega_x[d] = 2*pi*(d+1)*sqrt(2),
theta_y[d] = 2*pi*(d+1)*sqrt(3), replacing the blueprint sketch's randn
init (8.12 evidence: full-band random carriers are not reproducible and
the sketch's own gates are unreachable under unit-modulus constraints;
here amplitude is preserved so phase decorrelation ~1/sqrt(D)).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class AmplitudePreservingComplexIngress(nn.Module):
    """Amplitude-preserving incommensurate spatial ingress encoder.

    forward(grid [H, W] int) -> [1, D] complex64, un-normalized:
      z = sum_{k in fg} c_k * exp(j * (x_k * omega_x + y_k * theta_y))
    Background pixels (value 0) are masked to zero amplitude. Empty grid
    -> zero wave [1, D] complex64 (fail-closed, matches blueprint).
    """

    def __init__(
        self,
        dimension: int = 65536,
        num_blocks: int = 8192,
        block_size: int = 8,
        device: Optional[str] = None,
    ):
        super().__init__()
        d = num_blocks * block_size
        if d != dimension:
            raise ValueError(
                f"dimension {dimension} != num_blocks*block_size {d}"
            )
        self.dimension = dimension
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Deterministic incommensurate spatial frequency carriers (D4).
        # Irrational multiples of 2*pi -> incommensurate -> per-dimension
        # phase decorrelation for distinct pixel positions.
        k = torch.arange(1, dimension + 1, dtype=torch.float32)
        omega_x = (2.0 * math.pi * k * math.sqrt(2.0)) % (2.0 * math.pi)
        theta_y = (2.0 * math.pi * k * math.sqrt(3.0)) % (2.0 * math.pi)
        self.register_buffer("omega_x", omega_x.to(self._device))  # [D]
        self.register_buffer("theta_y", theta_y.to(self._device))  # [D]

    @torch.no_grad()
    def forward(self, grid_tensor: torch.Tensor) -> torch.Tensor:
        """Map a 2D integer grid [H, W] to an un-normalized complex wave.

        Returns [1, D] complex64 on self._device. Amplitude = foreground
        occupancy (color-weighted); background masked to zero.
        """
        grid = grid_tensor.to(self._device)
        H, W = grid.shape
        non_zero_mask = grid > 0  # [H, W] bool
        if not non_zero_mask.any():
            return torch.zeros(
                1, self.dimension, dtype=torch.complex64, device=self._device
            )
        y_indices, x_indices = torch.where(non_zero_mask)  # [N_fg]
        colors = grid[y_indices, x_indices].float()  # [N_fg]

        # exp(j * (x*omega_x + y*theta_y))  per foreground pixel [N_fg, D]
        x_phase = x_indices.float().unsqueeze(-1) * self.omega_x  # [N_fg, D]
        y_phase = y_indices.float().unsqueeze(-1) * self.theta_y  # [N_fg, D]
        spatial_wave = torch.polar(
            torch.ones_like(x_phase), x_phase + y_phase
        )  # [N_fg, D] complex64

        color_amplitude = colors.unsqueeze(-1)  # [N_fg, 1]
        weighted = color_amplitude * spatial_wave  # [N_fg, D]
        state_wave = torch.sum(weighted, dim=0, keepdim=True)  # [1, D]
        return state_wave

    def distinct_cosine(
        self, grid_a: torch.Tensor, grid_b: torch.Tensor
    ) -> float:
        """Complex cosine similarity between two grids' waves."""
        za = self.forward(grid_a).reshape(-1)
        zb = self.forward(grid_b).reshape(-1)
        num = torch.abs(torch.vdot(za, zb))
        den = torch.norm(za, p=2) * torch.norm(zb, p=2) + 1e-12
        return float(num / den)
