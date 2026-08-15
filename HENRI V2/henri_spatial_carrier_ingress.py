"""
HENRI V2 Phase 8.8 — Vectorized Incommensurate Spatial Carrier Ingress (CC-OS Masked).

Representation repair for the FALSIFIED seeded-random state projection
(Phase 8.7 seal @ 2708aac): restores Lie-group spatial equivariance in the
state wavefront.

Mechanism (blueprint `docs-HENRI_V2_PHASE_8_7_POSTMORTEM_AND_PHASE_8_8_S....pdf.pdf`,
raw SHA a07eb7d3...; gates unchanged):

    Psi_space(x, y) = e^{j (x * Omega_x + y * Theta_y)}
    Psi_Ok = Psi_identity(c_k) * e^{j (x_k * Omega_x + y_k * Theta_y)}
    Psi_state = sum_k Psi_Ok / || . ||

Uniform rigid translation is an EXACT global phase rotation (element-wise
complex product distributes over the superposition sum):

    Psi(x+dx, y+dy) = Psi(x, y) * e^{j (dx*Omega_x + dy*Theta_y)}

DEVIATION (documented pre-launch in experiments/sweeps/phase88_spatial_carriers_design.md):
the blueprint sketch draws Omega_x ~ randn(D)*pi, which yields E[cos(Omega_d)] ~ 0.007
and FAILS its own gate-1 (>= 0.85) by construction. This implementation uses
band-limited incommensurate carriers with s = 0.10 cycles/px:
E[cos(Omega_d)] = sinc(2*pi*s) ~= 0.937 >= 0.85. Mechanism preserved; only the
carrier bandwidth is corrected to satisfy the stated gate.

Complexity: O(N_objects * D/2) complex ops, fully vectorized (no per-pixel
Python loops); the per-object loop is replaced by stacked tensor ops.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

from connected_component_segmenter import ConnectedComponentSegmenter


_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]  # color 1..10 identity bases


class VectorizedIncommensurateSpatialIngress(torch.nn.Module):
    """CC-OS masked incommensurate spatial phase carrier ingress (Phase 8.8-A/B)."""

    def __init__(
        self,
        dimension: int = 65536,
        carrier_scale: float = 0.10,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert dimension % 2 == 0, "dimension must be even (Re ‖ Im halves)"
        self.dimension = dimension
        self.half_d = dimension // 2
        self.carrier_scale = carrier_scale
        self.device = device if device is not None else torch.device("cpu")
        self.segmenter = ConnectedComponentSegmenter(background_color=0)

        d = torch.arange(self.half_d, dtype=torch.float64, device=self.device)
        tau = (math.sqrt(5.0) - 1.0) / 2.0  # golden ratio conjugate: incommensurate
        # Band-limited incommensurate carriers: E[cos] = sinc(2*pi*s) ~= 0.937 @ s=0.10.
        omega = 2.0 * math.pi * carrier_scale * torch.frac(d * tau)
        theta = 2.0 * math.pi * carrier_scale * torch.frac(d * math.sqrt(3.0))
        self.register_buffer("omega_x", omega.to(torch.float32), persistent=False)
        self.register_buffer("theta_y", theta.to(torch.float32), persistent=False)

        # Per-color identity carrier: I_c = e^{j 2*pi*frac(d*sqrt(p_c))}, c=1..10.
        identity_phase = torch.zeros(11, self.half_d, dtype=torch.float32, device=self.device)
        for c in range(1, 11):
            p = float(_PRIMES[(c - 1) % len(_PRIMES)])
            phase = 2.0 * math.pi * torch.frac(d * math.sqrt(p))
            identity_phase[c] = phase.to(torch.float32)
        self.register_buffer("_identity_phase", identity_phase, persistent=False)

    # -- carriers ---------------------------------------------------------
    def translation_wave(self, dx: float, dy: float) -> torch.Tensor:
        """Exact global phase-rotation wave e^{j(dx*Omega + dy*Theta)}: complex [half_d]."""
        phase = dx * self.omega_x + dy * self.theta_y
        return torch.polar(torch.ones_like(phase), phase)

    # -- core encoders ----------------------------------------------------
    def _object_phases(self, colors: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
        """[N, half_d] complex phases for a batch of objects at (cx, cy) with colors."""
        identity = self._identity_phase[colors]  # [N, half_d]
        spatial = cx.unsqueeze(-1) * self.omega_x + cy.unsqueeze(-1) * self.theta_y
        return identity + spatial

    def encode_objects(
        self,
        object_masks: torch.Tensor,   # [N, H, W] (kept for blueprint API parity; not required)
        object_types: torch.Tensor,   # [N, half_d] complex identity waves
        centroids: torch.Tensor,      # [N, 2] (x, y)
    ) -> torch.Tensor:
        """Blueprint sketch API: [N, D_complex] identity * e^{j(x*Omega + y*Theta)} -> [1, D] unit real."""
        x = centroids[:, 0].unsqueeze(-1)
        y = centroids[:, 1].unsqueeze(-1)
        spatial_phase = x * self.omega_x + y * self.theta_y
        spatial_wave = torch.polar(torch.ones_like(spatial_phase), spatial_phase)
        bound = object_types * spatial_wave          # [N, half_d]
        state_wave = bound.sum(dim=0, keepdim=True)  # [1, half_d]
        return self._to_real_unit(state_wave)        # [1, D]

    def encode_grid(self, grid) -> torch.Tensor:
        """CC-OS segment + centroid-carrier superposition -> [D] real unit wave.

        Fails closed on empty foreground (all-background grid).
        """
        comps = self.segmenter.segment_grid(np.asarray(grid, dtype=int))
        if not comps:
            raise ValueError(
                "empty foreground: CC-OS found no objects; carrier superposition "
                "would be zero (fail-closed)"
            )
        colors = torch.tensor([min(max(c.color, 0), 10) for c in comps], device=self.device)
        cx = torch.tensor([c.tracking_key[1] for c in comps], dtype=torch.float32, device=self.device)
        cy = torch.tensor([c.tracking_key[0] for c in comps], dtype=torch.float32, device=self.device)
        phases = self._object_phases(colors, cx, cy)          # [N, half_d]
        acc = torch.exp(1j * phases).sum(dim=0)               # [half_d] complex
        return self._to_real_unit(acc.unsqueeze(0)).squeeze(0)  # [D]

    def encode_single_object(self, color: int, cx: float, cy: float) -> torch.Tensor:
        """Single-object state wave: I_c * e^{j(cx*Omega + cy*Theta)} -> [D] real unit.

        Used by the physics-env `state_to_wave(use_carrier_ingress=True)` path:
        continuous state coordinates map directly to continuous carrier centroids
        (no grid quantization).
        """
        colors = torch.tensor([int(color)], device=self.device)
        cx_t = torch.tensor([float(cx)], dtype=torch.float32, device=self.device)
        cy_t = torch.tensor([float(cy)], dtype=torch.float32, device=self.device)
        phases = self._object_phases(colors, cx_t, cy_t)  # [1, half_d]
        acc = torch.exp(1j * phases).sum(dim=0)           # [half_d]
        return self._to_real_unit(acc.unsqueeze(0)).squeeze(0)  # [D]

    def apply_translation(self, wave: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
        """Exact phase rotation of an encoded [D] real wave by (dx, dy)."""
        complex_wave = wave[: self.half_d] + 1j * wave[self.half_d :]
        rotated = complex_wave * self.translation_wave(dx, dy)
        return self._to_real_unit(rotated.unsqueeze(0)).squeeze(0)

    @staticmethod
    def _to_real_unit(complex_wave: torch.Tensor) -> torch.Tensor:
        """[..., half_d] complex -> [..., D] real unit wave."""
        real_wave = torch.cat([complex_wave.real, complex_wave.imag], dim=-1)
        return F.normalize(real_wave, p=2, dim=-1)

    def to_blocks(self, wave: torch.Tensor, num_blocks: int = 8192) -> torch.Tensor:
        """[D] real -> [num_blocks, 8] real, per-block unit norm (planner boundary)."""
        blocks = wave.reshape(num_blocks, 8)
        return F.normalize(blocks, p=2, dim=-1)

    # -- convenience --------------------------------------------------------
    def mean_cos_carrier(self) -> float:
        """E[cos(Omega_d)] over the carrier — the gate-1 continuity anchor."""
        return float(torch.cos(self.omega_x).mean().item())
