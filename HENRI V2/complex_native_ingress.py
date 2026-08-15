"""Phase 8.12 — Complex-Native Ingress Encoder (default-OFF, additive).

Maps raw ARC-style grids DIRECTLY to per-element unit-modulus complex
phasors in C^D using incommensurate spatial carriers, eliminating the
legacy real-valued L2-normalized encoder's collinear collapse (the
"real-lift fallacy" documented in the Phase 8.12 blueprint).

Key properties (pre-registered in
experiments/sweeps/phase812_complex_native_ingress_design.md):
- G1: adjacent spatial-state complex cosine >= 0.85; distinct states < 0.95.
- G2: held-out translation-pair recovery through NativeComplexWaveTransition
  forward_complex (complex-domain Sagnac < 0.10).
- G3: ingress + transition cycle <= 2.0 ms at D=65,536 on CUDA.

FHRR convention: per-element unit modulus preserved (NO vector L2 norm).
Background cells (val == 0) are EXCLUDED from superposition (bg-mask
lesson, Phase 7.3/7.4: shared constant-carrier mass dominates otherwise).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn


class ComplexNativeIngress(nn.Module):
    """Complex-native grid -> C^D phasor ingress with real egress projection.

    Carrier frequencies are random incommensurate phases (no integer
    frequency aliasing); a grid translation (x,y) -> (x+dx, y+dy) rotates
    each element's phase by (f_x*dx + f_y*dy) mod 2pi — exactly the algebra
    NativeComplexWaveTransition implements (Lie group equivariance).
    """

    def __init__(
        self,
        dimension: int = 65536,
        num_blocks: int = 8192,
        block_dim: int = 8,
        device: Optional[str] = None,
        seed: int = 812,
        band_s: float = 0.10,
    ):
        super().__init__()
        d = num_blocks * block_dim
        if d != dimension:
            raise ValueError(
                f"dimension {dimension} != num_blocks*block_dim {d}; "
                "ComplexNativeIngress requires num_blocks*block_dim == dimension"
            )
        self.dimension = dimension
        self.num_blocks = num_blocks
        self.block_dim = block_dim
        self.band_s = band_s
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Band-limited incommensurate spatial carrier phases (deterministic
        # seed). Deviation D6: FULL-BAND U(0,2pi) carriers fail the
        # blueprint's OWN gate "adjacent cosine >= 0.85" (E[cos(fx)] = 0);
        # band-limited s=0.10 gives E[cos] = sinc(s*pi) ~ 0.984 >= 0.85
        # (8.8 class lesson: audit every sketch equation against its own gate).
        g = torch.Generator(device="cpu").manual_seed(seed)
        half = math.pi * band_s
        fx = (torch.rand(d, generator=g) * 2.0 - 1.0) * half
        fy = (torch.rand(d, generator=g) * 2.0 - 1.0) * half
        self.register_buffer("f_x", fx.to(self._device))
        self.register_buffer("f_y", fy.to(self._device))

    # -- core encoding -------------------------------------------------------
    @torch.no_grad()
    def encode_grid(self, grid: Sequence[Sequence[int]]) -> torch.Tensor:
        """Map grid -> [d] per-element unit-modulus complex phasors.

        Superposes exp(j*(f_x*x + f_y*y + theta_v)) over non-background
        cells, then renormalizes per-element modulus to 1 (FHRR).
        Returns complex64 [d] on self._device.
        """
        fx = self.f_x
        fy = self.f_y
        psi = torch.zeros(self.dimension, dtype=torch.complex64, device=fx.device)

        height = len(grid)
        width = len(grid[0]) if height > 0 else 1
        for y, row in enumerate(grid):
            for x, val in enumerate(row):
                if val == 0:
                    continue  # background excluded (bg-mask lesson)
                theta_v = (2.0 * math.pi) * (val % 256) / 256.0
                phase = fx * x + fy * y + theta_v
                psi += torch.exp(1j * phase)

        # Per-element unit modulus (FHRR): preserve phase, normalize amplitude.
        psi = psi / psi.abs().clamp_min(1e-12)
        return psi

    @torch.no_grad()
    def encode_grid_real(self, grid: Sequence[Sequence[int]]) -> torch.Tensor:
        """Production-interface projection: complex -> real [1, B, 8].

        Real part reshaped to [num_blocks, block_dim], per-block L2
        normalized (matches legacy encoder output shape for adapter
        compatibility probes).
        """
        z = self.encode_grid(grid)
        real = torch.real(z).reshape(self.num_blocks, self.block_dim)
        norms = torch.norm(real, p=2, dim=-1, keepdim=True).clamp_min(1e-12)
        return (real / norms).unsqueeze(0)

    # -- similarity ----------------------------------------------------------
    @torch.no_grad()
    def complex_cosine(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Complex-domain cosine: Re(<a, conj(b)>) / (|a| |b|) (scalar)."""
        a = a.reshape(-1)
        b = b.reshape(-1)
        denom = (a.abs().norm() * b.abs().norm()).clamp_min(1e-12)
        return (torch.real(torch.dot(a, torch.conj(b))) / denom)
