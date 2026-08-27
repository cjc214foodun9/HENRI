"""Contract tests: ARC Spatial Phase-Map Verifier (Phase 7.2 Step 3). CPU-only."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from arc_phase_map import (
    STATUS_NONINVERTIBLE,
    STATUS_NO_BASIS,
    _color_signature,
    fractional_unbind_coordinate,
    verify_phase_map_invertibility,
)


class _CollinearEncoder:
    """Replicates the production defect: spatial_phases_x == spatial_phases_y."""

    def __init__(self, d_model=512, max_grid_dim=8):
        half = d_model // 2
        ph = torch.linspace(0, 2 * 3.141592653589793 * 127, half)
        coords = torch.arange(max_grid_dim, dtype=torch.float32).unsqueeze(1)
        self.spatial_basis_x = torch.exp(1j * (coords * ph.unsqueeze(0)))
        self.spatial_basis_y = torch.exp(1j * (coords * ph.unsqueeze(0)))
        self.d_model = d_model

    def encode_spatial_grid(self, grid):
        import math
        half = self.d_model // 2
        ph = torch.linspace(0, 2 * math.pi * 127, half)
        color_angles = torch.linspace(0, 2 * math.pi * 15 / 16, 16)[:, None]
        color_freqs = torch.arange(1, half + 1, dtype=torch.float32)[None, :]
        codebook = torch.exp(1j * (color_angles * color_freqs))
        wave = torch.zeros(half, dtype=torch.complex64)
        H, W = len(grid), len(grid[0])
        for r in range(H):
            for c in range(W):
                v = int(grid[r][c])
                if v > 0:
                    wave = wave + torch.exp(1j * (c * ph + r * ph)) * codebook[v]
        real = torch.cat([wave.real, wave.imag], dim=-1)
        return torch.nn.functional.normalize(real, p=2, dim=-1).view(1, self.d_model // 8, 8)


class _SeparableEncoder(_CollinearEncoder):
    def __init__(self, d_model=512, max_grid_dim=8):
        super().__init__(d_model, max_grid_dim)
        half = d_model // 2
        phy = torch.linspace(0, 2 * 3.141592653589793 * 89, half)  # different ramp
        coords = torch.arange(max_grid_dim, dtype=torch.float32).unsqueeze(1)
        self.spatial_basis_y = torch.exp(1j * (coords * phy.unsqueeze(0)))

    def encode_spatial_grid(self, grid):
        import math
        half = self.d_model // 2
        phx = torch.linspace(0, 2 * math.pi * 127, half)
        phy = torch.linspace(0, 2 * math.pi * 89, half)
        color_angles = torch.linspace(0, 2 * math.pi * 15 / 16, 16)[:, None]
        color_freqs = torch.arange(1, half + 1, dtype=torch.float32)[None, :]
        codebook = torch.exp(1j * (color_angles * color_freqs))
        wave = torch.zeros(half, dtype=torch.complex64)
        H, W = len(grid), len(grid[0])
        for r in range(H):
            for c in range(W):
                v = int(grid[r][c])
                if v > 0:
                    wave = wave + torch.exp(1j * (c * phx + r * phy)) * codebook[v]
        real = torch.cat([wave.real, wave.imag], dim=-1)
        return torch.nn.functional.normalize(real, p=2, dim=-1).view(1, half // 4, 8)


def test_collinear_basis_detected_non_invertible():
    enc = _CollinearEncoder()
    verdict = verify_phase_map_invertibility(enc, grid_dim=4, color=5, device="cpu")
    assert verdict.status == STATUS_NONINVERTIBLE
    assert verdict.degeneracy_detected
    assert verdict.same_sum_cos > 0.9


def test_separable_basis_passes_verdict():
    enc = _SeparableEncoder()
    verdict = verify_phase_map_invertibility(enc, grid_dim=4, color=5, device="cpu")
    assert verdict.status != STATUS_NONINVERTIBLE
    assert not verdict.degeneracy_detected


def test_fractional_unbind_recovers_on_separable():
    enc = _SeparableEncoder(d_model=1024, max_grid_dim=8)
    grid = [[0] * 6 for _ in range(6)]
    grid[3][2] = 7  # object at (r=3, c=2), color 7
    wave = enc.encode_spatial_grid(grid).view(-1)
    r, c, cos = fractional_unbind_coordinate(wave, enc, color=7, grid_dim=6, device="cpu")
    assert (r, c) == (3, 2), f"recovered {(r, c)} != (3, 2) with cos {cos:.4f}"
    assert cos > 0.5


def test_fractional_unbind_rejects_collinear():
    enc = _CollinearEncoder(d_model=1024, max_grid_dim=8)
    grid = [[0] * 6 for _ in range(6)]
    grid[3][2] = 7
    wave = enc.encode_spatial_grid(grid).view(-1)
    with pytest.raises(ValueError):
        fractional_unbind_coordinate(wave, enc, color=7, grid_dim=6, device="cpu")


def test_no_basis_verdict():
    class _NoBasis:
        pass  # no encode_spatial_grid at all

    v = verify_phase_map_invertibility(_NoBasis(), grid_dim=2)
    assert v.status == STATUS_NO_BASIS


def test_color_signature_phase_formula():
    sig = _color_signature(0, 64)
    assert torch.isfinite(sig).all()
    assert sig.dtype == torch.complex64
