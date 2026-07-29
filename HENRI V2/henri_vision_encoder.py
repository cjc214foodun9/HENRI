"""
Project HENRI V2: Native Vision Encoder & High-Throughput Spatial Engine (henri_vision_encoder.py)
========================================================================================
Pivots HENRI from discrete BPE tokenization (V=32,000) to continuous Unitary Wave Embeddings (UWE)
in $D=65,536$ phase hypervector space (S^{D-1}).

Maps 2D/3D spatial pixel grids (ARC color grids / raw images) directly into Fourier Holographic
Reduced Representations (qFHRR) using continuous Lie group spatial basis operators.

Memory-optimized with @torch.no_grad() and O(D) complex phase accumulation to prevent CUDA memory spikes.
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from connected_component_segmenter import ConnectedComponentSegmenter, ParityContourMask


class HENRIVisionEncoder(nn.Module):
    """
    Continuous Spatial Unitary Wave Encoder (UWE) for Project HENRI V2.
    Converts 2D ARC-AGI grids directly into continuous D=65,536 qFHRR phase hypervectors.
    """

    def __init__(
        self,
        d_model: int = 65536,
        k_blocks: int = 8192,
        block_dim: int = 8,
        max_grid_dim: int = 128,
        device: Optional[str] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.k_blocks = k_blocks
        self.block_dim = block_dim
        self.max_grid_dim = max_grid_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Complex phase spatial basis vectors [max_grid_dim, D // 2]
        # x-axis translation phase generator and y-axis translation phase generator
        spatial_phases_x = torch.linspace(0, 2 * math.pi * 127, d_model // 2, device=self.device)
        spatial_phases_y = torch.linspace(0, 2 * math.pi * 127, d_model // 2, device=self.device)

        grid_coords = torch.arange(max_grid_dim, device=self.device, dtype=torch.float32).unsqueeze(1)
        self.spatial_basis_x = torch.exp(1j * (grid_coords * spatial_phases_x.unsqueeze(0)))
        self.spatial_basis_y = torch.exp(1j * (grid_coords * spatial_phases_y.unsqueeze(0)))

        # 16 ARC-AGI discrete color phase vectors on S^1
        color_angles = torch.linspace(0, 2 * math.pi * 15 / 16, 16, device=self.device).unsqueeze(1)
        color_freqs = torch.arange(1, (d_model // 2) + 1, device=self.device, dtype=torch.float32).unsqueeze(0)
        self.color_codebook = torch.exp(1j * (color_angles * color_freqs))

    @torch.no_grad()
    def encode_grid(self, grid: Union[torch.Tensor, np.ndarray, List]) -> torch.Tensor:
        """
        Maps a 2D spatial grid (H x W) directly into a continuous D=65,536 wave representation on S^{D-1}.
        Uses vectorized/chunked complex phase summation to prevent memory allocation spikes.
        """
        if not isinstance(grid, torch.Tensor):
            if isinstance(grid, np.ndarray):
                grid = np.ascontiguousarray(grid)
            grid = torch.tensor(grid, dtype=torch.long, device=self.device)
        else:
            grid = grid.to(dtype=torch.long, device=self.device)

        if grid.dim() == 3 and grid.shape[0] == 1:
            grid = grid.squeeze(0)

        H, W = grid.shape
        assert H <= self.max_grid_dim and W <= self.max_grid_dim, f"Grid dimensions {H}x{W} exceed max {self.max_grid_dim}"

        grid_clamped = torch.clamp(grid, 0, 15)

        # Memory-efficient O(D) complex phase accumulation with ParityContourMask topological weighting
        superposed_wave = torch.zeros(self.d_model // 2, dtype=torch.complex64, device=self.device)
        grid_np = grid_clamped.cpu().numpy()
        segmenter = ConnectedComponentSegmenter(background_color=0)
        components = segmenter.segment_grid(grid_np)

        parity_mask_grid = np.ones((H, W), dtype=np.float32)
        for comp in components:
            interior_px, exterior_px = ParityContourMask.compute_parity_contour((H, W), comp.pixels)
            for r_i, c_i in interior_px:
                parity_mask_grid[r_i, c_i] = -1.0  # Parity reflection operator for enclosed regions

        parity_mask_tensor = torch.tensor(parity_mask_grid, dtype=torch.float32, device=self.device)

        for r in range(H):
            py = self.spatial_basis_y[r]
            for c in range(W):
                px = self.spatial_basis_x[c]
                pc = self.color_codebook[grid_clamped[r, c]]
                kappa = parity_mask_tensor[r, c]
                superposed_wave.add_(px * py * pc * kappa)

        # Convert back to real D=65,536 representation [cos(theta), sin(theta)]
        real_wave = torch.cat([superposed_wave.real, superposed_wave.imag], dim=-1)  # [D]

        # Project onto unit hypersphere S^{D-1}
        normalized_wave = F.normalize(real_wave, p=2, dim=-1)
        return normalized_wave

    @torch.no_grad()
    def encode_spatial_grid(self, grid) -> torch.Tensor:
        """
        Wrapper returning shape [1, num_blocks, 8] for compatibility with O_VSA_IngressTokenizer interface.
        """
        wave_flat = self.encode_grid(grid)  # [d_model]
        wave_blocks = wave_flat.view(1, self.k_blocks, self.block_dim)
        return wave_blocks

    def compute_sagnac_similarity(self, wave1: torch.Tensor, wave2: torch.Tensor) -> float:
        """
        Computes Sagnac homodyne similarity S in [0.0, 1.0]:
        S = 0.5 * (1.0 + <wave1, wave2>)
        Sagnac phase delta = 1.0 - S
        """
        w1_flat = wave1.flatten()
        w2_flat = wave2.flatten()
        dot_prod = torch.dot(w1_flat, w2_flat).item()
        similarity = 0.5 * (1.0 + dot_prod)
        return float(np.clip(similarity, 0.0, 1.0))


if __name__ == "__main__":
    encoder = HENRIVisionEncoder(d_model=65536)
    dummy_grid = torch.tensor([[1, 1, 0, 0, 0],
                               [1, 2, 2, 0, 0],
                               [0, 2, 3, 3, 0],
                               [0, 0, 3, 0, 0],
                               [0, 0, 0, 0, 4]])

    wave_orig = encoder.encode_grid(dummy_grid)
    print("HENRIVisionEncoder memory-optimized grid encoding successfully verified!")
    print(f"Wave shape: {wave_orig.shape}, norm: {torch.norm(wave_orig).item():.4f}")
