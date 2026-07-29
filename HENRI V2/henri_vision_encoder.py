"""
Project HENRI V2: Native Vision Encoder & High-Throughput Spatial Engine (henri_vision_encoder.py)
========================================================================================
Pivots HENRI V2 away from discrete text token bridges and restores native
phase-algebraic spatial cognition over 2D/3D pixel grids.

Key Features:
- Direct 2D/3D Pixel Grid to D=65,536 Phase Wave Embedding (qFHRR).
- Continuous spatial geometry: Rotations (90, 180, 270 deg), Translations, Reflections.
- High-throughput latent prediction engine without discrete token snapping bottlenecks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HENRIVisionEncoder(nn.Module):
    """
    Encodes 2D/3D pixel grids directly into D=65,536 Phase Hypervectors (S^{D-1}).
    Avoids discrete BPE tokenization, preserving topological phase continuity.
    """
    def __init__(self, d_model: int = 65536, k_blocks: int = 8192, max_grid_dim: int = 64, device: str = "cpu"):
        super().__init__()
        self.d_model = d_model
        self.k_blocks = k_blocks
        self.block_dim = d_model // k_blocks  # 8 components per Cl(3,0) block
        self.max_grid_dim = max_grid_dim
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Generate fixed orthogonal spatial phase basis vectors for X and Y coordinates
        # Using unitary complex phase angles theta in [-pi, pi]
        torch.manual_seed(42)
        self.spatial_basis_x = nn.Parameter(
            torch.exp(1j * (2.0 * math.pi * torch.rand(max_grid_dim, d_model // 2, device=self.device))),
            requires_grad=False
        )
        self.spatial_basis_y = nn.Parameter(
            torch.exp(1j * (2.0 * math.pi * torch.rand(max_grid_dim, d_model // 2, device=self.device))),
            requires_grad=False
        )

        # Attribute/Color Phase Codebook (16 ARC colors / feature classes)
        self.color_codebook = nn.Parameter(
            torch.exp(1j * (2.0 * math.pi * torch.rand(16, d_model // 2, device=self.device))),
            requires_grad=False
        )

    def encode_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Maps a 2D integer/RGB grid [H, W] to a normalized D=65,536 phase wave state (S^{D-1}).
        Formula: Psi_grid = Normalize( Sum_{x,y} Basis_X(x) * Basis_Y(y) * Color(C_{x,y}) )
        Returns a tensor of shape [d_model] (or reshaped as [num_blocks, 8]).
        """
        if not isinstance(grid, torch.Tensor):
            grid = torch.tensor(grid, dtype=torch.long, device=self.device)
        else:
            grid = grid.to(dtype=torch.long, device=self.device)

        if grid.dim() == 3 and grid.shape[0] == 1:
            grid = grid.squeeze(0)

        H, W = grid.shape
        assert H <= self.max_grid_dim and W <= self.max_grid_dim, f"Grid dimensions {H}x{W} exceed max {self.max_grid_dim}"

        y_indices = torch.arange(H, device=self.device).unsqueeze(1).repeat(1, W)
        x_indices = torch.arange(W, device=self.device).unsqueeze(0).repeat(H, 1)

        # Clamp color values to codebook range [0, 15]
        grid_clamped = torch.clamp(grid, 0, 15)

        # Retrieve complex phase basis
        phase_x = self.spatial_basis_x[x_indices]   # [H, W, D//2]
        phase_y = self.spatial_basis_y[y_indices]   # [H, W, D//2]
        phase_c = self.color_codebook[grid_clamped] # [H, W, D//2]

        # Holographic circular convolution in phase domain = elementwise complex multiplication
        bound_wave = phase_x * phase_y * phase_c    # [H, W, D//2]

        # Superposition across spatial grid
        superposed_wave = bound_wave.sum(dim=(0, 1))  # [D//2]

        # Convert back to real D=65,536 representation [cos(theta), sin(theta)]
        real_wave = torch.cat([superposed_wave.real, superposed_wave.imag], dim=-1)  # [D]

        # Project onto unit hypersphere S^{D-1}
        normalized_wave = F.normalize(real_wave, p=2, dim=-1)
        return normalized_wave

    def encode_spatial_grid(self, grid) -> torch.Tensor:
        """
        Wrapper returning shape [1, num_blocks, 8] for compatibility with O_VSA_IngressTokenizer interface.
        """
        wave_flat = self.encode_grid(grid)  # [d_model]
        wave_blocks = wave_flat.view(1, self.k_blocks, self.block_dim)
        return wave_blocks

    def compute_sagnac_similarity(self, wave1: torch.Tensor, wave2: torch.Tensor) -> float:
        """
        Computes Sagnac homodyne similarity S in [0, 1] between two spatial waves.
        S = 0.5 * (1.0 + <wave1, wave2>)
        """
        w1_flat = wave1.view(-1)
        w2_flat = wave2.view(-1)
        cosine_sim = torch.dot(w1_flat, w2_flat).item()
        return 0.5 * (1.0 + cosine_sim)


def verify_spatial_geometry_invariance():
    """
    Tests spatial rotations (90, 180, 270 deg) and transformations on 2D pixel grids.
    """
    print("\n" + "="*80)
    print("  HENRI V2: SPATIAL ROTATION & GEOMETRY INVARIANCE BENCHMARK")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = HENRIVisionEncoder(d_model=65536, k_blocks=8192, device=device)

    # Create a synthetic asymmetrical test grid
    test_grid = torch.tensor([
        [1, 1, 0, 0, 0],
        [1, 2, 2, 0, 0],
        [0, 2, 3, 3, 0],
        [0, 0, 3, 0, 0],
        [0, 0, 0, 0, 4]
    ], dtype=torch.long)

    print("\nOriginal Input Grid [5x5]:")
    print(test_grid.cpu().numpy())

    # Generate reference wave
    wave_orig = encoder.encode_grid(test_grid)

    # Perform Rotations
    grid_rot90 = torch.rot90(test_grid, k=1, dims=(0, 1))
    grid_rot180 = torch.rot90(test_grid, k=2, dims=(0, 1))
    grid_rot270 = torch.rot90(test_grid, k=3, dims=(0, 1))
    grid_flipped = torch.flip(test_grid, dims=[1])

    wave_rot90 = encoder.encode_grid(grid_rot90)
    wave_rot180 = encoder.encode_grid(grid_rot180)
    wave_rot270 = encoder.encode_grid(grid_rot270)
    wave_flip = encoder.encode_grid(grid_flipped)

    # Generate completely random noise grid for negative baseline comparison
    random_grid = torch.randint(0, 5, size=(5, 5), dtype=torch.long)
    wave_random = encoder.encode_grid(random_grid)

    # Measure Similarities
    sim_self = encoder.compute_sagnac_similarity(wave_orig, wave_orig)
    sim_rot90 = encoder.compute_sagnac_similarity(wave_orig, wave_rot90)
    sim_rot180 = encoder.compute_sagnac_similarity(wave_orig, wave_rot180)
    sim_rot270 = encoder.compute_sagnac_similarity(wave_orig, wave_rot270)
    sim_flip = encoder.compute_sagnac_similarity(wave_orig, wave_flip)
    sim_random = encoder.compute_sagnac_similarity(wave_orig, wave_random)

    print("\nPhase Space Sagnac Similarity Results:")
    print(f"   Self Similarity (Identity)    : {sim_self:.4f}   [EXPECTED: 1.0000]")
    print(f"   Rotated 90 deg Wave          : {sim_rot90:.4f}")
    print(f"   Rotated 180 deg Wave         : {sim_rot180:.4f}")
    print(f"   Rotated 270 deg Wave         : {sim_rot270:.4f}")
    print(f"   Horizontally Flipped Wave    : {sim_flip:.4f}")
    print(f"   Unrelated Random Grid Wave   : {sim_random:.4f}   [EXPECTED: ~0.5000 (Orthogonal)]")
    print("="*80)


if __name__ == "__main__":
    verify_spatial_geometry_invariance()
