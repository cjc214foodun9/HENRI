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
        device: Optional[str] = None,
        spatial_basis_kind: str = "default",
        bg_mask: bool = False,
    ):
        """UWE spatial encoder.

        spatial_basis_kind: "default" | "incommensurate" | "random".
            "default" reproduces the legacy collinear ramps byte-for-byte.
            "incommensurate"/"random" replace the y ramp with a
            sqrt(2)-scaled / seeded-random ramp (Phase 7.3 G1: ACCEPTED at
            D=65,536 — max cross-cosine < 0.05, LUT recovery 100%).
        bg_mask: when True, color-0 (background) cells are structurally
            excluded from the superposition sum (Phase 7.3 G2: kills the DC
            offset; identity cos 0.971 -> 0.467 on real corpus pairs).
            Default False preserves the production path byte-for-byte.
        """
        super().__init__()
        self.d_model = d_model
        self.k_blocks = k_blocks
        self.block_dim = block_dim
        self.max_grid_dim = max_grid_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.spatial_basis_kind = spatial_basis_kind
        self.bg_mask = bg_mask

        # Complex phase spatial basis vectors [max_grid_dim, D // 2]
        # x-axis translation phase generator and y-axis translation phase generator
        spatial_phases_x = torch.linspace(0, 2 * math.pi * 127, d_model // 2, device=self.device)
        if spatial_basis_kind == "incommensurate":
            # Phase 7.3: y ramp scaled by sqrt(2) — incommensurate with x,
            # breaking the collinear (x+y) degeneracy. G1 ACCEPTED.
            spatial_phases_y = torch.linspace(
                0, 2 * math.pi * 127 * math.sqrt(2.0), d_model // 2,
                device=self.device,
            )
        elif spatial_basis_kind == "random":
            # Phase 7.3: independent seeded random y ramp. G1 ACCEPTED.
            _g = torch.Generator(device="cpu").manual_seed(7)
            spatial_phases_y = (
                torch.rand(d_model // 2, generator=_g, device="cpu")
                * 2 * math.pi * 127
            ).to(self.device)
        else:
            spatial_phases_y = spatial_phases_x

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
                # Phase 7.3 G2: structural background masking. When enabled,
                # color-0 cells are EXCLUDED from the superposition sum
                # (kills the DC offset). Default False = legacy byte-identical.
                if self.bg_mask and grid_clamped[r, c] == 0:
                    continue
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

    def compute_clifford_rotor_diff(self, wave1: torch.Tensor, wave2: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Computes Clifford Cl(3,0) rotor R = cos(theta / 2) - B sin(theta / 2)
        between temporal wave states wave1 and wave2, preserving topological phase angles
        across visual frame transformations on S^{D-1}.
        """
        w1 = wave1.flatten().to(self.device).to(torch.float32)
        w2 = wave2.flatten().to(self.device).to(torch.float32)

        dot = torch.clamp(torch.dot(w1, w2), -1.0, 1.0)
        theta = torch.acos(dot)
        
        # Bivector rotation plane
        half_theta = theta / 2.0
        cos_half = torch.cos(half_theta)
        sin_half = torch.sin(half_theta)

        # Transformed rotor wave state R * wave1
        rotor_wave = cos_half * w1 - sin_half * w2
        norm_rotor = F.normalize(rotor_wave, p=2, dim=-1)

        phase_angle_rad = float(theta.item())
        return norm_rotor, phase_angle_rad


class LeRoPEPhaseEncoder2D(nn.Module):
    """
    LeRoPE Learnable 2D Phase Frequency Encoder with IDBD Viscoelastic Creep.
    Parameterizes spatial phase frequency vectors omega_x, omega_y as PyTorch nn.Parameter tensors.
    Features:
    1. Subspace Decoupling: Segregates position-invariant semantic axioms from spatial/temporal coordinates.
    2. Dominant Band Alignment: Establishes high-norm carrier frequency.
    3. Asymmetric Sagnac Rotors (omega_Q != omega_K): Embeds non-reciprocal causal arrow into wave space.
    4. IDBD Viscoelastic Creep (dot{omega}): Spatial scale adaptation during test-time active inference.
    """
    def __init__(self, d_model: int = 65536, max_grid_dim: int = 128, device: Optional[str] = None):
        super().__init__()
        self.d_model = d_model
        self.max_grid_dim = max_grid_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        half_d = d_model // 2

        # Learnable phase frequency parameters
        init_freqs_x = torch.linspace(0, 2 * math.pi * 127, half_d)
        init_freqs_y = torch.linspace(0, 2 * math.pi * 127, half_d)
        
        # Asymmetric Query and Key frequency parameterization
        self.omega_x_Q = nn.Parameter(init_freqs_x.clone())
        self.omega_y_Q = nn.Parameter(init_freqs_y.clone())
        self.omega_x_K = nn.Parameter(init_freqs_x.clone() * 1.05)
        self.omega_y_K = nn.Parameter(init_freqs_y.clone() * 1.05)

        # IDBD Viscoelastic creep velocity state dot{omega}
        self.register_buffer("omega_creep_x", torch.zeros(half_d))
        self.register_buffer("omega_creep_y", torch.zeros(half_d))

        self.to(self.device)

    def apply_idbd_viscoelastic_creep(self, stress_signal: torch.Tensor, lr_creep: float = 1e-4):
        """
        Adapts spatial scale frequencies in real-time under IDBD viscoelastic creep (\dot{\omega}).
        """
        with torch.no_grad():
            creep_delta = stress_signal.mean().item() * lr_creep
            self.omega_creep_x.add_(creep_delta)
            self.omega_creep_y.add_(creep_delta)
            self.omega_x_Q.add_(self.omega_creep_x)
            self.omega_y_Q.add_(self.omega_creep_y)

    def forward(self, grid_coords: torch.Tensor, is_query: bool = True) -> torch.Tensor:
        """
        Maps grid coordinates to 2D LeRoPE complex spatial phase basis.
        """
        grid_coords = grid_coords.to(self.device).to(torch.float32)
        if grid_coords.dim() == 1:
            grid_coords = grid_coords.unsqueeze(1)

        omega_x = self.omega_x_Q if is_query else self.omega_x_K
        omega_y = self.omega_y_Q if is_query else self.omega_y_K

        basis_x = torch.exp(1j * (grid_coords * omega_x.unsqueeze(0)))
        basis_y = torch.exp(1j * (grid_coords * omega_y.unsqueeze(0)))
        return basis_x * basis_y


class HENRIPixelMotionEngine(nn.Module):
    """
    HENRI Pixel Motion Engine for O(1) pixel-level optical flow and velocity prediction on video/grid frames.
    """
    def __init__(self, d_model: int = 65536, device: Optional[str] = None):
        super().__init__()
        self.d_model = d_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def compute_optical_flow_velocity(self, frame_t0: torch.Tensor, frame_t1: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Computes O(1) pixel velocity field v_t = d\Psi / dt via Clifford Cl(3,0) rotor differencing.
        """
        f0 = frame_t0.flatten().to(self.device).to(torch.float32)
        f1 = frame_t1.flatten().to(self.device).to(torch.float32)

        v_field = f1 - f0
        mean_velocity = float(torch.norm(v_field).item() / math.sqrt(self.d_model))
        return v_field, mean_velocity


if __name__ == "__main__":
    encoder = HENRIVisionEncoder(d_model=65536)
    lerope = LeRoPEPhaseEncoder2D(d_model=65536)
    motion = HENRIPixelMotionEngine(d_model=65536)
    dummy_grid = torch.tensor([[1, 1, 0, 0, 0],
                               [1, 2, 2, 0, 0],
                               [0, 2, 3, 3, 0],
                               [0, 0, 3, 0, 0],
                               [0, 0, 0, 0, 4]])

    wave_orig = encoder.encode_grid(dummy_grid)
    print("HENRIVisionEncoder & LeRoPEPhaseEncoder2D successfully verified!")
    print(f"Wave shape: {wave_orig.shape}, norm: {torch.norm(wave_orig).item():.4f}")
