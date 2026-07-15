import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FractionalBindingLayer(nn.Module):
    """
    ENGINEERING SPECIFICATION: PROJECT HENRI - FRACTIONAL COORDINATE BINDING (V1.0.0)
    Implements continuous-phase fractional binding in the spectral domain.
    X^x = iFFT( FFT(X)^x )
    """
    def __init__(self, dim=4096):
        super().__init__()
        self.dim = dim
        # Base phase carriers for X and Y coordinates (initialized on the unit hypersphere)
        # We generate random angles and use them as the spectral phase components.
        # Ensure Hermitian symmetry so the time-domain signals are strictly real.
        
        # Real-domain random vectors
        x_base_fft = torch.fft.rfft(torch.randn(dim))
        y_base_fft = torch.fft.rfft(torch.randn(dim))
        
        # We need to constrain the phase to a narrow band to prevent 2*pi aliasing 
        # when multiplying by x (which ranges up to grid size, e.g., 30).
        # Max phase angle should be roughly pi / max_x
        max_x = 30.0
        x_angle = torch.angle(x_base_fft) * (np.pi / (max_x * np.pi)) # bound to [-1/max_x, 1/max_x]
        y_angle = torch.angle(y_base_fft) * (np.pi / (max_x * np.pi))
        
        # Reconstruct as pure phase operators (magnitude 1.0)
        self.register_buffer('X_fft', torch.polar(torch.ones_like(x_angle), x_angle))
        self.register_buffer('Y_fft', torch.polar(torch.ones_like(y_angle), y_angle))

    def bind_coordinate(self, obj_identity: torch.Tensor, x: float, y: float) -> torch.Tensor:
        """
        Binds an object identity vector to a continuous spatial (x,y) coordinate.
        Ψ_spatial = O ⊛ X^x ⊛ Y^y
        """
        # 1. To spectral domain
        obj_fft = torch.fft.rfft(obj_identity)
        
        # 2. Fractional exponentiation in spectral domain is just raising the complex phase
        # FFT(X)^x = exp(i * x * angle(FFT(X))) since |FFT(X)| == 1
        x_phase = self.X_fft ** x
        y_phase = self.Y_fft ** y
        
        # 3. Spectral binding (Hadamard product)
        bound_fft = obj_fft * x_phase * y_phase
        
        # 4. Return to spatial domain
        bound_spatial = torch.fft.irfft(bound_fft, n=self.dim)
        return bound_spatial

class BlockSparseFeaturizer(nn.Module):
    """
    ENGINEERING SPECIFICATION: PROJECT HENRI - GRASSMANNIAN BLOCK-SPARSE FEATURIZER
    Replaces 1D point-attractor projections with continuous Grassmannian Subspaces.
    z_BSF = Σ (W_c * W_c^T) * z
    """
    def __init__(self, dim=4096, num_blocks=16, block_rank=64):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.block_rank = block_rank
        
        # Initialize B block projection bases W_c ∈ R^(dim x rank)
        self.blocks = nn.Parameter(torch.randn(num_blocks, dim, block_rank))
        self.reset_parameters()

    def reset_parameters(self):
        # Stiefel manifold constraint: orthogonalize the blocks using QR decomposition
        with torch.no_grad():
            for c in range(self.num_blocks):
                q, r = torch.linalg.qr(self.blocks[c])
                self.blocks[c].copy_(q)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Projects wave state z onto the active Grassmannian subspaces.
        """
        z_out = torch.zeros_like(z)
        for c in range(self.num_blocks):
            W_c = self.blocks[c]  # (dim, rank)
            # W_c^T * z
            proj = torch.matmul(z, W_c) # (batch, rank) if z is batched, or (rank,)
            # W_c * (W_c^T * z)
            subspace_vector = torch.matmul(proj, W_c.T)
            z_out += subspace_vector
            
        # Optional: Normalize the superposition back to the unit hypersphere
        return F.normalize(z_out, p=2, dim=-1)

def run_clean_room_verification():
    print("--- INITIATING FRACTIONAL BINDING VERIFICATION PROBE ---")
    dim = 4096
    
    binder = FractionalBindingLayer(dim=dim)
    featurizer = BlockSparseFeaturizer(dim=dim, num_blocks=8, block_rank=32)
    
    obj_identity = F.normalize(torch.randn(dim), p=2, dim=0)
    
    # 1. Test Photonic Isomorphism Coefficient (Σ)
    # Generate points along a continuous trajectory (e.g., sliding along the X axis)
    points = np.linspace(0.0, 10.0, 50)
    waves = []
    
    for x in points:
        wave = binder.bind_coordinate(obj_identity, x, 0.0)
        # Pass through the Grassmannian Featurizer
        wave_bsf = featurizer(wave)
        waves.append(wave_bsf)
        
    waves = torch.stack(waves)
    
    # Compute the phase distance (1.0 - cosine_similarity) between the first point and all others
    cos_sims = F.cosine_similarity(waves[0:1], waves, dim=1)
    phase_distances = 1.0 - cos_sims.detach().numpy()
    
    # The spatial distances from the first point
    spatial_distances = points - points[0]
    
    # Σ is the Pearson correlation between spatial distances and phase distances
    # It proves the geometry is preserved in the high-dimensional space
    sigma = np.corrcoef(spatial_distances, phase_distances)[0, 1]
    
    print(f"[METRIC] Photonic Isomorphism Coefficient (Sigma): {sigma:.6f}")
    if sigma > 0.90:
        print("  -> [PASS] Continuous geometry is linearly preserved in phase space.")
    else:
        print("  -> [FAIL] Representation is collapsing or highly non-linear.")
        
    # 2. Test Reconstruction Fidelity Index (F)
    # Bind an object, then unbind it using the exact inverse fractional shift (-x, -y)
    test_x, test_y = 3.14, 2.71
    wave_bound = binder.bind_coordinate(obj_identity, test_x, test_y)
    wave_unbound = binder.bind_coordinate(wave_bound, -test_x, -test_y)
    
    fidelity = F.cosine_similarity(obj_identity, wave_unbound, dim=0).item()
    print(f"[METRIC] Reconstruction Fidelity Index (F): {fidelity:.6f}")
    if fidelity > 0.99:
        print("  -> [PASS] Perfect wave reconstruction achieved via conjugate unbinding.")
    else:
        print("  -> [FAIL] Energy lost during fractional binding process.")

if __name__ == '__main__':
    run_clean_room_verification()
