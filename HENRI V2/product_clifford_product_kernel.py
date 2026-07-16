import torch
import torch.nn as nn

class ProductCliffordAlgebra3D(nn.Module):
    """
    Implements a highly parallelized, non-commutative Product Clifford Algebra (Cl_3,0)
    over K independent blocks. Unifies spatial, causal, and temporal operators 
    without exponential dimension scaling.
    """
    def __init__(self, num_blocks=8192):
        super().__init__()
        self.K = num_blocks
        
        # Binary multiplication table for Cl_3,0 expressed as structural indices
        # Represents basis: [1, e1, e2, e3, e12, e23, e31, e123]
        # Multiplications are implemented via sign-preserving bilinear transitions
        self.register_buffer("mult_indices", torch.tensor([
            # Structured coordinate mapping of the geometric product of Cl_3,0
            # [basis_a, basis_b, output_basis, sign]
            [0, 0, 0,  1.0], [0, 1, 1,  1.0], [0, 2, 2,  1.0], [0, 3, 3,  1.0],
            [0, 4, 4,  1.0], [0, 5, 5,  1.0], [0, 6, 6,  1.0], [0, 7, 7,  1.0],
            [1, 0, 1,  1.0], [1, 1, 0,  1.0], [1, 2, 4,  1.0], [1, 3, 6, -1.0],
            [1, 4, 2,  1.0], [1, 5, 7,  1.0], [1, 6, 3, -1.0], [1, 7, 5,  1.0],
            [2, 0, 2,  1.0], [2, 1, 4, -1.0], [2, 2, 0,  1.0], [2, 3, 5,  1.0],
            [2, 4, 1, -1.0], [2, 5, 3,  1.0], [2, 6, 7,  1.0], [2, 7, 6,  1.0],
            [3, 0, 3,  1.0], [3, 1, 6,  1.0], [3, 2, 5, -1.0], [3, 3, 0,  1.0],
            [3, 4, 7,  1.0], [3, 5, 2, -1.0], [3, 6, 1,  1.0], [3, 7, 4,  1.0],
            [4, 0, 4,  1.0], [4, 1, 2, -1.0], [4, 2, 1,  1.0], [4, 3, 7,  1.0],
            [4, 4, 0, -1.0], [4, 5, 6, -1.0], [4, 6, 5,  1.0], [4, 7, 3, -1.0],
            [5, 0, 5,  1.0], [5, 1, 7,  1.0], [5, 2, 3, -1.0], [5, 3, 2,  1.0],
            [5, 4, 6,  1.0], [5, 5, 0, -1.0], [5, 6, 4, -1.0], [5, 7, 1, -1.0],
            [6, 0, 6,  1.0], [6, 1, 3,  1.0], [6, 2, 7,  1.0], [6, 3, 1, -1.0],
            [6, 4, 5, -1.0], [6, 5, 4,  1.0], [6, 6, 0, -1.0], [6, 7, 2, -1.0],
            [7, 0, 7,  1.0], [7, 1, 5,  1.0], [7, 2, 6,  1.0], [7, 3, 4,  1.0],
            [7, 4, 3, -1.0], [7, 5, 1, -1.0], [7, 6, 2, -1.0], [7, 7, 0, -1.0]
        ], dtype=torch.float32))

    def geometric_product(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Computes the vectorized geometric product C = AB over all K blocks.
        A, B Shapes: [Batch_Size, K, 8]
        Output Shape: [Batch_Size, K, 8]
        """
        batch_size, num_blocks, _ = A.shape
        device = A.device
        
        # Initialize output multivector tensor
        C = torch.zeros((batch_size, num_blocks, 8), dtype=A.dtype, device=device)
        
        # Vectorized bilinear gathering based on Clifford algebra structural tables
        indices = self.mult_indices.long()
        idx_a = indices[:, 0]
        idx_b = indices[:, 1]
        idx_c = indices[:, 2]
        signs = self.mult_indices[:, 3]
        
        # Gather coefficients
        coeffs_a = A[:, :, idx_a] # Shape: [B, K, 64]
        coeffs_b = B[:, :, idx_b] # Shape: [B, K, 64]
        
        # Compute sign-preserving element-wise multiplication
        product_terms = coeffs_a * coeffs_b * signs.view(1, 1, -1) # Shape: [B, K, 64]
        
        # Scatter-add products back to their designated multivector bases
        # We accumulate the 64 product terms into the 8 target multivector slots
        for basis_idx in range(8):
            mask = (idx_c == basis_idx)
            C[:, :, basis_idx] = product_terms[:, :, mask].sum(dim=-1)
            
        return C

    def forward(self, state_wave: torch.Tensor, rotor_wave: torch.Tensor) -> torch.Tensor:
        """
        Executes a directional rotor transformation: State' = R * State * R_reverse
         rotor_wave must contain unit-modulus spinors (rotors)
        """
        # 1. Compute R * State
        half_transformed = self.geometric_product(rotor_wave, state_wave)
        
        # 2. Compute rotor reversion R_reverse: reverse the sign of bivectors (indices 4, 5, 6)
        rotor_reversion = rotor_wave.clone()
        rotor_reversion[:, :, [4, 5, 6]] *= -1.0
        
        # 3. Compute (R * State) * R_reverse
        transformed_state = self.geometric_product(half_transformed, rotor_reversion)
        return transformed_state
