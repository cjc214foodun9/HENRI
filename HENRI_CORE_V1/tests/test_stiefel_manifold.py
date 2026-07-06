import pytest
import torch
import os
import sys

# Add the parent directory to the path so we can import the core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viscoelastic_swarm_core_shared_baseplate import ProprietaryHENRICore

def test_bjorck_newton_stiefel_projection():
    """
    Mathematically verifies that the Bjorck-Newton algorithm perfectly 
    projects a warped weight matrix back onto the orthogonal Stiefel manifold.
    """
    dim = 256
    num_layers = 2
    num_experts = 4
    
    # Initialize core (which calls reset_parameters to establish random orthogonality)
    core = ProprietaryHENRICore(dim=dim, num_layers=num_layers, num_experts=num_experts)
    
    # Forcefully break orthogonality of the first layer's shared weights 
    # (Simulating a violent AdamW gradient step without clipping)
    with torch.no_grad():
        broken_matrix = core.shared_layers[0].data * 1.1 + torch.randn(dim, dim, dtype=torch.complex64) * 0.01
        core.shared_layers[0].data.copy_(broken_matrix)
        
        # Verify it is broken
        X = core.shared_layers[0].data
        identity_check = torch.matmul(X, X.conj().t())
        I = torch.eye(dim, dtype=torch.complex64)
        
        # It should NOT be close to identity
        assert not torch.allclose(identity_check, I, atol=1e-3), "Matrix should be broken before projection."
        
    # Run the Bjorck-Newton Orthogonalization algorithm (5 iterations)
    core.bjorck_newton_orthonormalize(iterations=5)
    
    # Verify it has mathematically snapped back to the Stiefel manifold
    with torch.no_grad():
        X_restored = core.shared_layers[0].data
        identity_check_restored = torch.matmul(X_restored, X_restored.conj().t())
        
        # It MUST be perfectly unitary/orthogonal now
        assert torch.allclose(identity_check_restored, I, atol=1e-4), "Bjorck-Newton failed to project matrix to Stiefel manifold!"

if __name__ == "__main__":
    pytest.main([__file__])
