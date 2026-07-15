import unittest
import torch
import math
import numpy as np

# Ensure float32 globally to avoid casting errors
torch.set_default_dtype(torch.float32)

from o_vsa_ingress_tokenizer import O_VSA_IngressTokenizer
from darwinian_phase_swarm import NewtonSchulzProjector
from arc_agi_zone_c_seed import WirtingerComplexMatmul
from bioactive_thermodynamic_master import BioactiveThermodynamicMaster
from phylogenetic_memory import EngramStore

class TestProductionIntegrity(unittest.TestCase):
    
    def test_ovsa_ingress_orthogonality(self):
        """
        Asserts that the True Local Tokenizer maps to the S^d-1 hypersphere
        and generates highly orthogonal Unitary Wave Embeddings.
        """
        d = 4096
        tokenizer = O_VSA_IngressTokenizer(d=d, vocab_size=256, device='cpu')
        
        # Test 1: L2 Norm constraint (S^d-1 surface)
        waves = tokenizer.encode("ABC")
        self.assertEqual(waves.shape, (3, d), "Incorrect embedding shape.")
        
        norms = torch.norm(waves, p=2, dim=1)
        for norm in norms:
            # Note: Since the basis is generated as polar(ones, angles), 
            # its raw norm is sqrt(d) = sqrt(4096) = 64.0
            self.assertTrue(math.isclose(norm.item(), math.sqrt(d), rel_tol=1e-4), 
                            "UWE vectors did not maintain unitary amplitude.")
                            
        # Test 2: Orthogonality (Expected Dot Product ~ 0)
        dot_product = torch.abs(torch.vdot(waves[0], waves[1]))
        # Expected variance for random phase dot product is sqrt(d)
        self.assertTrue(dot_product.item() < 3.0 * math.sqrt(d), 
                        "Basis vectors are dangerously correlated. Chaos entropy failure.")

    def test_stiefel_drift_boundaries(self):
        """
        Asserts that the continuous Newton-Schulz retraction rigorously locks
        weight matrices to the Stiefel Manifold (W^H W = I).
        """
        dim = 64
        # Generate a drifting (non-orthogonal) complex weight matrix
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        W = torch.randn(dim, dim, dtype=torch.complex64, device=device)
        
        # Retract to the manifold
        W_stiefel = NewtonSchulzProjector.retract(W, iterations=7)
        
        # Verify Stiefel constraint: W^H W = I
        identity_approx = torch.matmul(torch.conj(W_stiefel.T), W_stiefel)
        identity_true = torch.eye(dim, dtype=torch.complex64)
        
        error = torch.norm(identity_approx - identity_true, p='fro').item()
        
        # We expect the error to be extremely small (e.g. < 1e-4) if Newton-Schulz converged
        self.assertTrue(error < 1e-4, 
                        f"Newton-Schulz retraction failed Stiefel constraint! FroError: {error}")

    def test_langevin_cooling_efficiency(self):
        """
        Asserts that the Thermodynamic Master applies extreme heat when Sagnac Delta is high,
        and achieves crystalline lock (T_eff = 0) when Sagnac Delta -> 0.
        """
        dim = 256
        num_experts = 4
        master = BioactiveThermodynamicMaster(num_oscillators=dim, num_experts=num_experts, device='cpu')
        
        K_matrix = torch.zeros(dim, dim) # Dummy K-matrix
        t_step = 1.0
        
        # Scenario A: High Sagnac Error (Mismatch)
        # Target and Pred are completely uncorrelated
        target_angles = torch.rand(dim) * 2 * math.pi
        target_wave = torch.polar(torch.ones(dim), target_angles).to(torch.complex64)
        pred_phases = torch.randn(num_experts, dim) * 2 * math.pi
        
        _, sagnac, _, T_eff_hot = master.execute_coupled_relaxation_step(pred_phases, K_matrix, target_wave, t_step)
        
        self.assertTrue(T_eff_hot > 1.5, 
                        f"Failed to generate sufficient Langevin heat under high entropy. T_eff = {T_eff_hot}")
        
        # Scenario B: Crystalline Lock (Zero Error)
        # Pred phases exactly match target wave angle
        perfect_angles = torch.angle(target_wave)
        pred_phases_perfect = perfect_angles.unsqueeze(0).repeat(num_experts, 1)
        
        _, _, _, T_eff_cold = master.execute_coupled_relaxation_step(pred_phases_perfect, K_matrix, target_wave, t_step)
        
        self.assertTrue(T_eff_cold == 0.0, 
                        f"Thermodynamic Credit Assigner failed to cool. Crystalline lock broken! T_eff = {T_eff_cold}")

    def test_wirtinger_calculus_gradients(self):
        """
        Asserts that the custom PyTorch autograd function correctly propagates 
        complex conjugate gradients across phase masks without breaking.
        """
        psi = torch.randn(16, 64, dtype=torch.complex64, requires_grad=True)
        weight = torch.randn(64, 64, dtype=torch.complex64, requires_grad=True)
        
        # Apply Wirtinger Matmul
        out = WirtingerComplexMatmul.apply(psi, weight)
        
        # Compute dummy loss (Sum of absolute values)
        loss = torch.sum(torch.abs(out))
        loss.backward()
        
        # Check if gradients exist and are non-zero
        self.assertIsNotNone(psi.grad, "Silent gradient break on Input Wave!")
        self.assertIsNotNone(weight.grad, "Silent gradient break on Weight Mask!")
        
        self.assertTrue(torch.sum(torch.abs(psi.grad)) > 0, "Gradient on Input is zero!")
        self.assertTrue(torch.sum(torch.abs(weight.grad)) > 0, "Gradient on Weight is zero!")

    def test_int8_phylogenetic_memory_quantization(self):
        """
        Asserts that the L2=1.0 Unitary Wave structure survives the INT8 extreme quantization
        serialization process over the simulated CXL bus.
        """
        store = EngramStore("dummy_dsn")
        dim = 4096
        
        # Generate standard continuous random wave
        original_wave = torch.randn(dim, dtype=torch.complex64)
        
        # Serialize (Quantizes to INT8)
        serialized_str = store._serialize_wave(original_wave)
        
        # Deserialize (De-quantizes and restores to complex64 L2=1.0)
        restored_wave = store._deserialize_wave(serialized_str)
        
        self.assertEqual(restored_wave.shape, (dim,), "Dimensionality collapsed during quantization.")
        self.assertEqual(restored_wave.dtype, torch.complex64, "Wave collapsed into real-only space.")
        
        restored_norm = torch.norm(restored_wave, p=2).item()
        self.assertTrue(abs(restored_norm - 1.0) < 1e-3,
                        f"L2 Metric invariant violated! Expected 1.0, got {restored_norm}")
                        
if __name__ == '__main__':
    unittest.main()
