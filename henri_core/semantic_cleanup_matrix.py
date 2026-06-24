"""  
Project HENRI: Batch-Vectorized Modern Hopfield Semantic Cleanup Matrix  
Component: Dense Associative Memory, Multi-Expert Epiplexity Sieve  
Author: Joseph Valentine (Bespoke Silicon Photonic Architecture Core)  
Date: 2026-06-23  
"""

import math  
import torch  
import torch.nn as nn  
import torch.nn.functional as F

class BatchedHopfieldSemanticCleanup(nn.Module):  
    """  
    Hardware-accelerated Dense Associative Memory operating on complex wave tensors.  
    Processes all 16 parallel expert streams simultaneously to filter out cross-talk noise.  
    """  
    def __init__(self, dim: int = 4096, beta: float = 20.0, max_iterations: int = 5, tolerance: float = 1e-5):  
        super().__init__()  
        self.dim = dim  
        self.beta = beta          # Inverse temperature scaling factor  
        self.max_iterations = max_iterations  
        self.tolerance = tolerance  
          
        # Internalized dictionary storage arrays  
        self.labels = []  
        self.register_buffer("attractor_matrix", torch.empty(0, dim, dtype=torch.complex64))  
        self.register_buffer("repeller_matrix", torch.empty(0, dim, dtype=torch.complex64))  
          
    def register_concepts_batch(self, labels: list, vectors: torch.Tensor):  
        """Loads an entire matrix of pristine semantic vectors into memory."""  
        if vectors.shape[-1] != self.dim:  
            raise ValueError(f"Vector dimension error. Expected {self.dim}, got {vectors.shape[-1]}")  
          
        # Project vectors onto the unit circle to guarantee modulus conservation  
        mags = torch.abs(vectors)  
        mags = torch.clamp(mags, min=1e-8)  
        unit_vectors = vectors / mags  
          
        self.labels.extend(labels)  
        if self.attractor_matrix.numel() == 0:  
            self.attractor_matrix = unit_vectors.to(dtype=torch.complex64)  
        else:  
            self.attractor_matrix = torch.cat([self.attractor_matrix, unit_vectors.to(dtype=torch.complex64)], dim=0)

    def register_repellers_batch(self, vectors: torch.Tensor):  
        """Appends a batch of repeller coordinates to build negative optimization fields."""  
        if vectors.shape[-1] != self.dim:  
            raise ValueError(f"Repeller dimension error. Expected {self.dim}, got {vectors.shape[-1]}")  
          
        mags = torch.abs(vectors)  
        mags = torch.clamp(mags, min=1e-8)  
        unit_repellers = vectors / mags  
          
        if self.repeller_matrix.numel() == 0:  
            self.repeller_matrix = unit_repellers.to(dtype=torch.complex64)  
        else:  
            self.repeller_matrix = torch.cat([self.repeller_matrix, unit_repellers.to(dtype=torch.complex64)], dim=0)  
              
        # Hard cap repeller allocations to 128 states to prevent memory fragmentation  
        if self.repeller_matrix.size(0) > 128:  
            self.repeller_matrix = self.repeller_matrix[-128:]

    def clear_repeller_fields(self):  
        """Flushes active repeller registers."""  
        self.repeller_matrix = torch.empty(0, self.dim, dtype=torch.complex64, device=self.attractor_matrix.device)

    def forward(self, noisy_waves: torch.Tensor, gamma: float = 0.5, sagnac_delta: float = None) -> dict:  
        """  
        Ingests a batch of noisy wave states [Streams, Dim] and executes competitive  
        energy minimization. Returns a dictionary containing the clean epiplexity field.  
        """  
        if self.attractor_matrix.numel() == 0:  
            raise ValueError("[HOPFIELD CORE] Vocabulary matrix is unprimed. Ingestion failed.")  
          
        # Enforce structural matrix shape invariants  
        if noisy_waves.ndim == 1:  
            noisy_waves = noisy_waves.unsqueeze(0)  
              
        device = noisy_waves.device  
        if self.attractor_matrix.device != device:
            self.attractor_matrix = self.attractor_matrix.to(device)  
        if self.repeller_matrix.device != device:
            self.repeller_matrix = self.repeller_matrix.to(device)  
          
        # 1. Calculate Sagnac-Proportional Temperature (SPT) loop
        beta_t = self.beta
        if sagnac_delta is not None:
            alpha_spt = 5.0
            beta_t = self.beta * (1.0 - math.exp(-alpha_spt * max(0.0, sagnac_delta)))
            beta_t = max(1.0, beta_t)
            
        # Force incoming states onto the complex unit circle boundary  
        s_mags = torch.abs(noisy_waves)  
        s_mags = torch.clamp(s_mags, min=1e-8)  
        s = noisy_waves / s_mags  
          
        last_s = s.clone()  
        num_streams = s.size(0)  
          
        for iteration in range(self.max_iterations):  
            # 1. Parallel Attractor Core Projections  
            # Compute inner products across all active streams: [Streams, Dim] x [Dim, Attractors]  
            dot_products_a = torch.matmul(s, torch.conj(self.attractor_matrix).T)  
            similarities_a = torch.real(dot_products_a) / float(self.dim)  
              
            # Row-wise max extraction to protect exponential scaling limits  
            max_sim, _ = torch.max(similarities_a, dim=-1, keepdim=True)  
            weights_a = torch.exp(beta_t * (similarities_a - max_sim))  
            weights_a = weights_a / (torch.sum(weights_a, dim=-1, keepdim=True) + 1e-8)  
              
            # Blend attractor matrices based on computed similarities: [Streams, Attractors] x [Attractors, Dim]  
            x = torch.matmul(weights_a.to(dtype=self.attractor_matrix.dtype), self.attractor_matrix)  
              
            # 2. Competitive Repeller Push Mechanics  
            if self.repeller_matrix.numel() > 0:  
                dot_products_r = torch.matmul(s, torch.conj(self.repeller_matrix).T)  
                similarities_r = torch.real(dot_products_r) / float(self.dim)  
                  
                weights_r = torch.exp(beta_t * (similarities_r - max_sim))  
                weights_r = weights_r / (torch.sum(weights_r, dim=-1, keepdim=True) + 1e-8)  
                  
                x_repellers = torch.matmul(weights_r.to(dtype=self.repeller_matrix.dtype), self.repeller_matrix)  
                x = x - gamma * x_repellers  
              
            # Re-project tracking matrices onto the unit circle boundary  
            x_mags = torch.abs(x)  
            x_mags = torch.clamp(x_mags, min=1e-8)  
            s = x / x_mags  
            
            # Langevin Heat Quenching (Proportional to the soft derivative of the error energy / Sagnac Delta)
            if sagnac_delta is not None:
                T_base = 0.4
                prev_sagnac = getattr(self, "_prev_sagnac", None)
                if prev_sagnac is not None and sagnac_delta < prev_sagnac:
                    # System is cooling/stabilizing (error decreased): quench noise to baseline floor
                    temp_variance = T_base
                else:
                    # Error flat/increasing: scale thermal noise proportional to stress
                    temp_variance = T_base + 2.0 * max(0.0, sagnac_delta)
                    
                self._prev_sagnac = sagnac_delta
                
                # Apply phase noise on the complex circle
                phase_noise = torch.randn_like(s.real) * (temp_variance * 0.001)
                langevin_kick = torch.complex(torch.cos(phase_noise), torch.sin(phase_noise))
                s = s * langevin_kick
                s = s / (torch.abs(s) + 1e-8)
              
            # Evaluate global convergence delta across the batch  
            convergence_delta = torch.sum(torch.abs(s - last_s)).item()  
            if convergence_delta < self.tolerance:  
                break  
            last_s = s.clone()  
              
        # 3. Final Invariant Metric Calculations  
        final_dot_products = torch.matmul(s, torch.conj(self.attractor_matrix).T)  
        final_similarities = torch.real(final_dot_products) / float(self.dim)  
          
        best_match_indices = torch.argmax(final_similarities, dim=-1)  
        confidence_scores = torch.gather(final_similarities, dim=-1, index=best_match_indices.unsqueeze(-1)).squeeze(-1)  
          
        resolved_labels = [self.labels[idx.item()] if idx.item() < len(self.labels) else "unknown_attractor"   
                           for idx in best_match_indices]  
          
        return {  
            "epiplexity_wave": s,                          # Denoised structural shadow vector field [Streams, Dim]  
            "resolved_labels": resolved_labels,            # Explicit symbolic categorical tags  
            "confidence_metrics": confidence_scores        # Phase alignment resonance scores per expert stream [Streams]  
        }

def run_phase_2_validation():  
    print("=== INITIALIZING HENRI PHASE 2: SEMANTIC CLEANUP VALIDATION ===")  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print(f"[BOOT] Target accelerator environment initialized: {device}")  
      
    # 1. Instantiate the Vectorized Core  
    cleanup_matrix = BatchedHopfieldSemanticCleanup(dim=4096, beta=20.0).to(device)  
      
    # Generate orthogonal reference attractors  
    torch.manual_seed(777)  
    ref_labels = [f"programming_primitive_0x{i:02x}" for i in range(10)]  
    mock_attractors_real = torch.randn(10, 4096, device=device)  
    mock_attractors_imag = torch.randn(10, 4096, device=device)  
    mock_attractors = torch.complex(mock_attractors_real, mock_attractors_imag)  
      
    cleanup_matrix.register_concepts_batch(ref_labels, mock_attractors)  
    print(f"[SUCCESS] Loaded {len(ref_labels)} complex-valued attractors into the DAM matrix.")  
      
    # 2. Forge 16 Noisy Multi-Stream Inputs targeting Attractor Index 4  
    target_concept = cleanup_matrix.attractor_matrix[4].unsqueeze(0).repeat(16, 1)  
      
    # Inject severe phase noise to simulate thermodynamic static  
    phase_noise_angle = torch.randn(16, 4096, device=device) * 1.5  
    noise_vector = torch.complex(torch.cos(phase_noise_angle), torch.sin(phase_noise_angle))  
    noisy_inputs = target_concept + noise_vector * 0.8  
      
    # Measure initial uncleaned cosine similarities  
    normalized_noisy = noisy_inputs / torch.abs(noisy_inputs)
    initial_dot = torch.sum(normalized_noisy * torch.conj(cleanup_matrix.attractor_matrix[4]), dim=-1)  
    initial_res = torch.real(initial_dot).mean().item() / 4096.0  
    print(f"[DATA INFRASTRUCTURE] Average uncleaned input phase alignment: {initial_res:.4f}")  
      
    # 3. Add Competitive Repeller States to simulate active boundary shielding  
    mock_repellers_real = torch.randn(5, 4096, device=device)  
    mock_repellers_imag = torch.randn(5, 4096, device=device)  
    mock_repellers = torch.complex(mock_repellers_real, mock_repellers_imag)  
    cleanup_matrix.register_repellers_batch(mock_repellers)  
      
    # 4. Run Vectorized Egress Cleanup Pass  
    output_payload = cleanup_matrix(noisy_inputs, gamma=0.3)  
      
    clean_waves = output_payload["epiplexity_wave"]  
    resolved_labels = output_payload["resolved_labels"]  
    confidence_metrics = output_payload["confidence_metrics"]  
      
    # 5. Assert Metric Invariants  
    print(f"[MANIFOLD] Processed batch matrix shape: {clean_waves.shape}")  
    assert clean_waves.shape == torch.Size([16, 4096]), "Fatal: Batch operation altered sequence dimension constraints!"  
      
    # Verify strict energy conservation on the complex unit circle boundary  
    modulus_footprint = torch.abs(clean_waves)  
    modulus_deviation = torch.max(torch.abs(modulus_footprint - 1.0)).item()  
    print(f"[SUCCESS] Hyperspherical modulus conservation verified (Max Deviation: {modulus_deviation:.2e}).")  
    assert modulus_deviation < 1e-5, "Fatal: State evolution violated complex circle boundary invariants!"  
      
    # Verify convergence and label classification stability  
    print(f"[CLEANUP SUCCESS] Resolved Labels across 16 streams: {set(resolved_labels)}")  
    print(f"[CLEANUP SUCCESS] Mean Cleaned Phase Alignment Resonance: {confidence_metrics.mean().item():.4f}")  
    assert resolved_labels[0] == "programming_primitive_0x04", "Fatal: Cleanup loop failed to recover target attractor!"  
    assert confidence_metrics.mean().item() > initial_res, "Fatal: Dense Associative Memory expanded wave entropy!"  
      
    print("=== PHASE 2 COGNITIVE SIEVE MATRIX INVARIANTS SECURED ===")

if __name__ == "__main__":  
    run_phase_2_validation()
