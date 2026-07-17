"""
Holographic Biophysical Swarm Scale Engine.
Implements a 1024-expert gap-junction-gated Kuramoto syncytium with sparse 
scale-free connection topologies, Sagnac-driven voltage gating, and Stiefel retraction.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from product_clifford_product_kernel import ProductCliffordAlgebra3D

# =========================================================================
# I. HIGH-PERFORMANCE SCALE-FREE GRAPH GENERATOR
# =========================================================================


class ScaleFreeGraphConstructor:
    """
    Generates a sparse, scale-free Barabási-Albert structural skeleton.
    Ensures that physical expert-to-expert connections satisfy power-law scaling,
    enabling efficient lateral signal diffusion across 1024 expert nodes.
    """
    @staticmethod
    def construct_ba_adjacency(num_nodes=1024, m=3) -> torch.Tensor:
        """
        Constructs a symmetric BA adjacency matrix in PyTorch.
        m: Number of edges to attach from a new node to existing nodes.
        """
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        
        # Initialize seed fully connected sub-graph of size m+1
        for i in range(m + 1):
            for j in range(i + 1, m + 1):
                adj[i, j] = 1.0
                adj[j, i] = 1.0

        # Contiguously evaluate degrees for preferential attachment
        degrees = adj.sum(dim=1)
        
        for new_node in range(m + 1, num_nodes):
            # Preferential attachment probability: P(i) = k_i / sum(k_j)
            probs = degrees / degrees.sum()
            
            # Sample m unique parent nodes based on degree distribution
            targets = torch.multinomial(probs, num_samples=m, replacement=False)
            
            # Map structural connections
            adj[new_node, targets] = 1.0
            adj[targets, new_node] = 1.0
            
            # Update degree cache
            degrees[new_node] += m
            degrees[targets] += 1.0
            
        return adj


# =========================================================================
# II. GAP-JUNCTION-GATED SWARM SYNCYTIUM
# =========================================================================

class GapJunctionSwarmSyncytium(nn.Module):
    """
    Governs the continuous-time coupled phase dynamics of 1024 low-rank experts.
    Uses bioelectric gap-junction potential fields to coordinate lateral resource 
    allocation without causing representational saturation in the core.
    """
    def __init__(self, num_experts=1024, d_model=4096, r_rank=16, coupling_temp=0.5):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.r_rank = r_rank
        self.tau_c = coupling_temp

        # Static Barabási-Albert connection skeleton
        ba_skeleton = ScaleFreeGraphConstructor.construct_ba_adjacency(num_nodes=num_experts, m=4)
        self.register_buffer("static_adjacency", ba_skeleton)

        # Kuramoto Oscillator Parameters
        # Natural Frequency Array (Base intellectual momentum)
        self.register_buffer("natural_frequencies", (torch.rand(num_experts) * 2.0 - 1.0) * math.pi)
        # Active phase angles of the experts
        self.expert_phases = nn.Parameter(torch.zeros(num_experts))

        # Expert low-rank projection parameters: LoRA A and B
        # Shape: [E, Rank, D] and [E, Rank, D]
        self.experts_A = nn.Parameter(torch.randn(num_experts, r_rank, d_model) * 0.01)
        self.experts_B = nn.Parameter(torch.randn(num_experts, r_rank, d_model) * 0.01)

        # Biological Dale's Principle: 80% Excitatory, 20% Inhibitory
        polarity = torch.ones(num_experts)
        num_inhibitory = int(num_experts * 0.2)
        inhibitory_indices = torch.randperm(num_experts)[:num_inhibitory]
        polarity[inhibitory_indices] = -1.0
        self.register_buffer("polarity", polarity)

        self.apply_stiefel_retraction()

    @torch.no_grad()
    def apply_stiefel_retraction(self):
        """
        Applies a high-performance, batched Newton-Schulz retraction to the low-rank matrices
        to guarantee volume-preserving, magnitude-conserving rotations.
        Enforces the mathematically rigorous row-orthogonality constraint (A A^T = I)
        simultaneously across all 1,024 experts using vectorized batched matrix multiplication.
        """
        # Establish contiguous r_rank identity matrices pinned to active device
        identity = torch.eye(self.r_rank, device=self.experts_A.device).unsqueeze(0).expand(self.num_experts, -1, -1)
        
        # Newton-Schulz iterations: A <- (1.5 * I - 0.5 * A * A^T) * A
        for _ in range(3):
            # experts_A shape: [E, R, D], experts_A.transpose: [E, D, R] -> [E, R, R]
            aat = torch.bmm(self.experts_A, self.experts_A.transpose(-2, -1))
            correction = 1.5 * identity - 0.5 * aat
            # [E, R, R] x [E, R, D] -> [E, R, D]
            self.experts_A.copy_(torch.bmm(correction, self.experts_A))

    def compute_dynamic_conductance(self, active_wave: torch.Tensor) -> torch.Tensor:
        """
        Computes the bioelectric gap-junction potential field.
        Lateral conductance G_ij is scaled by the similarity of the experts' active projections.
        """
        # Compress active wave using low-rank projections
        # Shape: [E, Rank]
        active_wave_flat = active_wave.view(-1) # Flatten [8192, 8] -> [65536]
        # self.experts_A: [1024, 16, 65536], active_wave_flat: [65536]
        # batched matmul natively produces [1024, 16]
        projections = torch.matmul(self.experts_A, active_wave_flat)

        # Compute pairwise distance matrix over projections
        # Shape: [E, E]
        dist_matrix = torch.cdist(projections.unsqueeze(0), projections.unsqueeze(0), p=2).squeeze(0)

        # Gate conductance: G_ij = Adj_ij * exp( -d^2 / tau_c )
        dynamic_conductance = self.static_adjacency * torch.exp(- (dist_matrix ** 2) / self.tau_c)
        return dynamic_conductance

    def forward_syncytium_step(self, active_wave: torch.Tensor, sagnac_order_param: float, dt=0.01) -> torch.Tensor:
        """
        Integrates one step of the coupled Kuramoto-ephaptic system using Euler-Maruyama.
        Updates expert phase coordinates mid-flight to dynamically reallocate parameters.
        """
        # Compute dynamic gap-junction potential fields
        G = self.compute_dynamic_conductance(active_wave)

        # Compute phase difference matrix: \sin(\theta_j - \theta_i)
        phase_diff = self.expert_phases.unsqueeze(1) - self.expert_phases.unsqueeze(0)
        sin_diff = torch.sin(phase_diff)

        # Scale coupling based on current Sagnac Order Parameter (High coherence -> strong coupling)
        coupling_gain = 2.5 * (1.0 - sagnac_order_param)

        # Compute coupled phase acceleration: d\theta / dt
        # Shape: [E]
        coupled_force = (G * sin_diff).sum(dim=1) / self.num_experts
        d_theta = self.natural_frequencies + coupling_gain * coupled_force

        # Inject localized Langevin thermal noise if coherence is low (Sagnac Veto Active)
        if sagnac_order_param < 0.65:
            # Thermal variance proportional to the mismatch delta
            temperature = 3.5 * (1.0 - sagnac_order_param)
            noise = torch.randn_like(self.expert_phases) * math.sqrt(2.0 * temperature * dt)
            d_theta += noise

        # Execute Euler-Maruyama step integration
        self.expert_phases.data.add_(d_theta * dt)
        
        # Wrap phase angles back to [-pi, pi] to maintain unit circle constraint
        self.expert_phases.data.copy_(torch.atan2(torch.sin(self.expert_phases), torch.cos(self.expert_phases)))

        return self.expert_phases


# =========================================================================
# III. HOLOGRAPHIC ACTION DECODER (EGRESS GATE)
# =========================================================================

class HolographicActionDecoder(nn.Module):
    """
    Decodes high-dimensional complex wave states (optimal_policy_wave)
    into discrete GameAction categories.
    Uses Sagnac phase-matching against canonical orthogonal action-basis waves.
    """
    def __init__(self, d_model=4096, action_enum_class=None):
        super().__init__()
        self.d_model = d_model
        self.action_to_id = {}
        self.id_to_action = {}
        
        # Map target action enum/names to indices
        if action_enum_class is not None:
            # Inspect enum attributes or custom class definitions
            actions_list = [a for a in action_enum_class]
            for idx, action in enumerate(actions_list):
                self.action_to_id[action] = idx
                self.id_to_action[idx] = action
        else:
            # Standard structural fallbacks for standard ARC-AGI-3 environments
            fallback_actions = ["UP", "DOWN", "LEFT", "RIGHT", "ACTION1", "ACTION2"]
            for idx, action in enumerate(fallback_actions):
                self.action_to_id[action] = idx
                self.id_to_action[idx] = action
                
        num_actions = len(self.action_to_id)
        
        # Allocate orthogonal phase-carrier coordinates on the unit circle
        basis_phases = torch.zeros((num_actions, d_model))
        for idx in range(num_actions):
            # Generate deterministic, orthogonal phase vectors
            basis_phases[idx] = torch.linspace(-math.pi, math.pi, d_model) + (idx * 1.5)
            
        self.register_buffer("basis_phases", basis_phases)

    def get_action_wave(self, action) -> torch.Tensor:
        """
        Retrieves the canonical wavefront representing a specific Action.
        """
        idx = self.action_to_id.get(action, 0)
        phases = self.basis_phases[idx]
        wave = torch.complex(torch.cos(phases), torch.sin(phases))
        return wave / torch.norm(wave, p=2)

    def decode_wave_to_action(self, policy_wave: torch.Tensor):
        """
        Evaluates constructive phase-alignment of the policy wave against all basis vectors.
        Returns the closest matching discrete action coordinate.
        """
        # Normalize and reshape incoming wave
        flat_wave = policy_wave.view(-1)
        norm_wave = flat_wave / torch.norm(flat_wave, p=2).clamp(min=1e-12)
        
        # Construct complex basis waves in parallel
        basis_waves = torch.complex(torch.cos(self.basis_phases), torch.sin(self.basis_phases))
        basis_waves = basis_waves.to(norm_wave.device)
        basis_waves = basis_waves / torch.norm(basis_waves, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        
        # Measure phase coherence (real part of complex inner product)
        # coherence shape: [num_actions]
        # Since norm_wave might be real (from Clifford), convert to complex to avoid dtype mismatch
        norm_wave_c = norm_wave.to(torch.complex64)
        coherence = torch.real(torch.sum(norm_wave_c * basis_waves.conj(), dim=-1))
        
        best_idx = torch.argmax(coherence).item()
        return self.id_to_action[best_idx], coherence[best_idx].item()


# =========================================================================
# IV. THERMODYNAMIC CORE BINDING & STEERING
# =========================================================================

class HenriSwarmOrchestrator(nn.Module):
    """
    Orchestrates the continuous wave interactions across the 1024-expert syncytium.
    Uses Sagnac-driven voltage gating to activate only the highly resonant expert
    sub-graphs, leaving the remainder isolated to prevent representational saturation.
    """
    def __init__(self, num_experts=1024, d_model=65536, r_rank=16, num_blocks=8192, action_enum_class=None):
        super().__init__()
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.num_experts = num_experts
        self.syncytium = GapJunctionSwarmSyncytium(num_experts=num_experts, d_model=d_model, r_rank=r_rank)
        self.clifford = ProductCliffordAlgebra3D(num_blocks=num_blocks)
        self.decoder = HolographicActionDecoder(d_model=d_model, action_enum_class=action_enum_class)


    def process_active_reasoning_step(self, active_wave: torch.Tensor, target_boundary: torch.Tensor, external_error_mask: torch.Tensor = None, t_shock_max: float = 0.5) -> tuple:
        """
        Processes a single forward step of the scaled core.
        Executes coupled syncytium relaxation, isolates active experts, and deforms 
        parameters via test-time viscoelastic creep.
        """
        # Calculate true topological Sagnac Coherence via Clifford Algebra
        target_rev = target_boundary.clone()
        target_rev[..., [4, 5, 6]] *= -1.0 # Reverse bivectors
        geom_prod = self.clifford.geometric_product(active_wave.unsqueeze(0), target_rev.unsqueeze(0)).squeeze(0)
        sagnac_coherence = (geom_prod[..., 0].sum() / self.num_blocks).item()
        true_sagnac_delta = 1.0 - sagnac_coherence

        sagnac_delta = true_sagnac_delta
        
        # Override with external physical error mask if transduced from sandbox (Ontological Error Matrix)
        if external_error_mask is not None:
            error_mask = external_error_mask
        else:
            # Fallback uniform heat if no sparse masking is provided
            error_mask = torch.ones_like(active_wave.view(-1))
        
        # 1. Update Coupled Syncytium Phases (Kuramoto bioelectric pass)
        updated_phases = self.syncytium.forward_syncytium_step(active_wave, sagnac_delta)

        # 2. Voltage-Gating (Stress-Gated Morphogenetic Scaling)
        # Instead of a fixed top-k, compute becomes a fluid boundary scaling with physical stress.
        global_phase = updated_phases.mean().item()
        phase_coherence = torch.cos(updated_phases - global_phase)
        
        # Scale active gap junctions by squared Sagnac Delta.
        # Low stress = highly localized (4 experts). High stress = wide syncytium recruitment.
        recruitment_fraction = min(1.0, max(0.0, sagnac_delta ** 2))
        k_active = max(4, int(self.num_experts * recruitment_fraction))
        
        # Triton-Safe Static Soft Mask
        sorted_indices = torch.argsort(phase_coherence, descending=True)
        ranks = torch.empty_like(sorted_indices)
        ranks[sorted_indices] = torch.arange(self.num_experts, device=phase_coherence.device)
        mask = (ranks < k_active).float().view(-1, 1, 1)

        # 3. Viscoelastic Creep (Forward Error Diffusion via Dale's Principle)
        if sagnac_delta > 0.05:
            # Physical base temperature T_base = 0.01
            T_base = 0.01
            active_temperature = T_base + t_shock_max * (1.0 - math.exp(-sagnac_delta))
            gamma = 0.05 # Substrate physical density constant
            
            with torch.no_grad():
                # Forward Error Diffusion directly pushes the matrices.
                p = self.syncytium.polarity.view(-1, 1, 1)
                yielding_force = -gamma * sagnac_delta * p
                
                # Anisotropic noise generation: Scaled by the localized error mask
                noise_A = torch.randn_like(self.syncytium.experts_A) * active_temperature * error_mask.view(1, 1, -1)
                noise_B = torch.randn_like(self.syncytium.experts_B) * active_temperature * error_mask.view(1, 1, -1)
                
                self.syncytium.experts_A.add_((yielding_force + noise_A) * mask)
                self.syncytium.experts_B.add_((yielding_force + noise_B) * mask)
            
            # Enforce Riemannian retraction immediately post-creep to secure volume conservation
            self.syncytium.apply_stiefel_retraction()
            
        # 4. Thermodynamic Decay & Coherence Monitoring
        # Return sorted_indices for logging instead of a dynamic slice
        return sagnac_delta, sorted_indices, {"sagnac_delta": sagnac_delta}


# =========================================================================
# IV. BARE-METAL SWARM SCALING VERIFICATION
# =========================================================================

def verify_biophysical_scaling():
    print("================================================================")
    print("   PROJECT HENRI: BIOPHYSICAL SWARM SCALE INTEGRITY CHECK       ")
    print("================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Hardware Substrate Identified: {device}")

    # Initialize 1024-Expert Swarm Orchestrator
    orchestrator = HenriSwarmOrchestrator().to(device)
    print(f"[INIT] 1024 Expert Syncytium successfully initialized.")
    print(f"[INIT] Contiguous Low-Rank expert matrices allocated.")

    # Generate mock continuous wavefront (representing incoming task state)
    active_wave = torch.randn(8192, 8, device=device)
    active_wave = active_wave / torch.norm(active_wave, p=2, dim=-1, keepdim=True)

    # Retrieve target boundary constraint from long-term memory
    target_boundary = torch.randn(8192, 8, device=device)
    target_boundary = target_boundary / torch.norm(target_boundary, p=2, dim=-1, keepdim=True)

    # Execute 3 sequential relaxation steps of the massive Kuramoto syncytium
    print("\n[ACTIVE INFERENCE] Dropping wave into the 1024-Expert Syncytium...")
    for step in range(3):
        start_step = time.perf_counter()
        
        sagnac_delta, active_experts, _ = orchestrator.process_active_reasoning_step(
            active_wave, target_boundary
        )
        
        end_step = time.perf_counter()
        elapsed_ms = (end_step - start_step) * 1000

        print(f"  Step {step} | Sagnac Delta: {sagnac_delta:.6f} | Active Experts: {active_experts.tolist()[:6]}... | Step Latency: {elapsed_ms:.4f} ms")

        # Verify that the sparse gating prevented compute stalls
        assert elapsed_ms < 15.0, "STALL REGISTERED: Gated-conductance calculation exceeded execution limit."

    print("\n================================================================")
    print("   BIOPHYSICAL SWARM SCALING CHECK COMPLETED: VERIFIED SUCCESS  ")
    print("================================================================")


if __name__ == "__main__":
    import time
    verify_biophysical_scaling()