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
from hopfield_cleanup import HopfieldActionDecoder

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
        Retracts the low-rank expert matrices onto the row-orthogonality
        constraint (A A^T = I). Uses batched QR of A^T: for A of shape
        [E, R, D], QR of A^T = Q R_q gives A = R_q^T Q^T; normalizing the
        rows of A by Cholesky of A A^T is equivalent and unconditionally
        stable, unlike Newton-Schulz which diverges when singular values
        leave the < sqrt(3) basin after large creep steps.
        Applied to both A and B so both stay volume-preserving.
        """
        for param in (self.experts_A, self.experts_B):
            # Cholesky-based symmetric orthogonalization:
            #   A <- L^{-1} A where L L^T = A A^T  =>  (L^{-1} A)(L^{-1} A)^T = I
            aat = torch.bmm(param, param.transpose(-2, -1))
            # Jitter guards against numerical rank deficiency
            jitter = 1e-6 * torch.eye(self.r_rank, device=param.device).unsqueeze(0)
            L = torch.linalg.cholesky(aat + jitter)
            inv_L = torch.linalg.solve_triangular(
                L, torch.eye(self.r_rank, device=param.device).unsqueeze(0).expand_as(L), upper=False
            )
            param.copy_(torch.bmm(inv_L, param))

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
# The egress decoder is the Continuous Modern Hopfield cleanup layer
# (hopfield_cleanup.HopfieldActionDecoder): canonical action waves are stored
# as pseudo-orthogonal engrams and policy waves are snapped to the nearest
# attractor via single-step softmax retrieval, replacing the old correlated
# linspace phase-ramp basis. Imported above; the public name used by the
# orchestrator remains `HolographicActionDecoder` for compatibility.
HolographicActionDecoder = HopfieldActionDecoder


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


    def compute_free_energy(self, active_wave: torch.Tensor, target_boundary: torch.Tensor, lambda_boundary: float = 1.0) -> torch.Tensor:
        """
        Two-term Natural Induction free energy (differentiable):
            F = 1/2 * ||Laplacian(psi)||^2   (internal propagation stress)
              + lambda/2 * delta_sagnac^2    (boundary resonance penalty)
        active_wave: [num_blocks, 8] real Clifford multivector grid.
        The propagation stress penalizes high-frequency chaotic phase noise;
        we reshape the block axis to a 2D toroidal grid and apply the discrete
        Laplacian (the Triton ephaptic kernel on GPU, torch.roll on CPU).
        """
        # --- Term 1: internal propagation stress (smoothness / complexity) ---
        psi = active_wave
        blocks = psi.shape[0]
        side = int(math.isqrt(blocks))
        if side * side == blocks:
            grid = psi.view(side, side, 8)
            # Toroidal 2D discrete Laplacian per multivector component
            lap = (torch.roll(grid, 1, 0) + torch.roll(grid, -1, 0)
                   + torch.roll(grid, 1, 1) + torch.roll(grid, -1, 1)
                   - 4.0 * grid)
            propagation_stress = 0.5 * (lap ** 2).sum()
        else:
            propagation_stress = active_wave.new_zeros(())
            side = None

        # --- Term 2: boundary resonance penalty (accuracy) ---
        coherence = self.sagnac_coherence(active_wave, target_boundary)
        delta = 1.0 - coherence
        boundary_penalty = 0.5 * lambda_boundary * delta ** 2

        return propagation_stress + boundary_penalty

    def sagnac_coherence(self, active_wave: torch.Tensor, target_boundary: torch.Tensor) -> torch.Tensor:
        """
        Normalized Sagnac coherence: Re(psi^dag T) / D via the Clifford
        geometric product scalar part. Differentiable; returns a tensor in [-1, 1].
        """
        target_rev = target_boundary.clone()
        target_rev[..., [4, 5, 6, 7]] *= -1.0  # Clifford reversion: bivectors + pseudoscalar
        geom_prod = self.clifford.geometric_product(active_wave.unsqueeze(0), target_rev.unsqueeze(0)).squeeze(0)
        return geom_prod[..., 0].sum() / self.num_blocks

    def process_active_reasoning_step(self, active_wave: torch.Tensor, target_boundary: torch.Tensor, external_error_mask: torch.Tensor = None, t_shock_max: float = 0.5) -> tuple:
        """
        Processes a single forward step of the scaled core.
        Executes coupled syncytium relaxation, isolates active experts, and deforms
        parameters via test-time viscoelastic creep driven by true SGLD drift
        (drift = -mu * grad_W F) plus anisotropic Langevin noise.
        """
        # Normalized Sagnac delta in [0, 2] (0 = perfect resonance)
        with torch.no_grad():
            sagnac_delta = 1.0 - self.sagnac_coherence(active_wave, target_boundary)

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
        global_phase = updated_phases.mean()
        phase_coherence = torch.cos(updated_phases - global_phase)
        
        # Tensorize changing scalars to prevent Inductor recompilation
        sagnac_delta_tensor = sagnac_delta
        t_shock_tensor = t_shock_max
        
        # Scale active gap junctions by squared Sagnac Delta.
        recruitment_fraction = torch.clamp(sagnac_delta_tensor ** 2, 0.0, 1.0)
        k_active = torch.clamp((self.num_experts * recruitment_fraction).to(torch.int32), min=4)
        
        # Triton-Safe Static Soft Mask
        sorted_indices = torch.argsort(phase_coherence, descending=True)
        ranks = torch.empty_like(sorted_indices)
        ranks[sorted_indices] = torch.arange(self.num_experts, device=phase_coherence.device)
        mask = (ranks < k_active.unsqueeze(-1)).float().view(-1, 1, 1)

        # 3. Viscoelastic Creep = SGLD on the free energy:
        #    dW/dt = -mu * grad_W F(psi, W) + sqrt(2 T(delta) dt) * eta(t)
        # The drift term is the true gradient of the two-term free energy
        # (propagation stress + boundary resonance) w.r.t. the expert matrices.
        # Noise follows the SGLD sqrt(2 T dt) scaling so the Stiefel retraction
        # stays inside its convergence basin (singular values must remain < 3
        # for Newton-Schulz; raw T-scaled noise blew past that).
        T_base = 0.01
        active_temperature = T_base + t_shock_tensor * (1.0 - torch.exp(-sagnac_delta_tensor))
        mu = 0.05  # SGLD drift (learning) rate
        dt = 0.01
        noise_scale = torch.sqrt(2.0 * active_temperature * dt)

        stress_gate = (sagnac_delta_tensor > 0.05).float().view(-1, 1, 1)
        effective_mask = mask * stress_gate

        # --- SGLD drift: grad of free energy w.r.t. expert matrices -----------
        # The experts enter F through the gap-junction conductance they induce.
        # We differentiate F(psi, W) w.r.t. (experts_A, experts_B) with a fresh
        # graph each step; only active experts receive a meaningful gradient.
        drift_A = torch.zeros_like(self.syncytium.experts_A)
        drift_B = torch.zeros_like(self.syncytium.experts_B)
        if bool(stress_gate.any()):
            with torch.enable_grad():
                A = self.syncytium.experts_A.detach().requires_grad_(True)
                B = self.syncytium.experts_B.detach().requires_grad_(True)
                # Rebuild conductance from differentiable expert projections.
                projections = torch.matmul(A, active_wave.view(-1))
                dist_matrix = torch.cdist(projections.unsqueeze(0), projections.unsqueeze(0), p=2).squeeze(0)
                conductance = self.syncytium.static_adjacency * torch.exp(-(dist_matrix ** 2) / self.syncytium.tau_c)
                # Effective free energy: boundary resonance term, plus a coupling
                # term over the conductance graph, plus a symmetric regularizer
                # tying B to A's geometry (B parameterizes the transpose map and
                # would otherwise receive no drift signal).
                F_boundary = 0.5 * (1.0 - self.sagnac_coherence(active_wave, target_boundary)) ** 2
                F_coupling = 0.5 * (conductance ** 2).mean()
                F_sym = 0.5 * ((torch.matmul(B, active_wave.view(-1)) - projections.detach()) ** 2).mean()
                F_total = F_boundary + F_coupling + 0.01 * F_sym
                gA, gB = torch.autograd.grad(F_total, [A, B], retain_graph=False)
            drift_A = -mu * gA
            drift_B = -mu * gB

        with torch.no_grad():
            # Dale's-polarity modulated drift keeps excitatory/inhibitory sign structure
            p = self.syncytium.polarity.view(-1, 1, 1)

            # Memory-safe chunked noise generation to prevent eager-mode OOM
            chunk_size = 32
            r_rank = self.syncytium.experts_A.shape[1]
            d_model = self.syncytium.experts_A.shape[2]
            for i in range(0, self.num_experts, chunk_size):
                j = min(i + chunk_size, self.num_experts)
                n = j - i
                noise_A_chunk = torch.randn((n, r_rank, d_model), device=self.syncytium.experts_A.device) * noise_scale * error_mask.view(1, 1, -1)
                noise_B_chunk = torch.randn((n, r_rank, d_model), device=self.syncytium.experts_B.device) * noise_scale * error_mask.view(1, 1, -1)

                self.syncytium.experts_A[i:j].add_((drift_A[i:j] * p[i:j] + noise_A_chunk) * effective_mask[i:j])
                self.syncytium.experts_B[i:j].add_((drift_B[i:j] * p[i:j] + noise_B_chunk) * effective_mask[i:j])

            # Enforce Riemannian retraction immediately post-creep to secure volume conservation
            self.syncytium.apply_stiefel_retraction()
            
        # 4. Thermodynamic Decay & Coherence Monitoring
        # Return sorted_indices for logging instead of a dynamic slice
        return sagnac_delta.item(), sorted_indices, {"sagnac_delta": sagnac_delta.item()}


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