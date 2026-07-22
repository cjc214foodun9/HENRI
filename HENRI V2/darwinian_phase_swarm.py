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
from efe_planner import EFEPlanner
from idbd_swifttd import AdaptiveCreepController
from zone_c_segment_cache import SegmentCache

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

        # Per-expert adaptive step-size controllers (IDBD + SwiftTD) for the
        # SGLD creep: one scalar step-size per expert keeps state tiny while
        # letting stable experts crystallize and volatile ones stay plastic.
        self.creep_ctrl_A = AdaptiveCreepController(
            (num_experts, 1, 1), device=self.experts_A.device
        )
        self.creep_ctrl_B = AdaptiveCreepController(
            (num_experts, 1, 1), device=self.experts_B.device
        )

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
            # Chunked over experts to bound the bmm workspace on large swarms.
            eye = torch.eye(self.r_rank, device=param.device)
            chunk = 256
            for i in range(0, self.num_experts, chunk):
                j = min(i + chunk, self.num_experts)
                p = param[i:j]
                aat = torch.bmm(p, p.transpose(-2, -1))
                jitter = 1e-6 * eye.unsqueeze(0)
                L = torch.linalg.cholesky(aat + jitter)
                inv_L = torch.linalg.solve_triangular(
                    L, eye.unsqueeze(0).expand(j - i, -1, -1), upper=False
                )
                param[i:j].copy_(torch.bmm(inv_L, p))

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
        # Kuramoto threshold for uniform g(omega) on [-pi, pi]: kappa_c = 4.
        # The old 2.5*(1-delta) gain (~1.7 at delta=0.31) sits BELOW kappa_c
        # at all times, structurally guaranteeing desynchronization. Floor at
        # 6.0 keeps the swarm supercritical; coherence still modulates gain.
        coupling_gain = 6.0 + 2.5 * (1.0 - sagnac_order_param)

        # Compute coupled phase acceleration: d\theta / dt
        # Shape: [E]
        # Sign correction: the synchronizing Kuramoto force is
        #   -k * mean_j sin(theta_j - theta_i)   (attractive toward neighbors)
        # The previous (+) sign was REPULSIVE — more coupling drove r DOWN
        # (verified numerically: r 0.016 at gain 6.0 with +, 0.945 with -).
        # Degree-normalized Kuramoto force: mean over each node's ACTUAL
        # coupled neighbors, not the whole population. G is the sparse BA
        # skeleton gated by projection similarity (~8 effective edges/node);
        # dividing by num_experts dilutes the per-node coupling ~128x below
        # the supercritical gain the (all-to-all) simulation assumed.
        deg = G.sum(dim=1).clamp(min=1e-6)
        coupled_force = -(G * sin_diff).sum(dim=1) / deg
        d_theta = self.natural_frequencies + coupling_gain * coupled_force

        # Inject localized Langevin thermal noise if coherence is low (Sagnac Veto Active)
        # NOTE: with the corrected attractive coupling and supercritical gain
        # the swarm now phase-locks; whether this shock polarity helps or
        # hurts the locked state is a separate experiment (see audit notes),
        # so the original condition is preserved unchanged here.
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
    def __init__(self, num_experts=1024, d_model=65536, r_rank=16, num_blocks=8192, action_enum_class=None,
                 constraint_weight_max=5.0, constraint_reject_thresh=0.5, beta_pragmatic=1.0):
        super().__init__()
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.num_experts = num_experts
        self.syncytium = GapJunctionSwarmSyncytium(num_experts=num_experts, d_model=d_model, r_rank=r_rank)
        self.clifford = ProductCliffordAlgebra3D(num_blocks=num_blocks)
        self.decoder = HolographicActionDecoder(d_model=d_model, action_enum_class=action_enum_class)
        # EFE action planner: scores top-k candidate action waves by Expected
        # Free Energy (pragmatic surprise vs epistemic information gain) by
        # propagating each through the unitary transition operator.
        self.planner = EFEPlanner(num_blocks=num_blocks, d_model=d_model,
                                  constraint_weight_max=constraint_weight_max,
                                  constraint_reject_thresh=constraint_reject_thresh,
                                  beta_pragmatic=beta_pragmatic)
        # Seed the planner's retrieval store with the decoder's action waves,
        # flattened to real width d_model to match the planner's store.
        action_real = torch.stack([
            torch.view_as_real(self.decoder.get_action_wave(a)).reshape(-1)[:d_model]
            for a in self.decoder.id_to_action.values()
        ])
        self.planner.cleanup.store_engrams(action_real)
        # Zone C SegmentCache: long-term engram memory. Lazily connected on
        # first use so construction never blocks on a database round-trip.
        self._segment_cache = None
        self._segment_cache_dsn = None

    def attach_zone_c(self, dsn: str = None, top_k: int = 4, max_age_hours: float = 24.0):
        """Connect the Zone C SegmentCache (TimescaleDB live, or surrogate)."""
        self._segment_cache = SegmentCache.connect(
            dsn=dsn, num_blocks=self.num_blocks, top_k=top_k, max_age_hours=max_age_hours
        )
        return self._segment_cache

    @property
    def segment_cache(self):
        if self._segment_cache is None:
            self.attach_zone_c(dsn=self._segment_cache_dsn)
        return self._segment_cache

    def checkpoint_wave(self, wave: torch.Tensor, domain: str, sagnac_stress: float):
        """Persist the current wave as a Zone C engram checkpoint."""
        return self.segment_cache.checkpoint(wave, domain, sagnac_stress)

    def recall_conditioning_wave(self, query_wave: torch.Tensor):
        """
        GRM retrieval from Zone C: returns a gate-weighted conditioning wave
        fused from the most relevant past engrams, or None if memory is empty.
        Used to anchor the syncytium's relaxation with long-term context.
        """
        return self.segment_cache.retrieve(query_wave)["conditioning_wave"]

    def candidate_action_waves(self, top_k: int = 4):
        """
        Returns the top-k (action, action_wave) candidates from the decoder's
        engram store, reshaped to [num_blocks, 8] Clifford waves.
        """
        waves = []
        n = min(top_k, len(self.decoder.id_to_action))
        for idx in range(n):
            action = self.decoder.id_to_action[idx]
            w = self.decoder.get_action_wave(action)  # complex [d_model]
            w_real = torch.view_as_real(w).reshape(-1)[: self.d_model]
            w_grid = w_real.view(self.num_blocks, 8)
            w_grid = w_grid / (torch.norm(w_grid, p=2, dim=-1, keepdim=True) + 1e-9)
            waves.append((action, w_grid))
        return waves

    def plan_action(self, active_wave: torch.Tensor, boundary_axioms: torch.Tensor, top_k: int = 4,
                    return_chosen: bool = False):
        """
        EFE action selection: score the top-k candidate actions by Expected
        Free Energy and return (action, predicted_wave, score_table).
        With return_chosen=True, returns (action, predicted_wave, table, chosen).
        boundary_axioms: [N, num_blocks, 8] real waves.
        """
        candidates = self.candidate_action_waves(top_k=top_k)
        action, predicted, table, chosen = self.planner.select_action(
            active_wave, candidates, boundary_axioms
        )
        if return_chosen:
            return action, predicted, table, chosen
        return action, predicted, table


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

    def process_active_reasoning_step(self, active_wave: torch.Tensor, target_boundary: torch.Tensor, external_error_mask: torch.Tensor = None, t_shock_max: float = 0.5, valence: float = 0.0) -> tuple:
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
        # Wire B (valence thermal channel, swarm side): failure valence
        # (nu < 0) injects anisotropic heat to destabilize a failed-trajectory
        # attractor (the Dark Room shaker); success (nu > 0) cools toward
        # T_base, crystallizing the verified state. Neutral leaves the
        # surprise-driven schedule untouched.
        if valence < 0.0:
            active_temperature = active_temperature + 0.5 * (-valence)
        elif valence > 0.0:
            active_temperature = active_temperature / (1.0 + valence)
        mu = 0.05  # SGLD drift (learning) rate
        dt = 0.01
        noise_scale = torch.sqrt(2.0 * active_temperature * dt)

        stress_gate = (sagnac_delta_tensor > 0.05).float().view(-1, 1, 1)
        effective_mask = mask * stress_gate

        # --- SGLD drift: grad of free energy w.r.t. expert matrices -----------
        # The experts enter F through the gap-junction conductance they induce.
        # Computed with torch.func.grad (functional, no persistent autograd
        # graph) to keep the [1024,1024] cdist + projections from pinning
        # several GiB of activations across the creep loop.
        drift_A = None
        drift_B = None
        if bool(stress_gate.any()):
            adj = self.syncytium.static_adjacency
            tau_c = self.syncytium.tau_c
            wave_flat = active_wave.view(-1).detach()

            def coupling_energy(A, B):
                projections = torch.matmul(A, wave_flat)
                sq = torch.cdist(projections.unsqueeze(0), projections.unsqueeze(0), p=2).squeeze(0) ** 2
                conductance = adj * torch.exp(-sq / tau_c)
                F_coupling = 0.5 * (conductance ** 2).mean()
                F_sym = 0.5 * ((torch.matmul(B, wave_flat) - projections.detach()) ** 2).mean()
                return F_coupling + 0.01 * F_sym

            gA, gB = torch.func.grad(coupling_energy, argnums=(0, 1))(
                self.syncytium.experts_A.detach(), self.syncytium.experts_B.detach()
            )
            # IDBD + SwiftTD: per-expert adaptive step-size applied in place on
            # the gradient tensors (no extra 4 GiB copies). Drift direction is
            # the gradient; magnitude is the per-expert IDBD alpha, SwiftTD-bounded.
            delta_scalar = float(sagnac_delta_tensor)
            raw_A, raw_B = gA, gB  # reuse grad tensors in place
            del gA, gB
            raw_A.mul_(-mu)
            raw_B.mul_(-mu)
            # Normalized per-expert feature: RMS drift magnitude (scale-invariant
            # across d_model) so IDBD meta-learning engages at production dims.
            feat_A = raw_A.norm(dim=(1, 2)).div_(math.sqrt(raw_A[0].numel())).view(-1, 1, 1)
            feat_B = raw_B.norm(dim=(1, 2)).div_(math.sqrt(raw_B[0].numel())).view(-1, 1, 1)
            scale_A = self.syncytium.creep_ctrl_A.scaled_drift(delta_scalar, feat_A) / (feat_A + 1e-12)
            scale_B = self.syncytium.creep_ctrl_B.scaled_drift(delta_scalar, feat_B) / (feat_B + 1e-12)
            raw_A.mul_(scale_A)
            raw_B.mul_(scale_B)
            drift_A, drift_B = raw_A, raw_B
            del feat_A, feat_B, scale_A, scale_B
            if active_wave.is_cuda:
                torch.cuda.empty_cache()

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

                dA = drift_A[i:j] if drift_A is not None else 0.0
                dB = drift_B[i:j] if drift_B is not None else 0.0
                self.syncytium.experts_A[i:j].add_((dA * p[i:j] + noise_A_chunk) * effective_mask[i:j])
                self.syncytium.experts_B[i:j].add_((dB * p[i:j] + noise_B_chunk) * effective_mask[i:j])

            # Enforce Riemannian retraction immediately post-creep to secure volume conservation
            self.syncytium.apply_stiefel_retraction()
            
        # 4. Thermodynamic Decay & Coherence Monitoring
        # Return sorted_indices for logging instead of a dynamic slice
        return sagnac_delta.item(), sorted_indices, {"sagnac_delta": sagnac_delta.item()}
