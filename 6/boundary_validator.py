import torch
import torch.nn as torch_nn
import numpy as np
import math
from emergent_topological_manifold import EmergentManifold

class BoundaryAxiomValidator(torch_nn.Module):
    """
    Implements the Dirichlet/Neumann holographic boundary validation.
    Projects the 4096-D bulk wave to a 64-D CFT boundary tensor (h_cft),
    validates the three Dirichlet sectors against mathematical axioms, and 
    manages the adaptive Active Neumann Boundary sector.
    """
    def __init__(self, bulk_dim=4096, boundary_dim=64, epsilon_spine=0.35, seed=42):
        super().__init__()
        self.bulk_dim = bulk_dim
        self.boundary_dim = boundary_dim
        self.sector_dim = boundary_dim // 4  # 16 dimensions per sector
        self.epsilon_spine = epsilon_spine    # Lipschitz continuity threshold
        
        # 1. Initialize fixed projection matrix P: shape [64, 4096] (complex64)
        # Using a fixed seed ensures identical mapping across instances
        g = torch.Generator()
        g.manual_seed(seed)
        
        phases = (torch.rand(boundary_dim, bulk_dim, generator=g) * 2 * math.pi) - math.pi
        self.register_buffer('P_real', torch.cos(phases) / math.sqrt(bulk_dim))
        self.register_buffer('P_imag', torch.sin(phases) / math.sqrt(bulk_dim))
        
        # 2. Initialize the Dirichlet Axiom targets
        # Sector 0: Physical Invariants (thermodynamics, conservation)
        physics_phases = (torch.rand(self.sector_dim, generator=g) * 2 * math.pi) - math.pi
        self.register_buffer('dirichlet_physics_real', torch.cos(physics_phases))
        self.register_buffer('dirichlet_physics_imag', torch.sin(physics_phases))
        
        # Sector 1: Sagnac Logic Axioms (truth/hallucination limits)
        logic_phases = (torch.rand(self.sector_dim, generator=g) * 2 * math.pi) - math.pi
        self.register_buffer('dirichlet_logic_real', torch.cos(logic_phases))
        self.register_buffer('dirichlet_logic_imag', torch.sin(logic_phases))
        
        # Sector 2: Thermodynamic Guardrails (Lipschitz bounds, phase-proximity)
        guard_phases = (torch.rand(self.sector_dim, generator=g) * 2 * math.pi) - math.pi
        self.register_buffer('dirichlet_guardrails_real', torch.cos(guard_phases))
        self.register_buffer('dirichlet_guardrails_imag', torch.sin(guard_phases))
        
        # 3. Initialize Sector 3: Active Neumann Boundary (Living Playbook)
        # Starts as a pure Gaussian potential on the unit circle
        neumann_phases = (torch.rand(self.sector_dim, generator=g) * 2 * math.pi) - math.pi
        self.register_buffer('neumann_active_real', torch.cos(neumann_phases))
        self.register_buffer('neumann_active_imag', torch.sin(neumann_phases))
        
        # 4. Self-Organizing Topological Manifold: 64 Complex features = 128 Real features.
        # This preserves the circular (S^1) topology of the waves
        self.shared_manifold = EmergentManifold(in_features=boundary_dim * 2, hidden_features=boundary_dim * 2)
        
        # 5. Attention/Routing Layer: learns to route emergent orthogonal features back to Dirichlet sectors
        self.routing_layer = torch_nn.Linear(boundary_dim * 2, boundary_dim * 2)
        with torch.no_grad():
            self.routing_layer.weight.copy_(torch.eye(boundary_dim * 2))
            self.routing_layer.bias.zero_()
            
        self.routing_optimizer = torch.optim.Adam(self.routing_layer.parameters(), lr=0.01)

    def bulk_to_boundary(self, bulk_wave: torch.Tensor) -> torch.Tensor:
        """Projects a 4096-D bulk wave to 64-D boundary tensor."""
        # bulk_wave shape: [4096]
        # P shape: [64, 4096]
        # Output shape: [64]
        dev = bulk_wave.device
        P = torch.complex(self.P_real.to(dev), self.P_imag.to(dev))
        h_cft = torch.mv(P, bulk_wave)
        return h_cft

    def validate_boundary(self, bulk_wave: torch.Tensor) -> tuple:
        """
        Projects wave to CFT boundary and validates the Dirichlet conditions.
        Returns:
            is_valid: bool
            veto_reason: str or None
            error_energy: float
            h_cft: projected 64-D boundary tensor
        """
        dev = bulk_wave.device
        
        # 1. Project down to 64-D boundary
        h_cft = self.bulk_to_boundary(bulk_wave)
        
        # 1b. Map to Euclidean Space for the Manifold (Preserves S^1 topology)
        is_batched = h_cft.ndim > 1
        if not is_batched:
            h_cft_batched = h_cft.unsqueeze(0)
        else:
            h_cft_batched = h_cft
            
        real_part = h_cft_batched.real
        imag_part = h_cft_batched.imag
        euclidean_wave = torch.cat([real_part, imag_part], dim=-1)
        
        # Apply Emergent Self-Organization (Hebbian + Sanger + Topological Closure)
        # Shared manifold training state is automatically managed by the validator's training attribute
        self.shared_manifold.train(self.training)
        crystallized_euclidean = self.shared_manifold(euclidean_wave)
        
        # Detach to isolate localized Hebbian/Sanger updates in shared_manifold from backprop
        crystallized_euclidean_detached = crystallized_euclidean.detach()
        
        # Map back to target validation slots using Attention/Routing layer
        routed_euclidean = self.routing_layer(crystallized_euclidean_detached)
        
        # Split back to Real and Imaginary and reconstruct complex h_cft
        new_real, new_imag = torch.chunk(routed_euclidean, 2, dim=-1)
        crystallized_h_cft_batched = torch.complex(new_real, new_imag)
        
        if not is_batched:
            crystallized_h_cft = crystallized_h_cft_batched.squeeze(0)
        else:
            crystallized_h_cft = crystallized_h_cft_batched
            
        # 2. Extract 16-D sectors from crystallized CFT wave
        sector_physics = crystallized_h_cft[0:16]
        sector_logic = crystallized_h_cft[16:32]
        sector_guardrails = crystallized_h_cft[32:48]
        sector_neumann = crystallized_h_cft[48:64]
        
        # Normalize sectors to unit magnitude for phase-only comparison
        def norm_sec(s):
            m = torch.abs(s)
            m = torch.clamp(m, min=1e-8)
            return s / m
            
        sec_phys_norm = norm_sec(sector_physics)
        sec_log_norm = norm_sec(sector_logic)
        sec_guard_norm = norm_sec(sector_guardrails)
        
        # Reconstruct complex targets on the current device
        dirichlet_physics = torch.complex(self.dirichlet_physics_real.to(dev), self.dirichlet_physics_imag.to(dev))
        dirichlet_logic = torch.complex(self.dirichlet_logic_real.to(dev), self.dirichlet_logic_imag.to(dev))
        dirichlet_guardrails = torch.complex(self.dirichlet_guardrails_real.to(dev), self.dirichlet_guardrails_imag.to(dev))
        
        # 3. Compute deviations (distance from Dirichlet axioms)
        dev_physics = 1.0 - torch.real(torch.sum(sec_phys_norm * torch.conj(dirichlet_physics))) / self.sector_dim
        dev_logic = 1.0 - torch.real(torch.sum(sec_log_norm * torch.conj(dirichlet_logic))) / self.sector_dim
        dev_guardrails = 1.0 - torch.real(torch.sum(sec_guard_norm * torch.conj(dirichlet_guardrails))) / self.sector_dim
        
        # Compute overall error energy (mean of deviations)
        error_energy = (dev_physics + dev_logic + dev_guardrails).item() / 3.0
        
        # Update Routing Layer weights on the validation loss
        if self.training:
            loss = (dev_physics + dev_logic + dev_guardrails) / 3.0
            self.routing_optimizer.zero_grad()
            loss.backward()
            self.routing_optimizer.step()
            
        # 4. Check against Lipschitz / Epsilon-Spine threshold
        if dev_physics > self.epsilon_spine:
            return False, f"Physical Invariant Veto (deviation: {dev_physics.item():.4f} > {self.epsilon_spine:.2f})", error_energy, crystallized_h_cft
            
        if dev_logic > self.epsilon_spine:
            return False, f"Sagnac Logic Veto (deviation: {dev_logic.item():.4f} > {self.epsilon_spine:.2f})", error_energy, crystallized_h_cft
            
        if dev_guardrails > self.epsilon_spine:
            return False, f"Thermodynamic Guardrail Veto (deviation: {dev_guardrails.item():.4f} > {self.epsilon_spine:.2f})", error_energy, crystallized_h_cft
            
        return True, None, error_energy, crystallized_h_cft

    def update_neumann_boundary(self, reflection_delta_cft: torch.Tensor, alignment_score: float):
        """
        Dynamically shifts the Active Neumann Boundary (Living Playbook)
        in response to the error delta from the learning loop.
        """
        dev = reflection_delta_cft.device
        # Expose Sector 3 reflection delta
        delta_neumann = reflection_delta_cft[48:64]
        
        # Ensure alignment_score is a scalar float
        if isinstance(alignment_score, (np.ndarray, torch.Tensor)):
            if hasattr(alignment_score, "mean"):
                alignment_score = alignment_score.mean().item()
            else:
                alignment_score = float(np.mean(alignment_score))
        else:
            alignment_score = float(alignment_score)

        # Compute dynamic learning rate based on alignment
        lr = 0.05 * (1.0 - alignment_score)
        
        with torch.no_grad():
            # Reconstruct complex active neumann on device
            neumann_active = torch.complex(self.neumann_active_real.to(dev), self.neumann_active_imag.to(dev))
            
            # Apply shift along error direction
            neumann_active += lr * delta_neumann
            # Project back to unit circle to maintain normalization
            mags = torch.abs(neumann_active)
            mags = torch.clamp(mags, min=1e-8)
            neumann_active = neumann_active / mags
            
            self.neumann_active_real.copy_(neumann_active.real)
            self.neumann_active_imag.copy_(neumann_active.imag)
            
        print(f"[BOUNDARY] Active Neumann Boundary updated (Learning Rate: {lr:.4f}).")
