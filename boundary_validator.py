import torch
import numpy as np
import math

class BoundaryAxiomValidator:
    """
    Implements the Dirichlet/Neumann holographic boundary validation.
    Projects the 4096-D bulk wave to a 64-D CFT boundary tensor (h_cft),
    validates the three Dirichlet sectors against mathematical axioms, and 
    manages the adaptive Active Neumann Boundary sector.
    """
    def __init__(self, bulk_dim=4096, boundary_dim=64, epsilon_spine=0.35, seed=42):
        self.bulk_dim = bulk_dim
        self.boundary_dim = boundary_dim
        self.sector_dim = boundary_dim // 4  # 16 dimensions per sector
        self.epsilon_spine = epsilon_spine    # Lipschitz continuity threshold
        
        # 1. Initialize fixed projection matrix P: shape [64, 4096] (complex64)
        # Using a fixed seed ensures identical mapping across instances
        g = torch.Generator()
        g.manual_seed(seed)
        
        phases = (torch.rand(boundary_dim, bulk_dim, generator=g) * 2 * math.pi) - math.pi
        self.P = torch.polar(torch.ones(boundary_dim, bulk_dim), phases)
        # Normalize rows to make it a projection
        self.P = self.P / math.sqrt(bulk_dim)
        
        # 2. Initialize the Dirichlet Axiom targets
        # Sector 0: Physical Invariants (thermodynamics, conservation)
        self.dirichlet_physics = torch.polar(torch.ones(self.sector_dim), (torch.rand(self.sector_dim, generator=g) * 2 * math.pi) - math.pi)
        
        # Sector 1: Sagnac Logic Axioms (truth/hallucination limits)
        self.dirichlet_logic = torch.polar(torch.ones(self.sector_dim), (torch.rand(self.sector_dim, generator=g) * 2 * math.pi) - math.pi)
        
        # Sector 2: Thermodynamic Guardrails (Lipschitz bounds, phase-proximity)
        self.dirichlet_guardrails = torch.polar(torch.ones(self.sector_dim), (torch.rand(self.sector_dim, generator=g) * 2 * math.pi) - math.pi)
        
        # 3. Initialize Sector 3: Active Neumann Boundary (Living Playbook)
        # Starts as a pure Gaussian potential on the unit circle
        neumann_phases = (torch.rand(self.sector_dim, generator=g) * 2 * math.pi) - math.pi
        self.neumann_active = torch.polar(torch.ones(self.sector_dim), neumann_phases)

    def bulk_to_boundary(self, bulk_wave: torch.Tensor) -> torch.Tensor:
        """Projects a 4096-D bulk wave to 64-D boundary tensor."""
        # bulk_wave shape: [4096]
        # P shape: [64, 4096]
        # Output shape: [64]
        h_cft = torch.mv(self.P, bulk_wave)
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
        # 1. Project down to 64-D boundary
        h_cft = self.bulk_to_boundary(bulk_wave)
        
        # 2. Extract 16-D sectors
        sector_physics = h_cft[0:16]
        sector_logic = h_cft[16:32]
        sector_guardrails = h_cft[32:48]
        sector_neumann = h_cft[48:64]
        
        # Normalize sectors to unit magnitude for phase-only comparison
        def norm_sec(s):
            m = torch.abs(s)
            m = torch.clamp(m, min=1e-8)
            return s / m
            
        sec_phys_norm = norm_sec(sector_physics)
        sec_log_norm = norm_sec(sector_logic)
        sec_guard_norm = norm_sec(sector_guardrails)
        
        # 3. Compute deviations (distance from Dirichlet axioms)
        # Cosine similarity difference: 1.0 - Re(s * target*)
        dev_physics = 1.0 - torch.real(torch.sum(sec_phys_norm * torch.conj(self.dirichlet_physics))) / self.sector_dim
        dev_logic = 1.0 - torch.real(torch.sum(sec_log_norm * torch.conj(self.dirichlet_logic))) / self.sector_dim
        dev_guardrails = 1.0 - torch.real(torch.sum(sec_guard_norm * torch.conj(self.dirichlet_guardrails))) / self.sector_dim
        
        # Compute overall error energy (mean of deviations)
        error_energy = (dev_physics + dev_logic + dev_guardrails).item() / 3.0
        
        # 4. Check against Lipschitz / Epsilon-Spine threshold
        if dev_physics > self.epsilon_spine:
            return False, f"Physical Invariant Veto (deviation: {dev_physics.item():.4f} > {self.epsilon_spine:.2f})", error_energy, h_cft
            
        if dev_logic > self.epsilon_spine:
            return False, f"Sagnac Logic Veto (deviation: {dev_logic.item():.4f} > {self.epsilon_spine:.2f})", error_energy, h_cft
            
        if dev_guardrails > self.epsilon_spine:
            return False, f"Thermodynamic Guardrail Veto (deviation: {dev_guardrails.item():.4f} > {self.epsilon_spine:.2f})", error_energy, h_cft
            
        return True, None, error_energy, h_cft

    def update_neumann_boundary(self, reflection_delta_cft: torch.Tensor, alignment_score: float):
        """
        Dynamically shifts the Active Neumann Boundary (Living Playbook)
        in response to the error delta from the learning loop.
        """
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
            # Apply shift along error direction
            self.neumann_active += lr * delta_neumann
            # Project back to unit circle to maintain normalization
            mags = torch.abs(self.neumann_active)
            mags = torch.clamp(mags, min=1e-8)
            self.neumann_active.copy_(self.neumann_active / mags)
            
        print(f"[BOUNDARY] Active Neumann Boundary updated (Learning Rate: {lr:.4f}).")
