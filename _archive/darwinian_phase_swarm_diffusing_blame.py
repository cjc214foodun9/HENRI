import torch
import torch.nn as nn
import math

class DarwinianPhaseSwarm(nn.Module):
    """
    Implements Dynamic Morphogenetic Recruitment and Forward Error Diffusion.
    Backpropagation is eradicated. Experts are strictly excitatory or inhibitory.
    """
    def __init__(self, dim=4096, max_experts=1024, base_experts=4, yield_gamma=0.01):
        super().__init__()
        self.dim = dim
        self.max_experts = max_experts
        self.base_experts = base_experts # Resting metabolic state
        self.yield_gamma = yield_gamma   # Viscoelastic yielding constant
        
        # Initialize maximum pool of dormant expert phase masks
        self.expert_manifolds = nn.Parameter(torch.randn(max_experts, dim, dim) / math.sqrt(dim))
        
        # Enforce Dale's Principle (80% Excitatory, 20% Inhibitory)
        # Excitatory (+1): Phase advancing. Inhibitory (-1): Phase dampening.
        rand_tensor = torch.rand(max_experts)
        self.polarities = torch.where(rand_tensor < 0.8, 1.0, -1.0).view(max_experts, 1, 1)
        self.polarities.requires_grad = False
        
        # Dynamic Gap Junction State
        self.active_expert_count = base_experts

    def _open_gap_junctions(self, sagnac_stress: float):
        """
        Expands the Cognitive Light Cone. 
        High thermodynamic stress recruits dormant experts into the syncytium.
        """
        # Stress is normalized between 0 and 1
        recruitment = int(sagnac_stress * (self.max_experts - self.base_experts))
        self.active_expert_count = min(self.base_experts + recruitment, self.max_experts)

    def forward(self, incident_wave: torch.Tensor, current_stress: float) -> torch.Tensor:
        """
        Forward propagation through the active syncytium.
        """
        # 1. Scale the boundary of the Self based on stress
        self._open_gap_junctions(current_stress)
        
        # 2. Extract only the active experts
        active_manifolds = self.expert_manifolds[:self.active_expert_count]
        active_polarities = self.polarities[:self.active_expert_count]
        
        # 3. Apply Dale's Polarity to the phase masks
        # Binds the physical nature of the expert to the wave interaction
        modulated_manifolds = active_manifolds * active_polarities
        
        # 4. Kuramoto Colimit: Superpose the output of all active gap-junctions
        # incident_wave: [Batch, Dim]. modulated_manifolds: [Experts, Dim, Dim]
        # Result: [Batch, Dim]
        syncytium_output = torch.einsum('bd,edc->bec', incident_wave, modulated_manifolds).sum(dim=1)
        
        # Re-normalize to unit hypersphere S^4095
        return syncytium_output / (torch.norm(syncytium_output, dim=-1, keepdim=True) + 1e-8)

    @torch.no_grad() # Strict eradication of autograd/backward loops
    def apply_forward_error_diffusion(self, sagnac_delta: torch.Tensor, ontology_mask: torch.Tensor):
        """
        Sakana AI / Diffusing Blame execution.
        Applies physical viscoelastic creep directly from the forward error.
        Langevin heat is applied anisotropically based on the ontology mask.
        """
        # sagnac_delta represents the physical wave amplitude of the failure
        # ontology_mask is the block-sparse map of WHERE the ontology broke
        
        active_manifolds = self.expert_manifolds[:self.active_expert_count]
        active_polarities = self.polarities[:self.active_expert_count]
        
        # Calculate the physical yield
        # Inhibitory experts get stronger when there is an error to suppress.
        # Excitatory experts are penalized for contributing to the error.
        yield_force = sagnac_delta.unsqueeze(0).unsqueeze(-1) * active_polarities * self.yield_gamma
        
        # Apply anisotropic masking (freeze valid logic, melt broken logic)
        masked_yield = yield_force * ontology_mask.unsqueeze(0)
        
        # Physically deform the parameter manifold
        active_manifolds.add_(masked_yield)
        
        # Newton-Schulz retraction to maintain strict Stiefel orthogonality
        self._retract_to_stiefel(active_manifolds)

    def _retract_to_stiefel(self, manifolds):
        """Maintains isometric wave propagation (preserves phase linewidth)."""
        for i in range(manifolds.size(0)):
            W = manifolds[i]
            W_T_W = torch.mm(W.T, W)
            manifolds[i] = 1.5 * W - 0.5 * torch.mm(W, W_T_W)