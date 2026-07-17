import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional

class BiophysicalSwarm(nn.Module):
    """
    Biophysical Swarm Core utilizing Modulo Error Diffusion (arXiv:2606.31700).
    
    Scales to 1024+ parallel 'Explorers', represented strictly as lean phase-perturbation 
    vectors on the unit hypersphere S^4095, bypassing dense weight matrices entirely.
    Incorporates Temporal Error Centering, Asymmetric E/I Seeding, and Phase-Linewidth Gains
    for stable continuous-time learning at B=1.
    """
    def __init__(
        self,
        num_explorers: int = 1024,
        dimension: int = 4096,
        learning_rate: float = 0.05,
        coupling_strength: float = 0.1,
        alpha_gain: float = 3.0,
        ema_decay: float = 0.99,
        apoptosis_threshold_factor: float = 0.6,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.num_explorers = num_explorers
        self.dimension = dimension
        self.lr = learning_rate
        self.coupling_strength = coupling_strength
        
        self.alpha_gain = alpha_gain
        self.ema_decay = ema_decay
        self.apoptosis_threshold_factor = apoptosis_threshold_factor
        self.device = device

        # Phase parameters of the lean explorers: theta_k in [-pi, pi]
        self.explorers_theta = nn.Parameter(
            (torch.rand(num_explorers, dimension, device=device) * 2.0 * torch.pi) - torch.pi
        )

        # Asymmetric E/I Initialization (3:1 Expected Ratio)
        # Excitatory (E) represents phase-advancing forces; Inhibitory (I) represents phase-delaying.
        self.register_buffer("velocity_E", torch.rand(num_explorers, dimension, device=device) * 1.5)
        self.register_buffer("velocity_I", torch.rand(num_explorers, dimension, device=device) * 0.5)

        # Temporal Error Centering Buffer for B=1 streaming
        self.register_buffer("ema_phase_error", torch.zeros(dimension, device=device))

        # Dynamic Sagnac tracking metrics per explorer
        self.register_buffer("explorer_coherence", torch.zeros(num_explorers, device=device))
        self.register_buffer("explorer_temperature", torch.ones(num_explorers, device=device))

    def forward(
        self, 
        crystallized_baseline: torch.Tensor, 
        target_axioms: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Executes parallel wave-geometric exploration over the baseline engram.
        """
        # Ensure dimensions match for broadcast [num_explorers, dimension]
        M = crystallized_baseline.unsqueeze(0).expand(self.num_explorers, -1)
        
        if target_axioms.dim() == 1:
            T = target_axioms.unsqueeze(0).expand(self.num_explorers, -1)
        else:
            T = target_axioms

        # 1. Apply lean phase-perturbations: Psi_k = M * exp(i * theta_k)
        cos_theta = torch.cos(self.explorers_theta)
        sin_theta = torch.sin(self.explorers_theta)
        rotation_operator = torch.complex(cos_theta, sin_theta)
        
        candidate_waves = M * rotation_operator

        # 2. Extract Sagnac Interferometric Phase Mismatches
        angle_candidates = torch.angle(candidate_waves)
        angle_target = torch.angle(T)
        
        # Raw Sagnac phase error: delta_phi = (theta_k - theta_target) mod 2pi
        raw_error = torch.remainder(angle_candidates - angle_target + torch.pi, 2.0 * torch.pi) - torch.pi
        
        # Temporal Running-Average Error Centering (Replaces Batch-Centering)
        # Prevents persistent channel suppression in continuous B=1 streaming
        with torch.no_grad():
            mean_step_error = raw_error.mean(dim=0)
            self.ema_phase_error.mul_(self.ema_decay).add_(mean_step_error * (1.0 - self.ema_decay))
            
        centered_error = raw_error - self.ema_phase_error.unsqueeze(0)

        # Compute local explorer coherence: C_k = Mean(cos(raw_error))
        coherence = torch.mean(torch.cos(raw_error), dim=-1)
        self.explorer_coherence.copy_((coherence + 1.0) / 2.0)

        # 3. Modulo Error Diffusion (MED) under Dale's Excitatory/Inhibitory stream gating
        with torch.no_grad():
            # Phase-Linewidth Gain (Countering Attenuation)
            # Amplifies the feedback signal to prevent saturation deep in the phase baseline
            amplified_error = centered_error * self.alpha_gain
            
            # Excitatory phase-advancing force (+ values)
            force_E = F.relu(amplified_error)
            # Inhibitory phase-delaying force (- values)
            force_I = F.relu(-amplified_error)
            
            # Update Excitatory and Inhibitory velocities independently
            self.velocity_E.mul_(1.0 - self.coupling_strength).add_(force_E * self.lr)
            self.velocity_I.mul_(1.0 - self.coupling_strength).add_(force_I * self.lr)
            
            # Apply update to the active phase parameters
            new_theta = self.explorers_theta - (self.velocity_E - self.velocity_I)
            # Bound phase space to canonical [-pi, pi] manifold
            self.explorers_theta.copy_(
                torch.remainder(new_theta + torch.pi, 2.0 * torch.pi) - torch.pi
            )

        # 4. Generate Swarm Superposition via Coherence-Weighted Colimit
        weights = self.explorer_coherence / (torch.sum(self.explorer_coherence) + 1e-8)
        
        fused_output_wave = torch.sum(weights.unsqueeze(-1) * candidate_waves, dim=0)
        fused_output_wave = F.normalize(fused_output_wave, p=2, dim=-1)

        # Calculate global thermodynamic Free Energy
        epsilon = 1e-6
        free_energy = -torch.log((self.explorer_coherence + epsilon) / (1.0 + epsilon))

        telemetry = {
            "mean_coherence": torch.mean(self.explorer_coherence),
            "coherence_variance": torch.var(self.explorer_coherence),
            "global_free_energy": torch.mean(free_energy),
            "mean_velocity_E": torch.mean(self.velocity_E),
            "mean_velocity_I": torch.mean(self.velocity_I),
            "active_explorers": torch.tensor(float(self.num_explorers), device=self.device)
        }

        return fused_output_wave, telemetry

    def apoptotic_pruning_and_crystallization(self) -> List[int]:
        """
        Prunes highly discordant exploratory paths and regenerates them using
        the Right Kan pullback of the most coherent pathfinder.
        """
        mean_coherence = torch.mean(self.explorer_coherence)
        std_coherence = torch.std(self.explorer_coherence) if self.num_explorers > 1 else torch.tensor(0.0, device=self.device)
        
        threshold = mean_coherence - self.apoptosis_threshold_factor * std_coherence
        threshold = torch.clamp(threshold, min=0.1, max=0.9)

        pruned_indices = []
        elite_indices = []

        for k in range(self.num_explorers):
            if self.explorer_coherence[k] < threshold:
                pruned_indices.append(k)
            else:
                elite_indices.append(k)

        if len(pruned_indices) == 0 or len(elite_indices) == 0:
            return []

        with torch.no_grad():
            # Categorical Colimit: Temperature-scaled softmax superposition
            colimit_weights = F.softmax(self.explorer_coherence / 0.1, dim=0)
            complex_phases = torch.exp(1j * self.explorers_theta)
            weighted_complex = torch.sum(colimit_weights.unsqueeze(1) * complex_phases, dim=0)
            colimit_theta = torch.angle(weighted_complex)

            for p_idx in pruned_indices:
                self.explorer_temperature[p_idx] = 2.0 * (1.0 - self.explorer_coherence[p_idx])
                
                # Apply Right Kan completion with localized anisotropic phase jitter
                noise = torch.randn(self.dimension, device=self.device) * self.explorer_temperature[p_idx] * 0.05
                self.explorers_theta[p_idx].copy_(colimit_theta + noise)
                
                # Reset velocities while maintaining slight asymmetric bias
                self.velocity_E[p_idx].copy_(torch.rand(self.dimension, device=self.device) * 0.15)
                self.velocity_I[p_idx].copy_(torch.rand(self.dimension, device=self.device) * 0.05)
                self.explorer_coherence[p_idx] = mean_coherence

        return pruned_indices