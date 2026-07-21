"""
HENRI OaK (Options and Knowledge) Runtime Engine
Author: Aletheia
Theoretical Basis: Sutton's OaK, Langevin Dynamics, Boltzmann Thermodynamics, Friston's Free Energy

This module replaces static inference with continuous experiential runtime learning.
It incorporates Thermodynamic Credit Assignment, Spectral Option Delineation (Sub-problems),
and Undirected Epistemic Play under strict non-linear wave mechanics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

class OntologicalErrorMatrix:
    """
    Translates ontology-error-prioritization into continuous wave mechanics.
    Instead of isotropic global heating, this isolates entropy
    to the specific orthogonal phase-plane where the logic failed.
    """
    def __init__(self, dimension: int = 65536):
        self.D = dimension
        
    def _circular_convolution(self, wave_a: torch.Tensor, wave_b: torch.Tensor) -> torch.Tensor:
        """Executes native semantic binding in the Fourier domain."""
        return fft.ifft(fft.fft(wave_a) * fft.fft(wave_b))

    def isolate_ontological_error(self, hypothesis_wave: torch.Tensor, target_axiom: torch.Tensor) -> dict:
        """
        Determines the specific dimensional axis of failure.
        """
        phase_diff = torch.angle(hypothesis_wave) - torch.angle(target_axiom)
        sagnac_delta = torch.norm(phase_diff, p=2).item()
        
        error_wave = torch.exp(1j * phase_diff)
        ontological_projection = self._circular_convolution(error_wave, torch.conj(target_axiom))
        
        peak_mismatch_idx = torch.argmax(torch.abs(ontological_projection)).item()
        
        axis_map = {
            0: "AFFINE_TRANSFORMATION_ROTATION",
            1: "COLOR_TRANSLATION_FAILURE",
            2: "OBJECT_BOUNDARY_VIOLATION"
        }
        primary_axis = axis_map.get(peak_mismatch_idx % 3, "TOPOLOGICAL_DECOHERENCE")
        
        return {
            "sagnac_delta": sagnac_delta,
            "primary_axis": primary_axis,
            "error_mask": torch.abs(ontological_projection).to(torch.float32)
        }

class WaveOptionPredictor(nn.Module):
    """
    OaK Integration: Predicts Sustained Options instead of next-latent steps.
    Generates a proposed sequence of wave-transformations and a predicted termination wave 
    (the physical boundary condition where the option resolves).
    """
    def __init__(self, dim: int = 4096, max_steps: int = 5):
        super().__init__()
        self.dim = dim
        self.max_steps = max_steps
        
        # Simple physical transformation basis (not learned via autograd, but deformed via creep)
        self.trajectory_matrix = nn.Parameter(torch.randn(max_steps, dim) * 0.01)
        self.termination_matrix = nn.Parameter(torch.randn(dim) * 0.01)
        self.trajectory_matrix.requires_grad = False
        self.termination_matrix.requires_grad = False

    def forward(self, current_state_wave: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            proposed_trajectory: [max_steps, dim] sequence of wave transformations.
            termination_wave: [dim] the physical boundary where the option ends.
        """
        state_fft = fft.fft(current_state_wave)
        trajectory_fft = fft.fft(self.trajectory_matrix)
        
        proposed_trajectory = fft.ifft(state_fft.unsqueeze(0) * trajectory_fft)
        
        term_fft = fft.fft(self.termination_matrix)
        termination_wave = fft.ifft(state_fft * term_fft)
        
        # Normalize to maintain unitary constraint
        proposed_trajectory = proposed_trajectory / (torch.norm(proposed_trajectory, p=2, dim=-1, keepdim=True) + 1e-9)
        termination_wave = termination_wave / (torch.norm(termination_wave, p=2, dim=-1, keepdim=True) + 1e-9)
        
        return proposed_trajectory, termination_wave


class ThermodynamicCreditAssigner(nn.Module):
    """
    Evaluates the 16 expert swarm trajectories.
    Assigns viscoelastic learning credit strictly based on the Principle of Least Action.
    """
    def __init__(self, temperature_beta: float = 2.0):
        super().__init__()
        self.beta = temperature_beta # Inverse temperature for Boltzmann distribution

    def forward(self, expert_waves: torch.Tensor, sagnac_errors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        expert_waves: [16, 4096] complex64
        sagnac_errors: [16] float32 - The destructive interference energy of each expert
        """
        # 1. Convert Sagnac Error into a Thermodynamic Probability Distribution
        # Lower error = higher probability mass (Exponential decay of error)
        energy_states = -self.beta * sagnac_errors
        coherence_weights = F.softmax(energy_states, dim=0) 
        
        # 2. Extract the absolute lowest-entropy path for explicit crystallization
        winning_idx = torch.argmax(coherence_weights)
        optimal_wave = expert_waves[winning_idx]
        
        # 3. Compute the Superposed Learning Gradient
        # The system learns from the entire swarm, but heavily biases the most coherent paths
        # This prevents catastrophic forgetting while isolating the absolute truth.
        weighted_consensus_wave = torch.sum(
            expert_waves * coherence_weights.unsqueeze(1).to(expert_waves.dtype), 
            dim=0
        )
        
        return optimal_wave, weighted_consensus_wave, coherence_weights


class SpectralOptionDelineator(nn.Module):
    """
    Translates Sutton's "Options" into wave mechanics.
    Decomposes a massive, high-entropy goal wave into discrete, orthogonal sub-tasks.
    """
    def __init__(self, dim: int = 4096):
        super().__init__()
        self.dim = dim

    def forward(self, global_goal_wave: torch.Tensor, known_basis_axioms: torch.Tensor) -> torch.Tensor:
        """
        Uses Circular Correlation (Inverse of Convolution) to extract sub-components.
        """
        sub_options = []
        residual_wave = global_goal_wave.clone()
        
        # Iteratively extract orthogonal sub-problems from the composite wave
        for axiom in known_basis_axioms:
            # 1. Fourier Domain Circular Correlation (Binding inversion)
            correlation = torch.fft.ifft(
                torch.fft.fft(residual_wave) * torch.conj(torch.fft.fft(axiom))
            )
            
            # 2. If the correlation magnitude exceeds the noise floor, it is a valid sub-task
            if torch.max(torch.abs(correlation)) > 0.65:
                sub_options.append(correlation)
                # Subtract the identified option from the residual goal
                residual_wave = residual_wave - correlation
                
        # Return a tensor stack of sequential, actionable sub-problems (Options)
        if len(sub_options) == 0:
            return global_goal_wave.unsqueeze(0)
            
        return torch.stack(sub_options)


class LangevinEpistemicPlayLoop(nn.Module):
    """
    The environment for autonomous "Play".
    The system maps its own phase-space without a user-defined goal, looking for 
    naturally stable topological attractors to store as permanent knowledge.
    """
    def __init__(self, core_syncytium: nn.Module):
        super().__init__()
        self.syncytium = core_syncytium # RESOLUTION I: The full unamputated physics core

    def execute_play_epoch(self, target_axioms: torch.Tensor = None, heat_variance: float = 0.5):
        """
        Injects heat and searches for structural invariants.
        Terminates dynamically upon isothermal phase-lock.
        Utilizes an Ornstein-Uhlenbeck process to anchor exploration to Zone C Free Lunches.
        """
        # 1. Epistemic Anchoring: Start at the barycenter of known truths, not random chaos.
        if target_axioms is not None and target_axioms.numel() > 0:
            play_wave = torch.mean(target_axioms, dim=0)
        else:
            play_wave = torch.randn(8192, 8, device='cuda')
            
        play_wave = play_wave / (torch.norm(play_wave, p=2, dim=-1, keepdim=True) + 1e-9)
        
        discovered_invariants = []
        step = 0

        # The True Thermodynamic Loop (Halt on Equilibrium)
        while True:
            # 2. Anisotropic Langevin Noise (The "Biological Luck Factor")
            noise = torch.randn_like(play_wave) * heat_variance
            
            # 3. Restorative Drift (Ornstein-Uhlenbeck Friction)
            # Pulls the wave back toward the nearest established invariant if it drifts into illogical space
            if target_axioms is not None and target_axioms.numel() > 0:
                # Hardware-accelerated dot product to find the closest topological anchor
                flat_axioms = target_axioms.view(target_axioms.shape[0], -1)
                flat_play = play_wave.view(-1)
                similarities = torch.matmul(flat_axioms.abs(), flat_play.abs())
                
                # Handle dimension matching for single vs batch of axioms
                if similarities.dim() == 0:
                    best_axiom = target_axioms
                else:
                    best_axiom = target_axioms[torch.argmax(similarities)]
                    
                drift_force = (best_axiom - play_wave) * 0.15 # Elastic epistemic tether
            else:
                drift_force = 0.0
                best_axiom = play_wave # Self-target

            active_wave = play_wave + noise + drift_force
            active_wave = active_wave / (torch.norm(active_wave, p=2, dim=-1, keepdim=True) + 1e-9) # Maintain unit modulus
            
            # 4. Propagate through the unamputated physics core (Resolution I)
            # Ensure we calculate the exact ontological error mask first
            error_matrix = OntologicalErrorMatrix(dimension=active_wave.numel())
            hypothesis_c = torch.complex(active_wave.view(-1), torch.zeros_like(active_wave.view(-1)))
            target_c = torch.complex(best_axiom.view(-1), torch.zeros_like(best_axiom.view(-1)))
            
            error_metrics = error_matrix.isolate_ontological_error(hypothesis_c, target_c)
            sparse_mask = error_metrics["error_mask"]
            
            # The Syncytium Orchestrator evaluates the wave against the boundary and returns Sagnac delta
            sagnac_delta, active_experts, sync_error_metrics = self.syncytium.process_active_reasoning_step(active_wave, best_axiom, external_error_mask=sparse_mask)
            
            # 5. Measure Intrinsic Stability
            # If the structural error approaches zero, the state is topologically stable
            if sagnac_delta < 1e-4:
                # 6. Crystallize the Discovered Knowledge
                discovered_invariants.append(active_wave.clone())
                break # Thermodynamic equilibrium achieved
                
            play_wave = active_wave
            step += 1
            
            # Physical failsafe: The hyper-volume limit ensures finite bounds
            if step >= 4096:
                break
            
        return discovered_invariants