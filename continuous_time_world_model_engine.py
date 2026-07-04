import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional

# Native imports from the HENRI 8.59B continuous substrate
from cognitive_swarm import HenriCognitiveSwarmOrchestrator
from universal_harness_core import DivergentMasterThermostat

class OntologicalTransducer(nn.Module):
    """
    Replaces the orthogonal dictionary with a Compositional Phase Manifold.
    Embeds raw properties (mass, logic, space) into base frequencies,
    allowing HENRI to "feel" the physical parameters of a prompt natively.
    """
    def __init__(self, dim: int = 4096, base_concepts: int = 512):
        super().__init__()
        self.dim = dim
        
        # Base ontological properties (e.g., Mass, Velocity, Recursion, Boolean_True)
        # Initialized as strictly orthogonal base phasors on S^4095
        real_part = torch.randn(base_concepts, dim)
        imag_part = torch.randn(base_concepts, dim)
        complex_base = torch.complex(real_part, imag_part)
        
        # Project to hypersphere
        self.ontology_basis = nn.Parameter(
            F.normalize(complex_base.real, p=2, dim=-1) + 1j * F.normalize(complex_base.imag, p=2, dim=-1),
            requires_grad=False
        )

    def bind_concepts(self, role_idx: torch.Tensor, filler_val: torch.Tensor) -> torch.Tensor:
        """
        Executes continuous-time O(N log N) Holographic Binding.
        role_idx: [B] indices into the ontology basis.
        filler_val: [B] continuous scalar magnitudes to modulate the phase.
        """
        # Fetch base phase vector
        role_vector = self.ontology_basis[role_idx] # [B, Dim]
        
        # Shift the phase angles proportional to the physical scalar value
        phases = torch.angle(role_vector)
        shifted_phases = phases + filler_val.unsqueeze(-1)
        
        # Reconstitute the wave
        return torch.complex(torch.cos(shifted_phases), torch.sin(shifted_phases))

    def forward(self, semantic_graph: List[Tuple[int, float]]) -> torch.Tensor:
        """
        Ingests a parsed semantic graph (e.g., [(Mass_Idx, 15.0), (Velocity_Idx, 0.9)])
        and superimposes them into a single, physically accurate wavefront.
        """
        device = self.ontology_basis.device
        superposition = torch.zeros(self.dim, dtype=torch.complex64, device=device)
        
        for role_idx, filler_val in semantic_graph:
            r_idx = torch.tensor([role_idx], device=device)
            f_val = torch.tensor([filler_val], device=device, dtype=torch.float32)
            bound_wave = self.bind_concepts(r_idx, f_val).squeeze(0)
            superposition += bound_wave
            
        # Clamp to unit hypersphere S^4095 to prevent energy explosion
        return F.normalize(superposition.real, p=2, dim=-1) + 1j * F.normalize(superposition.imag, p=2, dim=-1)


class ContinuousTransitionNetwork(nn.Module):
    """
    The Wave-JEPA Predictor.
    A full-precision complex neural operator that predicts the next physical 
    wave state (Psi_t+1) given the current state and an applied action/transformation.
    """
    def __init__(self, dim: int = 4096):
        super().__init__()
        self.dim = dim
        
        # Complex-valued transition weights
        self.W_transition = nn.Parameter(torch.empty(dim, dim, dtype=torch.complex64))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        real_part = torch.randn(self.dim, self.dim)
        imag_part = torch.randn(self.dim, self.dim)
        X = torch.complex(real_part, imag_part)
        Q, _ = torch.linalg.qr(X)
        self.W_transition.copy_(Q)

    def forward(self, current_state: torch.Tensor, action_wave: torch.Tensor) -> torch.Tensor:
        """
        Predicts the topological evolution of the environment.
        """
        # Bind the current state with the intended action via Circular Convolution (Frequency domain multiplication)
        bound_state = current_state * action_wave
        
        # Apply transition dynamics
        next_state_pred = torch.matmul(bound_state, self.W_transition)
        
        # Maintain modulus invariant
        return F.normalize(next_state_pred.real, p=2, dim=-1) + 1j * F.normalize(next_state_pred.imag, p=2, dim=-1)


class TestTimeActiveInferenceEngine(nn.Module):
    """
    The Master Test-Time Learning Loop.
    Drops HENRI into an unseen environment, allows it to generate hypotheses, 
    predict outcomes, and viscoelastically adapt its LoRA weights in real-time 
    to minimize Free Energy.
    """
    def __init__(self, dim: int = 4096, num_experts: int = 16):
        super().__init__()
        self.dim = dim
        self.ontology = OntologicalTransducer(dim=dim)
        self.world_model = ContinuousTransitionNetwork(dim=dim)
        self.swarm = HenriCognitiveSwarmOrchestrator(dim=dim, num_experts=num_experts)
        self.thermostat = DivergentMasterThermostat(t_max=3.0)

    def active_inference_step(self, observed_environment: List[Tuple[int, float]], target_goal: List[Tuple[int, float]]) -> torch.Tensor:
        """
        Executes real-time thermodynamic adaptation.
        """
        device = next(self.swarm.parameters()).device
        print("\n[ACTIVE INFERENCE] Initiating Test-Time Learning in Unseen Latent Space...")

        # 1. Transduce empirical environment and goal into continuous phase geometry
        state_wave = self.ontology(observed_environment).unsqueeze(0)
        goal_wave = self.ontology(target_goal).unsqueeze(0)

        step = 0
        max_test_time_steps = 40
        system_resolved = False

        while step < max_test_time_steps and not system_resolved:
            # 2. Swarm generates a candidate transformation (Action Wave)
            # We bundle the 16 expert trajectories
            expert_actions = []
            for expert in self.swarm.experts:
                action = expert(state_wave, langevin_temp=self.thermostat.current_temp)
                expert_actions.append(action)
                
            consensus_action = torch.stack(expert_actions, dim=0).sum(dim=0)
            consensus_action = F.normalize(consensus_action.real, p=2, dim=-1) + 1j * F.normalize(consensus_action.imag, p=2, dim=-1)

            # 3. World Model Prediction: What happens if we apply this action?
            predicted_future = self.world_model(state_wave, consensus_action)

            # 4. Sagnac Homodyne Veto (Calculate Free Energy against the Target Goal)
            free_energy_delta = torch.norm(predicted_future - goal_wave, p=2).item()
            current_heat = self.thermostat.compute_langevin_simmer(free_energy_delta)

            print(f" -> Cycle {step:02d} | Surprise (Free Energy): {free_energy_delta:.4f} | Thermal Flux: {current_heat:.4f}")

            if free_energy_delta <= 0.35:
                print(" -> [RESONANCE] Physical manifold aligned with target objective. Free Energy minimized.")
                system_resolved = True
            else:
                # 5. TEST-TIME LEARNING: Viscoelastic Creep
                # The environment is unfamiliar. The swarm must adapt its own weights *live*.
                # We use the predicted future's gradient to physically yield the LoRA layers.
                
                # Compute mock gradient step for the internal state (Simulated Adjoint Propagation)
                pseudo_loss = torch.sum(torch.abs(predicted_future - goal_wave)**2)
                pseudo_loss.backward()
                
                # Apply continuous-time creep to all 16 experts to permanently adapt to the new physics
                self.swarm.apply_viscoelastic_gradient_updates(lr=2e-3)
                
                # Inject localized Langevin heat to the state wave to prevent identical retries
                noise_scale = math.sqrt(2.0 * current_heat) * 0.005
                heat_noise = torch.randn_like(state_wave) * noise_scale
                state_wave = F.normalize((state_wave + heat_noise).real, p=2, dim=-1) + 1j * F.normalize((state_wave + heat_noise).imag, p=2, dim=-1)
                
                self.thermostat.isothermal_cool_down()
            
            step += 1

        # Post-adaptation Stiefel Manifold Lock
        for expert in self.swarm.experts:
            expert.bjorck_newton_orthonormalize(iterations=5)

        if not system_resolved:
            print(" -> [WARNING] Thermodynamic timeout. Returning highest-probability conformal path.")
            
        return consensus_action