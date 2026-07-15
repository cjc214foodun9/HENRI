import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
import os

from bioactive_thermodynamic_master import BioactiveThermodynamicMaster
from grassmannian_kuramoto_init import GrassmannianKuramotoInitializer
from oak_thermodynamic_engine import ThermodynamicCreditAssigner, SpectralOptionDelineator, LangevinEpistemicPlayLoop

class FractionalBindingLayer(nn.Module):
    """
    ENGINEERING SPECIFICATION: PROJECT HENRI - FRACTIONAL COORDINATE BINDING
    Implements continuous-phase fractional binding in the spectral domain.
    X^x = iFFT( FFT(X)^x )
    """
    def __init__(self, dim=4096):
        super().__init__()
        self.dim = dim
        
        # We need to constrain the phase to a narrow band to prevent 2*pi aliasing 
        max_x = 30.0 # Standard max grid dimension in ARC
        x_base_fft = torch.fft.rfft(torch.randn(dim))
        y_base_fft = torch.fft.rfft(torch.randn(dim))
        
        x_angle = torch.angle(x_base_fft) * (np.pi / (max_x * np.pi)) 
        y_angle = torch.angle(y_base_fft) * (np.pi / (max_x * np.pi))
        
        self.register_buffer('X_fft', torch.polar(torch.ones_like(x_angle), x_angle))
        self.register_buffer('Y_fft', torch.polar(torch.ones_like(y_angle), y_angle))

    def bind_coordinate(self, obj_identity: torch.Tensor, x: float, y: float) -> torch.Tensor:
        obj_fft = torch.fft.rfft(obj_identity)
        x_phase = self.X_fft ** x
        y_phase = self.Y_fft ** y
        bound_fft = obj_fft * x_phase * y_phase
        return torch.fft.irfft(bound_fft, n=self.dim)

class GrassmannianBlockSparseFeaturizer(nn.Module):
    """
    ENGINEERING SPECIFICATION: PROJECT HENRI - GRASSMANNIAN BLOCK-SPARSE FEATURIZER
    z_BSF = Σ (W_c * W_c^T) * z
    """
    def __init__(self, dim=4096, num_blocks=8, block_rank=32):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        self.block_rank = block_rank
        self.blocks = nn.Parameter(torch.randn(num_blocks, dim, block_rank))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for c in range(self.num_blocks):
                q, r = torch.linalg.qr(self.blocks[c])
                self.blocks[c].copy_(q)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_out = torch.zeros_like(z)
        for c in range(self.num_blocks):
            # Apply continuous Newton-Schulz retraction to halt parameter drift
            W_c = NewtonSchulzProjector.retract(self.blocks[c])
            # Reassign to parameter to maintain Stiefel structure across epochs
            if self.training:
                with torch.no_grad():
                    self.blocks[c].copy_(W_c)
                    
            if z.dtype == torch.complex64:
                proj = torch.matmul(z, torch.conj(W_c.T))
                z_out += torch.matmul(proj, W_c)
            else:
                proj = torch.matmul(z, W_c)
                z_out += torch.matmul(proj, W_c.T)
        return F.normalize(z_out, p=2, dim=-1)

class NewtonSchulzProjector:
    """
    Retracts a square matrix back onto the Stiefel Manifold (W^H W = I).
    """
    @staticmethod
    def retract(W: torch.Tensor, iterations=5) -> torch.Tensor:
        # Prevent division by zero
        if torch.max(torch.abs(W)) == 0.0:
            return W
            
        try:
            from triton_physics_kernels import triton_complex_matmul
            triton_available = True
        except ImportError:
            triton_available = False

        X = W / (torch.norm(W, p='fro') + 1e-9) # strict spectral radius precondition
        I = torch.eye(W.shape[-1], device=W.device, dtype=W.dtype)
        
        for _ in range(iterations):
            X_H = torch.conj(X.T)
            
            if triton_available and X.dtype == torch.complex64 and X.is_cuda:
                A = triton_complex_matmul(X_H, X)
                update_term = 3.0 * I - A
                X = triton_complex_matmul(X, update_term) / 2.0
            else:
                A = torch.matmul(X_H, X)
                update_term = 3.0 * I - A
                X = torch.matmul(X, update_term) / 2.0
                
        return X

class DarwinianPhaseSwarm(nn.Module):
    """
    ENGINEERING SPECIFICATION: PROJECT HENRI - GRADIENT-FREE DARWINIAN PHASE SWARM
    Dual-Stage Architecture:
    1. Exploration Loop: 16 experts mutate strictly in the spectral phase domain [0, 2π).
    2. Crystallization Step: SVD/Newton-Schulz retraction onto Stiefel spatial weights when Δ_Sagnac -> 0.
    """
    def __init__(self, num_experts=16, dim=4096):
        super().__init__()
        self.num_experts = num_experts
        self.dim = dim
        
        # Bioactive Master & K Matrix
        self.bio_master = BioactiveThermodynamicMaster(num_oscillators=dim, num_experts=num_experts)
        self.register_buffer('K_matrix', GrassmannianKuramotoInitializer(d_ambient=dim, num_blocks=1024, block_dim=4).generate_block_sparse_coupling().float())
        
        # OaK Thermodynamic Credit Assigner
        self.credit_assigner = ThermodynamicCreditAssigner(temperature_beta=2.0)
        
        # 16 Experts: Initialize spatial phase angles [0, 2π)
        # Shape: (16, dim)
        initial_phases = torch.rand(num_experts, dim) * 2 * math.pi
        self.expert_phases = nn.Parameter(initial_phases, requires_grad=False)
        
        # The permanent spatial Stiefel weights (The "Bone" matrix)
        self.spatial_bone = nn.Parameter(torch.eye(dim), requires_grad=False)

    def get_expert_wave(self, expert_idx: int) -> torch.Tensor:
        """Constructs the spatial wave for an expert."""
        phase = self.expert_phases[expert_idx]
        return torch.polar(torch.ones_like(phase), phase)

    def forward(self, input_wave: torch.Tensor) -> torch.Tensor:
        """
        The unamputated physics core for Epistemic Play.
        The wave physically collides with the internal topology (the Stiefel bone matrix).
        """
        # Pass the wave through the structural topology of the network
        with torch.no_grad():
            output = torch.matmul(self.spatial_bone.to(input_wave.device, dtype=input_wave.dtype), input_wave)
            return output / torch.abs(output)

    def process_swarm(self, target_axioms: torch.Tensor, t_step: float):
        """
        Executes the biological coupled relaxation step via BioactiveThermodynamicMaster
        and assigns credit via the ThermodynamicCreditAssigner.
        Returns the crystallized_output, optimal_wave, consensus_wave, best_sagnac, T_eff
        """
        with torch.no_grad():
            # Extract the full thermodynamic state of the swarm
            expert_waves, sagnac_deltas, theta_new, T_eff = self.bio_master.execute_coupled_relaxation_step(
                self.expert_phases, self.K_matrix, target_axioms, t_step
            )
            self.expert_phases.copy_(theta_new)
            
            # Apply the Principle of Least Action (Thermodynamic Credit Assignment)
            optimal_wave, weighted_consensus_wave, coherence_weights = self.credit_assigner(
                expert_waves, sagnac_deltas
            )
            
            best_sagnac = torch.min(sagnac_deltas).item()
            
            # Crystallize using the continuous weighted consensus instead of isolated, brittle top-1 selection
            if best_sagnac < 0.05:
                self.crystallize_bone(weighted_consensus_wave)
            
            return optimal_wave, weighted_consensus_wave, best_sagnac, T_eff

    def crystallize_bone(self, consensus_wave: torch.Tensor):
        """
        Crystallization Step:
        When Δ_Sagnac -> 0, project the superposed consensus wave back onto the spatial Stiefel weights.
        """
        print(f"[CRYSTALLIZATION] Locking OaK Consensus Wave to Spatial Bone...")
        winning_real = consensus_wave.real
        update_matrix = torch.outer(winning_real, winning_real)
        new_bone = self.spatial_bone + (0.1 * update_matrix)
        
        # Hard-lock using Newton-Schulz retraction
        locked_bone = NewtonSchulzProjector.retract(new_bone)
        self.spatial_bone.copy_(locked_bone)
        print("[CRYSTALLIZATION] Stiefel spatial weights rigidly updated via Newton-Schulz Retraction.")

class PhaseSwarmOrchestrator:
    def __init__(self, dim=4096, telemetry_logger=None):
        self.dim = dim
        self.telemetry = telemetry_logger
        self.binder = FractionalBindingLayer(dim=dim)
        self.delineator = SpectralOptionDelineator(dim=dim)
        
    def encode_grid_to_wave(self, grid: list[list[int]]) -> torch.Tensor:
        """Converts a 2D ARC grid into a fractionally bound spatial wave."""
        # Simple one-hot object identity base
        obj_identity = F.normalize(torch.randn(self.dim), p=2, dim=0)
        waves = []
        for y, row in enumerate(grid):
            for x, color in enumerate(row):
                if color > 0: # Only bind foreground objects
                    waves.append(self.binder.bind_coordinate(obj_identity, float(x), float(y)))
                    
        if len(waves) == 0:
            return torch.zeros(self.dim)
            
        stacked = torch.stack(waves)
        return F.normalize(stacked.sum(dim=0), p=2, dim=-1)
        
    def crystallize_boundary_axiom(self, train_pairs: list[dict]) -> torch.Tensor:
        """Extracts the continuous affine transformation boundary."""
        transformations = []
        for pair in train_pairs:
            wave_in = self.encode_grid_to_wave(pair['input'])
            wave_out = self.encode_grid_to_wave(pair['output'])
            # T = Y - X (phase shift in spatial domain)
            transformations.append(wave_out - wave_in)
            
        if len(transformations) == 0:
            return torch.zeros(self.dim)
            
        return F.normalize(torch.stack(transformations).sum(dim=0), p=2, dim=-1)

    def run_active_inference(self, task_id: str, task_wave: torch.Tensor, boundary_axiom: torch.Tensor, zone_c_axioms: torch.Tensor = None, max_epochs: int = 1000000) -> torch.Tensor:
        """Executes the Darwinian Phase Swarm optimization with Spectral Option Delineation and Test-Time Epistemic Play."""
        swarm = DarwinianPhaseSwarm(dim=self.dim)
        
        # 0. Test-Time Learning (Dynamic Micro-Latent Space Exploration)
        # We spawn an Epistemic Play session right here dynamically to map the immediate micro-latent space
        # and extract invariants that are directly relevant to this specific step.
        print(f"\n[OaK] Initiating In-Situ Test-Time Epistemic Play for Task {task_id}...")
        play_engine = LangevinEpistemicPlayLoop(core_syncytium=swarm, dim=self.dim)
        
        # The play engine runs until it discovers an invariant (or hits the 4096 failsafe).
        # It is anchored to the Zone C axioms via Ornstein-Uhlenbeck drift.
        known_axioms = play_engine.execute_play_epoch(target_axioms=zone_c_axioms, heat_variance=0.5)
        print(f"[OaK] Test-Time Play Terminated. Extracted {len(known_axioms)} local invariants.")
        
        # 1. OaK Option Delineation
        # Decompose the goal wave into orthogonal sub-harmonics if we have known axioms
        if known_axioms:
            axioms_tensor = torch.stack(known_axioms)
            sub_options = self.delineator(boundary_axiom, axioms_tensor)
        else:
            sub_options = boundary_axiom.unsqueeze(0)
            
        print(f"[OaK] Delineated {sub_options.shape[0]} sub-options for Task {task_id}.")
        
        final_wave = None
        for opt_idx, target_option in enumerate(sub_options):
            print(f"[OaK] Executing Sub-Option {opt_idx+1}/{sub_options.shape[0]}...")
            for epoch in range(max_epochs):
                t_step = epoch * 0.01 # Simulated continuous time
                
                # Execute the biological coupled relaxation step and OaK credit assignment
                optimal_wave, consensus_wave, best_error, T_eff = swarm.process_swarm(target_option, t_step)
                
                # Log telemetry
                if self.telemetry:
                    self.telemetry.log_wave_state(
                        task_id=f"{task_id}_OPT_{opt_idx}",
                        epoch=epoch,
                        sagnac_error=best_error,
                        langevin_heat=T_eff,
                        policy_action_decoded="OAK_THERMODYNAMIC_RELAXATION",
                        is_isothermal_lock=(best_error < 0.05)
                    )
                    
                if best_error < 0.05:
                    final_wave = optimal_wave
                    break
                    
        # Return best wave even if not perfectly locked
        return final_wave if final_wave is not None else optimal_wave
