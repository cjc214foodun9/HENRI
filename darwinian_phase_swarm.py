import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
import os

# Import bioactive modules from scratch directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'scratch'))
from bioactive_thermodynamic_master import BioactiveThermodynamicMaster
from grassmannian_kuramoto_init import GrassmannianKuramotoInitializer

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
            W_c = self.blocks[c] 
            proj = torch.matmul(z, W_c)
            z_out += torch.matmul(proj, W_c.T)
        return F.normalize(z_out, p=2, dim=-1)

class NewtonSchulzProjector:
    """
    Retracts a square matrix back onto the Stiefel Manifold (W^T W = I).
    """
    @staticmethod
    def retract(W: torch.Tensor, iterations=5) -> torch.Tensor:
        # Prevent division by zero
        if torch.max(torch.abs(W)) == 0.0:
            return W
            
        X = W / torch.max(torch.abs(W)) # pre-condition
        for _ in range(iterations):
            A = torch.matmul(X.T, X)
            B = A.matmul(A)
            X = X.matmul(3.0 * torch.eye(W.shape[-1], device=W.device) - A) / 2.0
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

    def mutate_phases(self, target_wave: torch.Tensor, t_step: float) -> tuple[float, float, int]:
        """
        Executes the biological coupled relaxation step via BioactiveThermodynamicMaster.
        """
        with torch.no_grad():
            theta_new, T_eff, best_sagnac, best_idx = self.bio_master.execute_coupled_relaxation_step(
                self.expert_phases, self.K_matrix, target_wave, t_step
            )
            self.expert_phases.copy_(theta_new)
            return best_sagnac, T_eff, best_idx

    def crystallize_bone(self, winning_expert_idx: int):
        """
        Crystallization Step:
        When Δ_Sagnac -> 0, project the winning phase mask back onto the spatial Stiefel weights.
        """
        print(f"[CRYSTALLIZATION] Locking Expert {winning_expert_idx} to Spatial Bone...")
        winning_wave = self.get_expert_wave(winning_expert_idx)
        
        # Construct an outer product update mapping to push the bone matrix
        # W_new = Retract(W_old + outer(real(wave), real(wave)))
        winning_real = winning_wave.real
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

    def run_active_inference(self, task_id: str, task_wave: torch.Tensor, boundary_axiom: torch.Tensor, max_epochs: int = 200) -> torch.Tensor:
        """Executes the Darwinian Phase Swarm optimization."""
        swarm = DarwinianPhaseSwarm(dim=self.dim)
        
        # We need to find a policy wave that satisfies: task_wave + policy = boundary_axiom
        # Or more accurately, we just optimize phase to minimize Sagnac Error
        target_wave = boundary_axiom
        
        for epoch in range(max_epochs):
            t_step = epoch * 0.01 # Simulated continuous time
            
            # 1. Execute the biological coupled relaxation step
            best_error, T_eff, best_idx = swarm.mutate_phases(target_wave, t_step)
            
            # Log telemetry
            if self.telemetry:
                self.telemetry.log_wave_state(
                    task_id=task_id,
                    epoch=epoch,
                    sagnac_error=best_error,
                    langevin_heat=T_eff,
                    policy_action_decoded="BIOACTIVE_RELAXATION",
                    is_isothermal_lock=(best_error < 0.05)
                )
                
            if best_error < 0.05:
                # Crystallization
                swarm.crystallize_bone(best_idx)
                return swarm.get_expert_wave(best_idx)
            
        # Return best wave even if not perfectly locked
        best_idx = torch.argmin(sagnac_errors).item()
        return swarm.get_expert_wave(best_idx)
