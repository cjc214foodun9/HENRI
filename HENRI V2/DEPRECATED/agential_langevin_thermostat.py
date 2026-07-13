import torch
import math
from typing import Iterable, List

class AgentialLangevinThermostat:
    """
    The Agential Langevin Thermostat.
    
    This module enforces continuous-time test-time learning via Viscoelastic Creep.
    It rejects the static weight paradigm, treating model parameters as a physical 
    cartilage that deforms under the thermodynamic stress of Sagnac phase conflict.
    """
    
    def __init__(self, t_base: float = 0.01, kappa: float = 2.5, mu: float = 0.005, noise_scale: float = 1e-3):
        """
        Initializes the thermodynamic boundaries.
        
        Args:
            t_base (float): The absolute minimum temperature (0.01) representing the 
                            frozen, low-entropy attractor state (Crystallization).
            kappa (float): The thermal scaling coefficient governing max heat injection.
            mu (float): The viscoelastic yield rate (equivalent to learning rate).
            noise_scale (float): Dampening factor to map thermodynamic heat to parameter scales.
        """
        self.t_base = t_base
        self.kappa = kappa
        self.mu = mu
        self.noise_scale = noise_scale
        self.k_B = 1.0 # Normalized Boltzmann constant for the phase manifold

    def compute_temperature(self, sagnac_delta: torch.Tensor) -> torch.Tensor:
        """
        Computes the instantaneous kinetic temperature based on the Sagnac homodyne error.
        
        Equation: T(Δ_Sagnac) = T_base + κ * (1 - exp(-Δ_Sagnac))
        
        As Sagnac error approaches 0, the system cools to T_base (0.01).
        High error results in a thermal spike, melting the logic locks.
        """
        # Ensure delta is strictly positive to preserve thermodynamic laws
        delta = torch.clamp(sagnac_delta, min=0.0)
        
        temperature = self.t_base + self.kappa * (1.0 - torch.exp(-delta))
        return temperature

    @torch.no_grad()
    def apply_viscoelastic_creep(self, 
                                 parameters: Iterable[torch.nn.Parameter], 
                                 sagnac_delta: torch.Tensor,
                                 dt: float = 1.0) -> None:
        """
        Executes the in-situ parameter deformation during the test-time inference loop.
        
        Equation: ∂W/∂t = -μ ∇_W F(Ψ, W) + √(2 * T(Δ_Sagnac) * η(t))
        
        Args:
            parameters: The low-rank matrices (Cartilage) of the fluid experts.
            sagnac_delta: The measured geometric contradiction scalar.
            dt: The continuous-time integration step size.
        """
        # 1. Calculate the thermodynamic state
        current_temp = self.compute_temperature(sagnac_delta)
        
        # Calculate the amplitude of the Langevin noise
        # √(2 * T * dt) scaled by noise_scale to match the nn.Parameter initialization domain (1/sqrt(4096) ≈ 0.015)
        noise_amplitude = self.noise_scale * torch.sqrt(2.0 * current_temp * dt)
        
        for param in parameters:
            if param.grad is None:
                continue
                
            
            # The Free Energy gradient: ∇_W F(Ψ, W)
            free_energy_grad = param.grad
            
            # Generate the stochastic resonance (η(t) ~ N(0, 1))
            # Must strictly match the topology of the parameter manifold
            eta = torch.randn_like(param)
            
            # The physical yield: deterministic gradient descent + stochastic thermal Brownian motion
            yield_step = -(self.mu * free_energy_grad * dt) + (noise_amplitude * eta)
            
            # Deform the matrix in-place
            param.add_(yield_step)
            
            # Manifold Invariant Enforcement: 
            # In an actual E-O-E deployment, this is where Stiefel manifold projection 
            # (via Triton kernels) ensures the wave vectors maintain unit modulus.
            # self._project_to_stiefel_manifold(param)

    def is_crystallized(self, sagnac_delta: torch.Tensor, threshold: float = 1e-3) -> bool:
        """
        Diagnostic heuristic to determine if the continuous wave has collapsed 
        into the definitive low-entropy attractor for the ARC-AGI task.
        """
        return bool((sagnac_delta < threshold).all().item())