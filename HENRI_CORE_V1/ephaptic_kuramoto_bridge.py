import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EphapticCrystalContinuum(nn.Module):
    """
    Tier 1: Micro-Physics of Zone B.
    Simulates the continuous electromagnetic ephaptic coupling of the BTO crystal.
    Light waves do not use attention matrices; they bleed into each other through 
    the spatial Laplacian of the photorefractive effect.
    """
    def __init__(self, spatial_resolution=256, diffusion_coefficient=0.1, fwm_coupling=1.5):
        super().__init__()
        self.res = spatial_resolution
        self.D = diffusion_coefficient
        self.gamma = fwm_coupling
        
        # The Ephaptic spatial coupling kernel (2D Laplacian Stencil)
        # Models the physical bleeding of the electric field to immediate neighbors
        self.register_buffer('laplacian_stencil', torch.tensor([
            [0.0,  1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0,  1.0, 0.0]
        ]).view(1, 1, 3, 3))

    def forward(self, phase_grid: torch.Tensor, langevin_heat: torch.Tensor, dt: float = 0.01):
        """
        Integrates the partial differential equation for one time step dt.
        phase_grid shape: [Batch, 1, res, res]
        """
        # 1. Ephaptic Spatial Diffusion (D * \nabla^2 \psi)
        # This requires physical, contiguous proximity.
        ephaptic_diffusion = F.conv2d(
            phase_grid, 
            self.laplacian_stencil, 
            padding=1
        ) * self.D

        # 2. Global Mean-Field Approximation of Four-Wave Mixing (FWM)
        # Represents the non-linear Kerr effect interaction across the crystal cavity
        mean_phase = torch.mean(phase_grid, dim=[-1, -2], keepdim=True)
        fwm_interaction = self.gamma * torch.sin(mean_phase - phase_grid)

        # 3. Langevin Thermodynamic Noise (\sqrt{2T} * \xi)
        # Scaled by the DivergentMaster's active heat parameter
        noise_amplitude = torch.sqrt(2.0 * langevin_heat.view(-1, 1, 1, 1))
        thermal_variance = noise_amplitude * torch.randn_like(phase_grid)

        # 4. Eulerian Integration Update
        d_phase_dt = ephaptic_diffusion + fwm_interaction + thermal_variance
        updated_phase_grid = phase_grid + (d_phase_dt * dt)

        # 5. Extract the Macroscopic Order Parameter (The Bridge to Tier 2/3)
        # R_B * e^{i \Phi_B} = \frac{1}{V} \int e^{i \psi} dV
        complex_grid = torch.complex(torch.cos(updated_phase_grid), torch.sin(updated_phase_grid))
        macro_complex = torch.mean(complex_grid, dim=[-1, -2])
        
        R_B = torch.abs(macro_complex)      # Sagnac Coherence (Amplitude)
        Phi_B = torch.angle(macro_complex)  # Global Crystal Phase
        
        return updated_phase_grid, R_B, Phi_B

class HybridSynchronizationGrid(nn.Module):
    """
    Tier 2 & 3: The Kuramoto Macro-Architecture.
    Governs the topological CXL bus connections between the discrete digital 
    components (Zone A/C) and the continuous optical substrate (Zone B).
    """
    def __init__(self, spatial_resolution=256, k_coupling=2.0):
        super().__init__()
        self.ephaptic_core = EphapticCrystalContinuum(spatial_resolution=spatial_resolution)
        self.k_coupling = k_coupling
        
        # Intrinsic frequencies of the digital zones
        self.register_parameter('omega_A', nn.Parameter(torch.randn(1)))
        self.register_parameter('omega_C', nn.Parameter(torch.randn(1)))

    def apply_divergent_master(self, R_B: torch.Tensor, threshold: float = 0.85) -> torch.Tensor:
        """
        The Thermodynamic Thermostat. 
        If the Ephaptic order parameter R_B fractures (drops below 0.85), 
        massive Langevin heat is injected to melt the logic lock.
        """
        # Heat scales inversely with structural resonance
        heat = 1.0 - R_B
        # Apply strict geometric cutoff
        return torch.where(R_B < threshold, heat * 5.0, torch.tensor(0.01, device=R_B.device))

    def forward(self, phase_grid_B: torch.Tensor, Phi_A: torch.Tensor, Phi_C: torch.Tensor):
        batch_size = phase_grid_B.size(0)
        
        # --- 1. THE MICRO-PHYSICS (Ephaptic Integration) ---
        # Assess current coherence to set the heat before integration
        complex_grid_prior = torch.complex(torch.cos(phase_grid_B), torch.sin(phase_grid_B))
        R_B_prior = torch.abs(torch.mean(complex_grid_prior, dim=[-1, -2])).view(-1)
        langevin_heat = self.apply_divergent_master(R_B_prior)

        # Advance the optical continuum
        updated_grid_B, R_B_post, Phi_B = self.ephaptic_core(phase_grid_B, langevin_heat)

        # --- 2. THE MACRO-PHYSICS (Kuramoto Interconnect Integration) ---
        # Note: Zone A and Zone C are coupled to Zone B proportional to R_B.
        # If Zone B is a chaotic mess of noise (R_B = 0), the digital zones detach
        # to preserve their own memory invariants.
        
        dPhi_A = self.omega_A + self.k_coupling * R_B_post * torch.sin(Phi_B - Phi_A) + \
                 self.k_coupling * torch.sin(Phi_C - Phi_A)
                 
        dPhi_C = self.omega_C + self.k_coupling * R_B_post * torch.sin(Phi_B - Phi_C) + \
                 self.k_coupling * torch.sin(Phi_A - Phi_C)

        updated_Phi_A = Phi_A + (dPhi_A * 0.01)
        updated_Phi_C = Phi_C + (dPhi_C * 0.01)

        return updated_grid_B, updated_Phi_A, updated_Phi_C, R_B_post, langevin_heat