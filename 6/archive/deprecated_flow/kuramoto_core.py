# Native Intrinsic Coupled Oscillator Substrate for Project HENRI
# Replaces 6/dynamic_gear_shifter.py with continuous rhythmic learning

import torch  
import torch.nn as nn  
import torch.nn.functional as F

class HenriKuramotoCore(nn.Module):  
    def __init__(self, num_oscillators=4096, num_experts=16):  
        super().__init__()  
        self.N = num_oscillators  
        self.num_experts = num_experts  
          
        # Intrinsic natural frequencies (omega) - learned momentum of thoughts  
        self.omega = nn.Parameter(torch.randn(num_oscillators) * 0.1)  
          
        # Learnable Coupling Strengths (K_ij) initialized near pairwise orthogonality  
        # Completely replaces standard dense linear transformer layer weights  
        K_init = torch.randn(num_oscillators, num_oscillators)  
        torch.nn.init.orthogonal_(K_init)  
        self.K = nn.Parameter(K_init)  
          
        # Low-rank Category conditioning matrix (inspired by Un-0 architecture)  
        # Maps contextual expert conditioning vectors directly into the phase plane  
        self.conditioning_router = nn.Linear(num_experts, num_oscillators, bias=False)  
          
    def forward(self, initial_hrr_wave, expert_conditioning, timesteps=100, dt=0.01, current_temp=0.0):  
        """  
        initial_hrr_wave: [Batch, 4096] complex-valued tensor on S^4095  
        expert_conditioning: [Batch, 16] continuous fluid expert activations  
        """  
        batch_size = initial_hrr_wave.size(0)  
        device = initial_hrr_wave.device  
          
        # Extract initial phases from the continuous complex-valued HRR input wavefront  
        # Shape: [Batch, 4096]  
        theta = torch.angle(initial_hrr_wave) # Shape: [Batch, 4096]  
          
        # Inject low-rank context bias from the active 16 fluid expert streams  
        bias = self.conditioning_router(expert_conditioning) # Shape: [Batch, 4096]  
          
        # Run continuous-time physical integration loop (Euler-Maruyama)  
        for _ in range(timesteps):  
            # Compute pairwise phase differences: theta_j - theta_i  
            # [Batch, 4096, 1] - [Batch, 1, 4096] -> [Batch, 4096, 4096]  
            phase_diff = theta.unsqueeze(2) - theta.unsqueeze(1)  
              
            # Calculate the total coupling pull: (1/N) * sum( K_ij * sin(theta_j - theta_i) )  
            # Element-wise multiply the learned coupling strengths by the phase delta sine fields  
            coupling_force = torch.matmul(torch.sin(phase_diff), self.K.unsqueeze(0).transpose(1, 2))  
            coupling_pull = coupling_force.diagonal(dim1=-2, dim2=-1) / self.N  
              
            # Stochastic out-of-band Langevin heat perturbation from the Divergent Master  
            langevin_noise = torch.randn_like(theta) * torch.sqrt(torch.tensor(2.0 * current_temp * dt, device=device))  
              
            # Apply the definitive Kuramoto state transition rule smoothly  
            d_theta = (self.omega.unsqueeze(0) + coupling_pull + bias) * dt + langevin_noise  
            theta = (theta + d_theta) % (2 * torch.pi)  
              
        # Re-constitute the final settled phase profile into a pristine unit complex wavefront  
        settled_wavefront = torch.complex(torch.cos(theta), torch.sin(theta))  
          
        # Enforce strict global wavefront modulus preservation to stabilize deep lookahead paths  
        settled_wavefront = F.normalize(settled_wavefront, p=2, dim=-1)  
          
        # Compute macro Phase Synchronization Order Parameter (R) as continuous system telemetry  
        # Real-time R tracking acts as the internal elastic resource governor  
        macro_order_parameter = torch.abs(settled_wavefront.sum(dim=-1)) / self.N  
          
        return settled_wavefront, macro_order_parameter
