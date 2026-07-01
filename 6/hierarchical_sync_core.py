# Unified Three-Tier Nested Kuramoto Synchronization Grid for Project HENRI

import math
import torch  
import torch.nn as nn  
import torch.nn.functional as F


class HenriHierarchicalSyncCore(nn.Module):  
    def __init__(self, num_oscillators=4096, num_layers=32, num_experts=16):  
        super().__init__()  
        self.N = num_oscillators  
        self.L = num_layers  
        self.E = num_experts  
          
        # TIER 1: Micro-Oscillator Parameters (Inner Fluid Expert Channels)  
        # Learnable Coupling Strengths K_ij initialized with strict orthogonal priors  
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        K_init = torch.randn(num_layers, num_oscillators, num_oscillators, device=dev, dtype=torch.float32)  
        for l in range(num_layers):  
            torch.nn.init.orthogonal_(K_init[l])  
        self.K_micro = nn.Parameter(K_init.to(dtype=torch.bfloat16))  
          
        # TIER 2: Meso-Oscillator Parameters (Layer-to-Layer Depth Progression)  
        # Base natural frequencies across the 32 diffractive layers  
        self.omega_base = nn.Parameter(torch.randn(num_layers, num_oscillators) * 0.05)  
        self.alpha_meso = nn.Parameter(torch.tensor(0.1)) # Inter-layer torque coefficient  
          
        # TIER 3: Macro-Oscillator Parameters (Cross-Zone Interconnect)  
        # Coordinates tracking global clock states for Zone A, Zone B, and Zone C  
        self.zone_clocks = nn.Parameter(torch.tensor([0.0, 2.0 * torch.pi / 3.0, 4.0 * torch.pi / 3.0]))  
        self.K_macro = nn.Parameter(torch.tensor(0.25)) # Zone-to-Zone coupling rigidity  
          
        # Low-rank context adapter to fuse 16 active fluid experts directly into phase plane  
        self.fluid_context_router = nn.Linear(num_experts, num_oscillators, bias=False)

    def forward(self, hrr_wavefront, expert_activations, timesteps=50, dt=0.02, temperature=0.0, nudge_context=None):  
        """  
        hrr_wavefront: [Batch, 4096] complex-valued tensor or [Batch, 4096, 2] unrolled on the S^4095 hypersphere  
        expert_activations: [Batch, 16] continuous activations from the MoE swarm  
        nudge_context: [Batch, 4096] phase nudge pull vector (for Equilibrium Propagation)
        """  
        batch_size = hrr_wavefront.size(0)  
        device = hrr_wavefront.device  
          
        # Extract initial phase angles from the real-and-imaginary unrolled or complex input wavefront  
        # Shape: [Batch, 4096]  
        if torch.is_complex(hrr_wavefront):
            theta_current = torch.angle(hrr_wavefront)
        elif hrr_wavefront.ndim == 3 and hrr_wavefront.size(-1) == 2:
            theta_current = torch.atan2(hrr_wavefront[..., 1], hrr_wavefront[..., 0])
        else:
            theta_current = hrr_wavefront # Assume already in phase format
            
        # Project active fluid expert features into context-biasing phase offsets  
        context_bias = self.fluid_context_router(expert_activations) # [Batch, 4096]  
          
        # Track macro order parameters across the deep diffractive layers  
        layer_order_parameters = []  
          
        # Reset the macro zone clock states for this execution sequence iteration  
        zone_phases = self.zone_clocks.clone().unsqueeze(0).repeat(batch_size, 1) # [Batch, 3]  
          
        # =========================================================================  
        # CONTINUOUS INTEGRATION LOOP (Nested Multi-Scale Relaxation)  
        # =========================================================================  
        for l in range(self.L):  
            # Fetch natural driving frequencies for the current diffractive layer depth  
            omega_l = self.omega_base[l].unsqueeze(0) # [1, 4096]  
              
            # Fetch the Tier 1 coupling parameter slice for this specific layer  
            K_l = self.K_micro[l] # [4096, 4096]  
              
            for t in range(timesteps):  
                # -----------------------------------------------------------------  
                # TIER 1: Micro-Oscillator Phase Relaxation  
                # -----------------------------------------------------------------  
                # Compute coupling pull using trigonometric identity factorization:
                # sin(theta_i - theta_j) = sin(theta_i)cos(theta_j) - cos(theta_i)sin(theta_j)
                # This drops O(N^2) memory footprint to O(N) and executes in microseconds!
                S = torch.sin(theta_current)
                C = torch.cos(theta_current)
                K_sin = torch.matmul(S, K_l.t())
                K_cos = torch.matmul(C, K_l.t())
                coupling_pull = (S * K_cos - C * K_sin) / self.N
                  
                # Inject stochastic Langevin thermal kinetics from the Divergent Master  
                langevin_noise = torch.randn_like(theta_current) * torch.sqrt(torch.tensor(2.0 * temperature * dt, device=device))  
                  
                # Update micro phases  
                if nudge_context is not None:
                    d_theta = (omega_l + coupling_pull + context_bias + nudge_context) * dt + langevin_noise
                else:
                    d_theta = (omega_l + coupling_pull + context_bias) * dt + langevin_noise
                theta_current = (theta_current + d_theta) % (2 * torch.pi)  
                  
                # -----------------------------------------------------------------  
                # TIER 3: Macro-Oscillator Cross-Zone Interconnect Tracking  
                # -----------------------------------------------------------------  
                # Re-calculate the instantaneous global macro order parameter (R)  
                temp_complex = torch.complex(torch.cos(theta_current).float(), torch.sin(theta_current).float())  
                R_macro = torch.abs(temp_complex.sum(dim=-1)) / self.N # [Batch]  
                  
                # Phase delta between Ingress (Zone A) and current Core state (Zone B)  
                # Zone phases: index 0 = Zone A, index 1 = Zone B, index 2 = Zone C  
                ingress_phase = zone_phases[:, 0]  
                core_phase = zone_phases[:, 1]  
                  
                # High phase dispersion (R_macro -> 0) amplifies zone-to-zone synchronization torque  
                zone_torque = self.K_macro * (1.0 - R_macro) * torch.sin(ingress_phase - core_phase)  
                  
                # Update macro zone phase boundaries  
                zone_phases[:, 1] = (zone_phases[:, 1] + zone_torque * dt) % (2 * torch.pi)  
                  
            # ---------------------------------------------------------------------  
            # TIER 2: Meso-Oscillator Layer Depth Progression Handoff  
            # ---------------------------------------------------------------------  
            # Capture the finalized layer macro order score for telemetry logging  
            layer_order_parameters.append(R_macro.mean().item())  
              
            # Executing the continuous layer handoff:  
            # If the depth step is not terminal, the settled phase of layer l modulates  
            # the natural frequency landscape (omega) of layer l+1 to enforce NextLat paths  
            if l < self.L - 1:  
                wrapped_theta = torch.angle(torch.complex(torch.cos(theta_current).float(), torch.sin(theta_current).float()))
                phase_torque = self.alpha_meso * torch.sin(theta_current - wrapped_theta)
                self.omega_base.data[l+1] += phase_torque.mean(dim=0) * dt

        # Re-constitute the settled phase footprint back into a normalized complex wave  
        settled_phases = theta_current  
        synchronized_wavefront = torch.complex(torch.cos(settled_phases).float(), torch.sin(settled_phases).float())  
        synchronized_wavefront = F.normalize(synchronized_wavefront, p=2, dim=-1)  
          
        return sorted_phases_to_wavefront(synchronized_wavefront) if hasattr(self, 'sorted_phases_to_wavefront') else synchronized_wavefront, layer_order_parameters, zone_phases

def sorted_phases_to_wavefront(synchronized_wavefront):
    return synchronized_wavefront

class HenriThermodynamicFPRM(nn.Module):
    def __init__(self, d_wave=4096, num_experts=16):
        super().__init__()
        self.d_wave = d_wave
        self.num_experts = num_experts
        self.K_shared = nn.Parameter(torch.randn(d_wave, d_wave) * 0.02)

    def execute_ptrm_relaxation_step(self, theta: torch.Tensor, freq_weights: torch.Tensor, step_idx: int):
        """
        Implements Probabilistic Tiny Recursive Model dynamics in the phase domain.
        Proactively injects step-wise Langevin variance to map alternative trajectories.
        """
        # Calculate localized coupling torque across the frequency teeth
        coupling_torque = torch.matmul(torch.sin(theta), freq_weights)
        
        # Compute the instantaneous global phase-locking order parameter (R)
        # High phase dispersion (low R) scales the thermal noise organically
        r_norm = torch.abs(torch.exp(1j * theta).mean(dim=-1, keepdim=True))
        dynamic_temperature = 1.8 * (1.0 - r_norm)
        
        # Inject noise directly into the phase angle domain, preserving the unit modulus
        phase_langevin_noise = torch.randn_like(theta) * dynamic_temperature
        
        # Step-wise state update wrapped cleanly modulo 2*pi
        theta_next = (theta + 0.15 * coupling_torque + phase_langevin_noise) % (2 * math.pi)
        return theta_next
        
    def forward(self, initial_wave: torch.Tensor, max_loops: int = 24, fp_thresh: float = 0.996):
        """
        initial_wave shape: [Batch, num_experts, d_wave] complex phasors
        """
        batch_size = initial_wave.size(0)
        theta = torch.angle(initial_wave) # Extract clean phase [B, E, d_wave]
        
        for step in range(max_loops):
            old_theta = theta.clone()
            
            # Step-wise state update wrapped cleanly modulo 2*pi
            theta = self.execute_ptrm_relaxation_step(theta, self.K_shared, step)
            
            # Fixed-Point Convergence Halting Verification
            with torch.no_grad():
                delta_alignment = F.cosine_similarity(torch.sin(theta), torch.sin(old_theta), dim=-1).mean()
                if delta_alignment > fp_thresh:
                    # Avoid print statement to prevent log spam during distillation
                    break
                    
        return torch.complex(torch.cos(theta), torch.sin(theta))

class HenriParametricTokenHead(nn.Module):
    def __init__(self, d_wave=4096, vocab_size=256):
        super().__init__()
        # Decoupled projection paths to bridge the binary keyhole bottleneck
        self.op_decoder = nn.Linear(1024, vocab_size, bias=False)      # Op-Code Subspace
        self.color_decoder = nn.Linear(1024, 10, bias=False)          # Identity Subspace
        self.spatial_decoder = nn.Sequential(                         # Spatial Parameter Subspace
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Linear(512, 30) # Maps to maximum 30x30 ARC matrix index constraints
        )
        
    def forward(self, settled_wave: torch.Tensor):
        wave_angles = torch.angle(settled_wave)
        
        # Lossless fiber breakout of the hypervector channels
        psi_op = wave_angles[..., 0:1024]
        psi_id = wave_angles[..., 1024:2048]
        psi_geo = wave_angles[..., 2048:4096]
        
        return {
            "op_logits": self.op_decoder(psi_op),
            "color_logits": self.color_decoder(psi_id),
            "param_logits": self.spatial_decoder(psi_geo)
        }

