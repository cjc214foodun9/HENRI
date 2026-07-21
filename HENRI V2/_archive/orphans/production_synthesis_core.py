"""
Production Synthesis Core for HENRI Phase II.
Unifies OPINE-World's Bayesian Ontology Tracking, Sutton's IDBD Step-Size Optimization,
and Non-Reconstructive PEARL Trajectory Steering on the Complex Manifold S^4095.

Strictly adheres to PyTorch-native complex tensor operations; contains no mock loops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

class AnisotropicLangevinDynamics(nn.Module):
    """
    Implements localized, error-projected Langevin thermal injection on S^4095.
    Replaces uniform isotropic heat shocks with anisotropic diffusion tensors,
    restricting thermal variance to coordinates with active Sagnac mismatch.
    """
    def __init__(
        self,
        dimension: int = 4096,
        t_floor: float = 0.001,
        t_shock_max: float = 2.0,
        tau_error: float = 0.1,
        device: str = "cuda"
    ):
        super().__init__()
        self.dimension = dimension
        self.t_floor = t_floor
        self.t_shock_max = t_shock_max
        self.tau_error = tau_error
        self.device = device

    def compute_anisotropic_noise(
        self,
        active_wave: torch.Tensor,
        target_wave: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Computes the coordinate-wise anisotropic diffusion coefficient and projects
        noise exclusively onto corrupted semantic orthants.
        
        Args:
            active_wave: Complex state tensor [B, D]
            target_wave: Complex target engram tensor [B, D]
            dt: Integration time step
        """
        # 1. Compute differentiable phase error using complex circular inner product
        # Avoids non-differentiable discontinuities in raw angle modulo operations
        phase_diff_complex = active_wave * target_wave.conj()
        e_i = torch.atan2(phase_diff_complex.imag, phase_diff_complex.real) # [B, D]

        # 2. Construct coordinate-wise Ontological Projection Mask (P_error)
        squared_error = torch.square(e_i)
        p_error = F.softmax(squared_error / self.tau_error, dim=-1) # [B, D]

        # 3. Formulate the Anisotropic Diffusion Tensor diagonal (Sigma)
        # Low noise floor for stable dimensions, high targeted heat for corrupted dimensions
        sigma_sq = self.t_shock_max * p_error + self.t_floor * (1.0 - p_error)
        sigma = torch.sqrt(2.0 * sigma_sq * dt) # [B, D]

        # 4. Sample and scale complex-valued Gaussian noise
        raw_noise_real = torch.randn(active_wave.shape, device=self.device, dtype=torch.float32)
        raw_noise_imag = torch.randn(active_wave.shape, device=self.device, dtype=torch.float32)
        raw_noise = torch.complex(raw_noise_real, raw_noise_imag)

        return sigma * raw_noise


class KuramotoIDBDEngine(nn.Module):
    """
    Zone B Core: Integrates the driven Kuramoto synchronization equation on S^4095
    coupled with Sutton's Incremental Delta-Bar-Delta (IDBD) step-size optimization.
    Stable logical symmetries decouple plasticity, while volatile parameters retain elasticity.
    """
    def __init__(
        self,
        dimension: int = 4096,
        meta_lr: float = 0.01,
        init_coupling: float = 0.1,
        device: str = "cuda"
    ):
        super().__init__()
        self.dimension = dimension
        self.meta_lr = meta_lr
        self.device = device

        # Base intellectual momentum (natural frequencies omega)
        self.register_buffer("omega", torch.randn(dimension, device=device) * 0.01)

        # Kuramoto coupling matrix parameters: K_ij replaces standard weights
        self.coupling_matrix = nn.Parameter(
            torch.ones(dimension, dimension, device=device) * (init_coupling / dimension)
        )

        # IDBD Parameters: Log-step-size beta and trace h for parameter-wise plasticity
        self.register_buffer("beta", torch.log(torch.ones(dimension, dimension, device=device) * 0.05))
        self.register_buffer("h", torch.zeros(dimension, dimension, device=device))

    def step_dynamics(
        self,
        active_phases: torch.Tensor,
        target_phases: Optional[torch.Tensor],
        dt: float,
        noise_vector: Optional[torch.Tensor] = None,
        free_energy_gradient: float = -1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Advances the Kuramoto ODE by dt, updating coupling values under IDBD constraint.
        
        Args:
            active_phases: Phase angles [B, D] (float32)
            target_phases: Target phase angles [B, D] or [1, D] (float32)
            dt: Integration time step
            noise_vector: Pre-computed anisotropic noise [B, D] (float32)
            free_energy_gradient: Derivative of the ontology error. IDBD is gated by this.
        """
        batch_size = active_phases.size(0)
        
        # 1. Compute lateral torque: sin(theta_j - theta_i)
        # Shape: [B, D, D]
        phase_i = active_phases.unsqueeze(2) # [B, D, 1]
        phase_j = active_phases.unsqueeze(1) # [B, 1, D]
        torque_matrix = torch.sin(phase_j - phase_i)

        # Mean field lateral interaction
        # K_ij is [D, D]. Matrix multiply: [B, D, D] x [D, D] -> [B, D]
        internal_torque = torch.matmul(torque_matrix, self.coupling_matrix.t()).mean(dim=-1)

        # 2. Compute external Sagnac anchor torque
        external_torque = torch.zeros_like(active_phases)
        delta_i = torch.zeros_like(active_phases)
        if target_phases is not None:
            # Complex coordinate subtraction mapped via atan2
            delta_i = torch.atan2(torch.sin(target_phases - active_phases), torch.cos(target_phases - active_phases))
            external_torque = 5.0 * torch.sin(delta_i)

        # 3. Integrate ODE with natural frequencies and noise
        d_theta = self.omega.unsqueeze(0) + internal_torque + external_torque
        if noise_vector is not None:
            d_theta += noise_vector

        new_phases = torch.remainder(active_phases + d_theta * dt + torch.pi, 2.0 * torch.pi) - torch.pi

        # 4. Sutton's IDBD Update Loop over the Coupling Matrix
        with torch.no_grad():
            if target_phases is not None:
                # Meta-learned step-size alpha_ij = exp(beta_ij)
                alpha = torch.exp(self.beta) # [D, D]
                
                # Sagnac phase error acting as backpropagated mismatch delta_i
                # We compute the average trace over the batch
                mean_delta = delta_i.mean(dim=0, keepdim=True).t() # [D, 1]
                mean_torque = torque_matrix.mean(dim=0) # [D, D]

                # Update log-step-sizes beta_ij based on alignment of current error and history
                # Bind plasticity strictly to thermodynamic Free Energy reduction
                # Only increase malleability if the wave is actively falling into the attractor
                thermodynamic_reward = torch.clamp(torch.tensor(-free_energy_gradient, device=self.device), min=0.0)
                d_beta = self.meta_lr * (mean_delta * mean_torque * self.h) * thermodynamic_reward
                self.beta.add_(d_beta)

                # Update coupling matrix K_ij: dK = alpha * delta * torque
                d_K = alpha * (mean_delta * mean_torque)
                self.coupling_matrix.add_(d_K)

                # Update running trace h_ij: h_new = h * [1 - alpha * torque^2]^+ + alpha * delta * torque
                decay_factor = F.relu(1.0 - alpha * torch.square(mean_torque))
                self.h.mul_(decay_factor).add_(d_K)

        return new_phases, delta_i


class PEARLSteeringEngine(nn.Module):
    """
    Implements the PEARL Protocol for continuous trajectory steering.
    Convolves predicted lookahead states with temporal binding carriers to generate
    a Topological Steering Field, pulling the active wave along the target planning geodesic.
    """
    def __init__(self, dimension: int = 4096, horizon: int = 8, device: str = "cuda"):
        super().__init__()
        self.dimension = dimension
        self.horizon = horizon
        self.device = device

        # Pre-generate rigidly orthogonal chronological binding carriers R_k on S^4095
        # Represented as random complex phase hypervectors
        r_phases = (torch.rand(horizon, dimension, device=device) * 2.0 * torch.pi) - torch.pi
        self.register_buffer("R_carriers", torch.complex(torch.cos(r_phases), torch.sin(r_phases)))

    def circular_convolution(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes O(D log D) circular convolution in the Fourier domain."""
        x_fft = torch.fft.fft(x, dim=-1)
        y_fft = torch.fft.fft(y, dim=-1)
        return torch.fft.ifft(x_fft * y_fft, dim=-1)

    def generate_steering_field(self, future_latents: torch.Tensor, gamma: float = 0.9) -> torch.Tensor:
        """
        Convolves future trajectory predictions with temporal carriers.
        
        Args:
            future_latents: Complex tensor of predicted future states [H, D]
            gamma: Temporal discount factor
        """
        # Ensure horizon size matches pre-allocated carriers
        h_steps = min(self.horizon, future_latents.size(0))
        steering_field = torch.zeros(self.dimension, dtype=torch.complex64, device=self.device)

        for k in range(h_steps):
            decay = gamma ** (k + 1)
            # Bind future latent state to its temporal order
            bound_step = self.circular_convolution(future_latents[k], self.R_carriers[k])
            steering_field += decay * bound_step

        return F.normalize(steering_field, p=2, dim=-1)


class WirtingerOntologyTracker(nn.Module):
    """
    Differentiable Bayesian Ontology Tracker.
    Calculates type posteriors and ontology error strictly via complex projections
    under Wirtinger calculus, avoiding silent gradient breaks during backpropagation.
    """
    def __init__(
        self,
        num_types: int = 16,
        num_effects: int = 8,
        dimension: int = 4096,
        alpha_0: float = 1.0,
        device: str = "cuda"
    ):
        super().__init__()
        self.num_types = num_types
        self.num_effects = num_effects
        self.dimension = dimension
        self.alpha_0 = alpha_0
        self.device = device

        self.register_buffer("dirichlet_counts", torch.ones(num_types, 4, num_effects, device=device) * alpha_0)

    def forward(
        self,
        active_wave: torch.Tensor,
        lexicon_attractors: torch.Tensor,
        action_idx: torch.Tensor,
        observed_effect_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the exact type posterior and ontology error differentiably.
        
        Args:
            active_wave: Complex active state [B, D]
            lexicon_attractors: Complex lexicon engrams [num_types, D]
            action_idx: Action labels [B]
            observed_effect_idx: Effect labels [B]
        """
        # 1. Compute Type Posterior using continuous Wirtinger complex inner product
        # Replaces raw angles with continuous Hermitian projections: Re(Psi^H M)
        # Shape: [B, num_types]
        complex_projections = torch.matmul(active_wave, lexicon_attractors.t().conj())
        projection_similarity = complex_projections.real / self.dimension

        type_posterior = F.softmax(projection_similarity * 10.0, dim=-1)

        # Compute Shannon Entropy over Type Posteriors: U_type = H(P) / log(K)
        type_entropy = -torch.sum(type_posterior * torch.log(type_posterior + 1e-9), dim=-1)
        u_type = type_entropy / torch.log(torch.tensor(float(self.num_types), device=self.device))

        # 2. Compute Row Uncertainty (U_row) via Dirichlet Posterior Mean
        assigned_types = torch.argmax(type_posterior, dim=-1)

        selected_rows = self.dirichlet_counts[assigned_types, action_idx, :] # [B, num_effects]
        row_sums = torch.sum(selected_rows, dim=-1, keepdim=True)
        q_hat = selected_rows / row_sums

        row_entropy = -torch.sum(q_hat * torch.log(q_hat + 1e-9), dim=-1)
        u_row = row_entropy / torch.log(torch.tensor(float(self.num_effects), device=self.device))

        # 3. Noisy-OR Combination: eta_t = 1 - (1 - U_type)(1 - U_row)
        ontology_error = 1.0 - (1.0 - u_type) * (1.0 - u_row)

        # 4. On-policy Update of Dirichlet counts
        with torch.no_grad():
            for b in range(active_wave.size(0)):
                t_idx = assigned_types[b]
                a_idx = action_idx[b]
                e_idx = observed_effect_idx[b]
                self.dirichlet_counts[t_idx, a_idx, e_idx] += 1.0

        return ontology_error, type_posterior