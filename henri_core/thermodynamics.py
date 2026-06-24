import torch
import torch.nn as nn
import torch.nn.functional as F

class NaturalInductionLoss(nn.Module):
    """
    Computes the Thermodynamic Free Energy of the 4096-D HRR wave trajectory.
    Replaces Cross-Entropy Loss. Includes the Statistical Structural Regularizer R(X)
    from the Olivera et al. (2026) paper to maintain VSA representation statistics.
    """
    def __init__(self, lambda_boundary=10.0, reg_coefficient=1.0, dim=4096):
        super().__init__()
        self.lambda_boundary = lambda_boundary
        self.reg_coefficient = reg_coefficient
        self.dim = dim

    def statistical_regularizer(self, X: torch.Tensor, tau_n=1.0) -> torch.Tensor:
        """
        R(X) from Olivera et al. (2026).
        Ensures latent states have zero mean, target norm (tau_n = 1.0), and target component variance (1 / d).
        """
        # Shape: (Batch, Depth, Dim)
        flat_X = X.reshape(-1, X.shape[-1]) # (N, Dim)
        tau_v = tau_n / self.dim
        
        # 1. Norm Constraint
        norms_sq = torch.sum(flat_X**2, dim=-1)
        norm_loss = torch.mean((norms_sq - tau_n)**2)
        
        # 2. Mean Constraint
        mean_val = torch.mean(flat_X)
        mean_loss = mean_val**2
        
        # 3. Variance Constraint
        var_val = torch.var(flat_X, dim=0).mean()
        var_loss = (var_val - tau_v)**2
        
        return norm_loss + mean_loss + var_loss

    def forward(self, wave_trajectory: torch.Tensor, zone_c_attractors: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        wave_trajectory: Layer-by-layer states of the wave (Batch, Depth, Dim)
        zone_c_attractors: Target HRR topology from TimescaleDB (Batch, Dim)
        temperature: Scalar Langevin heat injected
        """
        # Normalize to conserve energy on the hypersphere
        wave_trajectory = F.normalize(wave_trajectory, p=2, dim=-1)
        zone_c_attractors = F.normalize(zone_c_attractors, p=2, dim=-1)
        
        batch_size, depth, dim = wave_trajectory.shape

        # 1. Internal Propagation Stress (Gradient of the wave across network depth)
        wave_gradients = wave_trajectory[:, 1:, :] - wave_trajectory[:, :-1, :]
        internal_stress = 0.5 * torch.sum(wave_gradients**2) / (batch_size * depth)

        # 2. Boundary Resonance Penalty (Dirichlet Constraint at the output)
        final_wave_state = wave_trajectory[:, -1, :]
        resonance = torch.sum(final_wave_state * zone_c_attractors, dim=-1)
        boundary_penalty = self.lambda_boundary * torch.mean(1.0 - resonance)

        # 3. Langevin Noise Factor (Entropic Allowance)
        # Allows higher energy path fluctuations at higher temperatures
        entropic_allowance = temperature * torch.mean(torch.norm(wave_trajectory, p=2, dim=-1))

        # 4. Statistical Structural Regularizer R(X)
        r_x = self.statistical_regularizer(wave_trajectory)

        # 5. Total Free Energy (Thermodynamic Loss)
        total_free_energy = internal_stress + boundary_penalty - entropic_allowance + (self.reg_coefficient * r_x)

        return total_free_energy

import math

class AgentialLangevinThermostat(nn.Module):
    """
    Replaces the scripted IFTTT DivergentMaster. Gives HENRI agential control
    over Langevin noise injection by analyzing spectral resonance profiles mid-flight
    and generating precise, localized phase-angle perturbations.
    """
    def __init__(self, dim=4096, num_channels=256, hidden_dim=128):
        super().__init__()
        self.dim = dim
        self.num_channels = num_channels
        self.num_bands = dim // num_channels # 16 spatial expert bands
        
        # Learned network to predict noise bursts based on internal cognitive friction
        self.resonance_analyzer = nn.Sequential(
            nn.Linear(self.num_bands, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_bands * 2), # Outputs both Burst Timing (Gate) and Intensity (Sigma)
            nn.Tanh()
        )

    def calculate_agential_perturbation(self, active_wave_state, zone_c_lexicon):
        """
        Senses localized logic locks and decides exactly how much noise to inject 
        and when to burst it to prevent continuous decoherence.
        
        Args:
            active_wave_state: [Batch, 4096] (Current phase vector in Zn domain)
            zone_c_lexicon: [Num_Axioms, 4096] (Canonical reference baseplate)
        """
        device = active_wave_state.device
        dtype = active_wave_state.dtype
        batch_size = active_wave_state.size(0)

        # Step 1: Measure real-time spectral resonance across the 16 spatial bands
        # Reshape to isolate the 16 expert channels firing 256 spectral teeth each
        reshaped_wave = active_wave_state.reshape(batch_size, self.num_bands, self.num_channels)
        reshaped_lexicon = zone_c_lexicon.reshape(-1, self.num_bands, self.num_channels)
        
        # Calculate localized inner products to map channel-specific coherence
        # Shape: [Batch, Num_Bands]
        spectral_coherence = torch.mean(
            F.cosine_similarity(reshaped_wave.unsqueeze(1), reshaped_lexicon.unsqueeze(0), dim=-1),
            dim=1
        )

        # Step 2: Project coherence profiles through the learned thermostat head
        if spectral_coherence.dtype == torch.bfloat16:
            # Linear layer might expect float32 or bfloat16 depending on casting
            # Convert to self.resonance_analyzer[0].weight.dtype to ensure compatibility
            self.resonance_analyzer = self.resonance_analyzer.to(device=device, dtype=spectral_coherence.dtype)
        
        modulation_payload = self.resonance_analyzer(spectral_coherence) # [Batch, Num_Bands * 2]
        
        # Separate the raw decision gates (timing) from the continuous intensities (sigma)
        burst_gate, burst_intensity = torch.chunk(modulation_payload, 2, dim=-1)
        
        # Apply sigmoid activation to the gate to enforce a sharp, non-linear burst threshold
        # If the gate is low, no noise passes. If it snaps open, it releases a localized flash.
        activation_gate = torch.sigmoid(burst_gate * 10.0)
        
        # Map intensity bounded uniformly to safe operational hardware ranges [0, 3.5V]
        target_sigma = (torch.relu(burst_intensity) * 3.5) * activation_gate

        # Step 3: Materialize the localized phase-scrambling noise field
        # Expand the 16-band sigma targets back out to the full 4096 dimension
        expanded_sigma = target_sigma.repeat_interleave(self.num_channels, dim=-1)
        
        # Generate raw Gaussian noise coordinates matching the packed register state
        raw_noise = torch.randn_like(active_wave_state)
        
        # Synthesize the final agential phase-burst tensor
        # Noise is scaled natively by the model's active decision boundaries
        agential_noise_burst = raw_noise * expanded_sigma
        
        # Log the targeted intervention parameters for system telemetry
        mean_applied_voltage = float(target_sigma.mean().item())
        if mean_applied_voltage > 0.5:
            print(f"[COGNITIVE THERMOSTAT] Localized Logic Lock sensed. Injected {mean_applied_voltage:.4f}V Agential Phase-Burst.")

        return agential_noise_burst, mean_applied_voltage


class DivergentMaster:
    """
    Thermodynamic Controller for the 7B HENRI Core.
    Manages Langevin noise temperature based on Sagnac error telemetry, triggering shocks on logic locks.
    """
    def __init__(self, 
                 t_min=0.0, 
                 t_max=5.0, 
                 cooling_rate=0.05, 
                 heat_sensitivity=0.2, 
                 lock_threshold=1e-4, 
                 shock_multiplier=2.0,
                 stagnation_limit=5):
        
        self.t_min = t_min
        self.t_max = t_max
        self.alpha = cooling_rate
        self.beta = heat_sensitivity
        
        self.lock_threshold = lock_threshold
        self.shock_multiplier = shock_multiplier
        self.stagnation_counter = 0
        self.stagnation_limit = stagnation_limit
        
        self.current_T = t_max # Start hot to encourage exploration
        self.moving_avg_energy = None
        self.ema_decay = 0.9

    def step(self, current_free_energy: float) -> float:
        """
        Evaluates the physical stress and updates system temperature.
        Returns the new temperature T.
        """
        if self.moving_avg_energy is None:
            self.moving_avg_energy = current_free_energy
            return self.current_T

        # Sagnac Delta (deviation from exponential moving average baseline)
        sagnac_delta = current_free_energy - self.moving_avg_energy

        # Logic-Lock detection: high energy (arbitrarily > 0.05) and stagnated gradients
        energy_gradient = abs(sagnac_delta)
        shock_penalty = 0.0
        
        if energy_gradient < self.lock_threshold and current_free_energy > 0.05:
            self.stagnation_counter += 1
            if self.stagnation_counter >= self.stagnation_limit:
                print(f"[DIVERGENT MASTER] Logic Lock Detected. Injecting {self.shock_multiplier}V Thermal Shock.")
                shock_penalty = self.shock_multiplier
                self.stagnation_counter = 0
        else:
            self.stagnation_counter = 0

        # Thermodynamic update rule
        heat_injection = self.beta * max(0.0, sagnac_delta)
        passive_cooling = (1.0 - self.alpha) * self.current_T
        
        new_T = passive_cooling + heat_injection + shock_penalty
        self.current_T = max(self.t_min, min(self.t_max, new_T))

        # Update EMA energy baseline
        self.moving_avg_energy = (self.ema_decay * self.moving_avg_energy) + ((1.0 - self.ema_decay) * current_free_energy)

        return self.current_T

    def get_temperature(self) -> float:
        return self.current_T

