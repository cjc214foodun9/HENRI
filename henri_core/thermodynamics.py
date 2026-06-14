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
        internal_stress = 0.5 * torch.sum(torch.norm(wave_gradients, p=2, dim=-1)**2) / (batch_size * depth)

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
