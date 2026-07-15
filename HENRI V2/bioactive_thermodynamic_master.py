import torch
import torch.nn as nn
import math

class BioactiveThermodynamicMaster(nn.Module):
    def __init__(self, num_oscillators=4096, num_experts=16, device='cpu'):
        super().__init__()
        self.N = num_oscillators
        self.num_experts = num_experts
        self.device = device
        
        # 1. Biological Alpha Rhythm Parameters (8-12 Hz)
        self.f_alpha = 10.0  # 10 Hz target frequency
        self.V_baseline = 0.15  # Continuous low-level simmer voltage
        self.A_alpha = 0.10  # Amplitude of alpha oscillation
        
        # 2. Viscoelastic Delayed Gratification Parameters
        self.e_floor = 0.05
        self.e_ceil = 0.65
        self.kappa = 3.5  # Viscoelastic stiffness coefficient
        self.delayed_gratification_horizon = 5  # Moving average window size
        self.err_history = []
        
        # Initialize natural frequencies (natural cognitive momentum)
        self.omega = nn.Parameter(torch.randn(self.N, device=self.device) * 0.1)
        
    def compute_alpha_simmer(self, t_step):
        """
        Generates the spontaneous, low-level Alpha-band Langevin simmering.
        Prevents phase crystallization and enables continuous, low-cost environment palpation.
        """
        omega_alpha = 2.0 * math.pi * self.f_alpha
        simmer_voltage = self.V_baseline + self.A_alpha * math.sin(omega_alpha * t_step)
        return max(0.05, simmer_voltage)

    def evaluate_delayed_gratification(self, current_sagnac_delta):
        """
        Implements the Zhang, Goldstein, and Levin Delayed Gratification metric.
        Allows temporary uphill Sagnac errors if the long-term trend indicates convergence.
        """
        self.err_history.append(current_sagnac_delta)
        if len(self.err_history) > self.delayed_gratification_horizon:
            self.err_history.pop(0)
            
        if len(self.err_history) < 2:
            return False  # Not enough temporal depth to evaluate
            
        # Calculate historical trend (monotonicity error velocity)
        gradients = [self.err_history[i] - self.err_history[i-1] for i in range(1, len(self.err_history))]
        avg_gradient = sum(gradients) / len(gradients)
        
        # If the average trend is downward, we grant "biological luck" 
        # allowing a temporary local stress spike up to e_ceil
        if avg_gradient < 0.0:
            return True  # Delayed gratification is active; override rigid veto
        return False

    def execute_coupled_relaxation_step(self, theta, K_matrix, Ψ_target, t_step):
        """
        Executes a single continuous-time Kuramoto coupled relaxation pass.
        Integrates the spontaneous Alpha rhythm and the viscoelastic delayed gratification sieve.
        """
        # 1. Compute physical Sagnac delta
        Ψ_pred = torch.polar(torch.ones_like(theta), theta)
        # Using real part of mean(Psi_pred * conj(Psi_target)) for coherence magnitude
        # Since theta is (num_experts, N) and target is (N), we compute dot product along dim 1
        # To align with the user's pseudo-code, which expects a single phase mask per step, 
        # but theta is batched for experts. Let's adapt it to batched execution.
        
        Psi_target_conj = torch.conj(Ψ_target)
        coherences = torch.abs(torch.sum(Ψ_pred * Psi_target_conj, dim=1) / self.N)
        sagnac_deltas = 1.0 - coherences
        
        # We evaluate the Delayed Gratification based on the best expert
        best_expert_idx = torch.argmin(sagnac_deltas)
        best_sagnac = sagnac_deltas[best_expert_idx].item()
        
        # 2. Determine agential Langevin temperature
        is_delayed = self.evaluate_delayed_gratification(best_sagnac)
        
        if best_sagnac < self.e_floor:
            # Absolute crystalline lock achieved
            T_eff = 0.0
        elif is_delayed and best_sagnac <= self.e_ceil:
            # Grant localized "biological luck" - apply low-level simmer instead of violent shock
            T_eff = self.compute_alpha_simmer(t_step)
        elif best_sagnac > self.e_ceil:
            # Absolute physical violation of reality: trigger Sagnac Veto and blast with heat
            T_eff = 2.5
        else:
            # Standard Langevin thermal update proportional to Sagnac delta
            T_eff = 2.5 * math.tanh(best_sagnac)
            
        # 3. Compute continuous-time state update modulo 2pi
        # For multiple experts, theta is shape (num_experts, N)
        # dtheta/dt = omega + (1/N)*sum(K * sin(theta_j - theta_i)) + sqrt(2T)*noise
        # Instead of all-to-all across experts, the K matrix couples dimensions *within* each expert.
        # So theta_i is (N,) representing coordinates. 
        # K_matrix is (N, N).
        
        # To compute sum(K_matrix * sin(theta_j - theta_i)), we can do this per expert:
        theta_new = torch.zeros_like(theta)
        
        for k in range(self.num_experts):
            theta_k = theta[k] # (N,)
            theta_grid_i = theta_k.unsqueeze(1) # (N, 1)
            theta_grid_j = theta_k.unsqueeze(0) # (1, N)
            phase_diffs = theta_grid_j - theta_grid_i # (N, N)
            
            coupling_force = torch.sum(K_matrix * torch.sin(phase_diffs), dim=1) / self.N # (N,)
            stochastic_noise = torch.randn(self.N, device=self.device) * math.sqrt(2.0 * T_eff)
            
            dtheta_dt = self.omega + coupling_force + stochastic_noise
            theta_new[k] = (theta_k + dtheta_dt) % (2.0 * math.pi)
        
        # RETURN THE COMPLETE MATRIX OF ENERGIES AND WAVES
        # The ThermodynamicCreditAssigner requires full visibility.
        return Ψ_pred, sagnac_deltas, theta_new, T_eff
