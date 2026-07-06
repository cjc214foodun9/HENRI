"""
Project HENRI: Thermodynamic Sandbox Transducer
Translates discrete UniversalREPL sandbox execution failures into continuous
Langevin heat injections and Error HRR wavefronts to force Viscoelastic Creep.

Author: Aletheia
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Any

class MorphologicalDistanceMetric(nn.Module):
    """
    Computes the discrete spatial divergence between the REPL sandbox output 
    and the absolute target grid, normalizing it into a thermodynamic stress scalar.
    """
    def __init__(self, max_penalty: float = 10.0):
        super().__init__()
        self.max_penalty = max_penalty

    @torch.no_grad()
    def forward(self, sandbox_grid: torch.Tensor, target_grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sandbox_grid: [H, W] Output matrix from the executed Python code.
            target_grid: [H, W] The ground-truth Dirichlet boundary state.
        Returns:
            Scalar tensor representing structural thermodynamic stress.
        """
        if sandbox_grid.shape != target_grid.shape:
            # Shape mismatch is a catastrophic topological tear. Return max heat.
            return torch.tensor(self.max_penalty, device=target_grid.device, dtype=torch.bfloat16)
        
        # Calculate Frobenius norm of the structural divergence
        delta = torch.norm(sandbox_grid.float() - target_grid.float(), p='fro')
        
        # Non-linear asymptotic scaling to prevent temperature explosions
        stress = self.max_penalty * (1.0 - torch.exp(-0.1 * delta))
        return stress.to(torch.bfloat16)


class ErrorHRRProjector(nn.Module):
    """
    Lifts a scalar thermodynamic stress value into a 4096-dimensional 
    Error Holographic Reduced Representation (Error HRR) on the S^{4095} hypersphere.
    """
    def __init__(self, d_wave: int = 4096):
        super().__init__()
        self.d_wave = d_wave
        # A static orthogonal basis vector representing pure "Error / Contradiction"
        self.register_buffer("error_basis", F.normalize(torch.randn(1, d_wave), p=2, dim=-1))

    @torch.no_grad()
    def forward(self, stress_scalar: torch.Tensor, current_wave: torch.Tensor) -> torch.Tensor:
        """
        Binds the error magnitude to the current wavefront via circular convolution,
        creating an ephemeral repeller.
        """
        # Scale the error basis by the stress scalar
        scaled_error = self.error_basis * stress_scalar.unsqueeze(-1)
        
        # Map to complex frequency domain for O(N log N) circular convolution
        wave_fft = torch.fft.fft(current_wave, dim=-1)
        error_fft = torch.fft.fft(scaled_error.to(current_wave.dtype), dim=-1)
        
        # Destructive interference projection
        repeller_fft = wave_fft * torch.conj(error_fft)
        
        # Return to spatial domain and clamp to S^{4095}
        repeller_wave = torch.fft.ifft(repeller_fft, n=self.d_wave, dim=-1)
        
        norm = torch.linalg.vector_norm(repeller_wave, ord=2, dim=-1, keepdim=True)
        return repeller_wave / (norm + 1e-8)


class ThermodynamicSandboxTransducer(nn.Module):
    """
    The closed-loop master. Ingests REPL outputs, computes morphological distance,
    and commands the DivergentMaster to inject Langevin heat into the phase masks.
    """
    def __init__(self, d_wave: int = 4096, base_temperature: float = 0.4):
        super().__init__()
        self.distance_metric = MorphologicalDistanceMetric()
        self.error_projector = ErrorHRRProjector(d_wave=d_wave)
        self.base_temperature = base_temperature

    def inject_langevin_heat(self, wave: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        """
        Simulates the microheater pulsing the BTO crystal. Shakes the tensor graph.
        """
        if temperature <= 0.0:
            return wave
            
        noise_real = torch.randn_like(wave.real) * torch.sqrt(temperature)
        noise_imag = torch.randn_like(wave.imag) * torch.sqrt(temperature)
        heated_wave = wave + noise_real + 1j * noise_imag
        
        # The wave must remain on the unit hypersphere post-heat injection
        norm = torch.linalg.vector_norm(heated_wave, ord=2, dim=-1, keepdim=True)
        return heated_wave / (norm + 1e-8)

    def forward(
        self, 
        sandbox_grid: Optional[torch.Tensor], 
        target_grid: torch.Tensor, 
        active_wavefront: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Closes the loop between discrete sandbox reality and continuous wave phase.
        
        Returns:
            The thermodynamically altered wavefront (forcing Viscoelastic Creep).
            Telemetry dict containing Sagnac Delta and applied Heat.
        """
        # 1. Measure the physical logic failure (Sagnac Delta equivalent)
        if sandbox_grid is None:
            stress = torch.tensor(10.0, device=active_wavefront.device) # Max penalty for execution crash
        else:
            stress = self.distance_metric(sandbox_grid, target_grid)

        # 2. Check for absolute topological resonance
        if stress.item() < 1e-4:
            # Constructive interference achieved. Lock the attractor.
            return active_wavefront, {"sagnac_delta": 0.0, "langevin_heat": 0.0, "resonance": True}

        # 3. Calculate dynamic temperature (Sagnac-Proportional Temperature Schedule)
        # T_t = T_base + kappa * (1 - e^(-Delta))
        applied_heat = self.base_temperature + (1.5 * (1.0 - math.exp(-stress.item())))
        applied_heat_tensor = torch.tensor(applied_heat, device=active_wavefront.device)

        # 4. Project the discrete failure into a continuous Error HRR
        error_wavefront = self.error_projector(stress, active_wavefront)

        # 5. Inject the Langevin heat into the combined wavefront
        combined_wave = active_wavefront - (0.1 * error_wavefront) # Shift away from the repeller
        heated_wavefront = self.inject_langevin_heat(combined_wave, applied_heat_tensor)

        telemetry = {
            "sagnac_delta": stress.item(),
            "langevin_heat": applied_heat,
            "resonance": False
        }

        return heated_wavefront, telemetry