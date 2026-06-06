import torch
import torch.nn as nn
import math

class AngularSpectrumPropagator(nn.Module):
    """
    Simulates physical free-space light diffraction between BTO phase masks.
    Calculates the exact Helmholtz transfer function to prevent aliasing.
    Removes in-place modifications to preserve Autograd activation snapshots cleanly.
    """
    def __init__(self, N: int, length: float, wavelength: float, z: float, device='cuda'):
        super().__init__()
        # Calculate spatial frequencies for the 2D meshgrid
        dx = length / N
        fx = torch.fft.fftfreq(N, d=dx, device=device)
        fy = torch.fft.fftfreq(N, d=dx, device=device)
        FX, FY = torch.meshgrid(fx, fy, indexing='ij')
        
        # Free-space transfer function argument
        k = 2 * math.pi / wavelength
        argument = 1.0 - (wavelength * FX)**2 - (wavelength * FY)**2
        
        # Evanescent wave filter (eliminates non-propagating waves to save compute)
        evanescent_filter = (argument > 0).float()
        argument = torch.clamp(argument, min=0.0)
        
        phase = k * z * torch.sqrt(argument)
        
        # Precompute the complex transfer function H(f_x, f_y)
        H = torch.polar(evanescent_filter, phase)
        self.register_buffer('H', H)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        # 1. Transform to spatial frequency domain via Orthonormal 2D FFT
        wave_fft = torch.fft.fft2(wave, norm='ortho')
        # 2. Out-of-place multiplication ensures Autograd retains clean forward state access
        wave_fft_modulated = wave_fft * self.H
        # 3. Transform back to the physical spatial domain
        return torch.fft.ifft2(wave_fft_modulated, norm='ortho')


class OpticalD2NNLayer(nn.Module):
    """
    A single 40-Million parameter BTO crystal cross-section.
    Acts as a learnable phase mask during HITL training.
    Utilizes Bounded Phase Perturbation to prevent gradient dead zones.
    """
    def __init__(self, N: int, device='cuda', max_phase_depth=0.1):
        super().__init__()
        self.N = N
        # Bounded Physical Phase Perturbation (Independent of N, prevents Xavier collapse)
        init_bounds = max_phase_depth * math.pi
        self.phase_mask = nn.Parameter(
            torch.empty((N, N), dtype=torch.float32, device=device).uniform_(-init_bounds, init_bounds)
        )
        
    def inject_langevin_noise(self, heat: float):
        """Thermodynamic Annealing: physically shakes the local BTO topology."""
        if heat > 0.0:
            with torch.no_grad():
                noise = torch.randn_like(self.phase_mask) * math.sqrt(heat)
                self.phase_mask.add_(noise)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        # Calculate complex transmission coefficient t = exp(i * phi) (out of place, no modulo)
        t = torch.polar(torch.ones_like(self.phase_mask), self.phase_mask)
        # Out-of-place element-wise wave modulation
        return wave * t
