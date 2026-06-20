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


class FourierSpectralDiffractionSurrogate(nn.Module):
    """
    Fourier Neural Operator (FNO) Surrogate Block.
    Replaces step-by-step ASM numerical Helmholtz wave propagation with an
    operator mapping that executes at native tensor core speeds.
    """
    def __init__(self, modes: int = 128, dimension: int = 4096):
        super().__init__()
        self.modes = modes
        self.dim = dimension
        
        # Parameterize the phase transformations directly as complex weights in Fourier space
        self.fourier_weight_real = nn.Parameter(torch.randn(modes, modes) * (1.0 / modes))
        self.fourier_weight_imag = nn.Parameter(torch.randn(modes, modes) * (1.0 / modes))

    def forward(self, psi_wave: torch.Tensor) -> torch.Tensor:
        # 1. Transform ingress wave into the frequency domain along the context row
        # Ensure input is complex, otherwise cast
        psi_complex = psi_wave.to(dtype=torch.complex64)
        psi_fft = torch.fft.fft(psi_complex, dim=-1)
        out_fft = torch.zeros_like(psi_fft)
        
        # 2. Extract and modulate higher spatial frequency modes natively
        w_complex = torch.complex(self.fourier_weight_real, self.fourier_weight_imag).to(device=psi_wave.device, dtype=torch.complex64)
        
        # Execute matrix multiplication spectral mixing pass
        modes_to_apply = min(self.modes, psi_fft.size(-1))
        w_complex_sliced = w_complex[:modes_to_apply, :modes_to_apply]
        out_fft[..., :modes_to_apply] = torch.matmul(psi_fft[..., :modes_to_apply], w_complex_sliced)
        
        # 3. Invert back to the boundary plane in a single execution step
        psi_diffracted = torch.fft.ifft(out_fft, dim=-1)
        return torch.nn.functional.normalize(psi_diffracted, p=2, dim=-1)
