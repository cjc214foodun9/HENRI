import torch
import torch.nn as nn
import math

class AngularSpectrumPropagator(nn.Module):
    """
    Simulates physical free-space light diffraction between BTO phase masks.
    Calculates the exact Helmholtz transfer function to prevent aliasing.
    Removes in-place modifications to preserve Autograd activation snapshots cleanly.
    """
    def __init__(self, N: int, length: float, wavelength: float, z: float, device='cpu'):
        super().__init__()
        # Calculate spatial frequencies for the 2D meshgrid
        dx = length / N
        fx = torch.fft.fftfreq(N, d=dx, device=device)
        fy = torch.fft.fftfreq(N, d=dx, device=device)
        FX, FY = torch.meshgrid(fx, fy, indexing='ij')
        
        # Free-space transfer function argument
        k = 2 * math.pi / wavelength
        argument = 1.0 - (wavelength * FX)**2 - (wavelength * FY)**2
        
        # evanescent wave filter (eliminates non-propagating waves to save compute)
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
    A single BTO crystal cross-section.
    Acts as a learnable phase mask during HITL training.
    Utilizes Bounded Phase Perturbation to prevent gradient dead zones.
    """
    def __init__(self, N: int, device='cpu', max_phase_depth=0.1):
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


class SagnacInterferometer(nn.Module):
    """
    Simulates the physical Sagnac loops that test logical consistency.
    Constructive interference = Truth; Destructive interference = Hallucination.
    """
    def __init__(self):
        super().__init__()

    def forward(self, current_wave: torch.Tensor, target_manifold: torch.Tensor):
        # Destructive interference (The Error Energy / Hallucination)
        reflection_delta = current_wave - target_manifold
        
        # Constructive interference (The Epiplexity / Truth)
        transmission_truth = current_wave - reflection_delta
        
        # Calculate Thermodynamic Penalty (E = sum(|delta|^2))
        error_energy = torch.sum(torch.real(reflection_delta * torch.conj(reflection_delta)))
        
        return transmission_truth, reflection_delta, error_energy


class ZoneBEmulator(nn.Module):
    """
    The 200-Million Parameter Software-Defined Physics Engine.
    """
    def __init__(self, resolution_scale: float = 1.0, length=0.016, wavelength=1550e-9, z=0.001, device='cpu'):
        super().__init__()
        self.N = int(6324 * resolution_scale)
        self.device = device
        
        # 5 cascading Diffractive Deep Neural Network (D2NN) layers
        self.layers = nn.ModuleList([OpticalD2NNLayer(self.N, device) for _ in range(5)])
        # Free-space propagation between the layers
        self.propagators = nn.ModuleList([AngularSpectrumPropagator(self.N, length, wavelength, z, device) for _ in range(4)])
        # The final Logic Gate
        self.sagnac_veto = SagnacInterferometer()

    def set_microheaters(self, langevin_heat: float):
        """Pulses heat into all 5 crystal phase masks to escape logical minima."""
        for layer in self.layers:
            layer.inject_langevin_noise(langevin_heat)

    def forward(self, wave: torch.Tensor, target_manifold: torch.Tensor, langevin_heat: float = 0.0):
        # 1. Check if the Zone A Divergent Master is calling for a thermodynamic shake
        if langevin_heat > 0.0:
            self.set_microheaters(langevin_heat)

        # 2. Prevent mutating the input tensor if grad is required by Zone A
        current_wave = wave.clone() 

        # 3. Propagate light through the 5 D2NN boundaries
        for i in range(5):
            current_wave = self.layers[i](current_wave)
            if i < 4:
                current_wave = self.propagators[i](current_wave)
                
        # 4. Geometrically verify the resulting thought against the Zone C Axiom
        transmission_truth, reflection_delta, error_energy = self.sagnac_veto(current_wave, target_manifold)
        
        return transmission_truth, reflection_delta, error_energy
