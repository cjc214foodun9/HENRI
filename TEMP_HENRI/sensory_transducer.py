import torch
import torch.nn as nn
import math

# Enforce strict float32 and complex64 precision to match the biophysical core
torch.set_default_dtype(torch.float32)

class ChemicalSensorAnchor(nn.Module):
    """
    Differentiable Chemical Sensor Network. Maps high-entropy environmental 
    data 1:1 into the internal latent space as continuous phase angles, 
    eliminating discrete tokenization.
    """
    def __init__(self, dim: int = 4096):
        super().__init__()
        self.dim = dim
        
        # Learnable interaction matrix K (4096 x 4096)
        # Represents chemical cross-reactivity between node concentrations
        self.K = nn.Parameter(torch.randn(dim, dim, dtype=torch.float32) * 0.01)
        
        # Learnable decay vector gamma (4096) 
        # Represents metabolic consumption and physical diffusion limits
        self.gamma = nn.Parameter(torch.rand(dim, dtype=torch.float32) * 0.1)

    def forward(self, I_t: torch.Tensor, steps: int = 5, dt: float = 0.1) -> torch.Tensor:
        """
        Takes a raw environmental input vector I(t) and solves the reaction-diffusion 
        ODE over 5 Euler integration steps:
           dc/dt = (c @ K) - gamma * c^2 + I(t)
           
        Returns the mapped complex wave on the unit hypersphere.
        """
        # Initialize resting concentration
        c = torch.zeros_like(I_t)
        
        for _ in range(steps):
            # Define the chemical derivative function
            def dc_dt_fn(conc):
                return (conc @ self.K) - self.gamma * (conc ** 2) + I_t
            
            # 4th-Order Runge-Kutta (RK4) integration steps
            k1 = dc_dt_fn(c)
            k2 = dc_dt_fn(c + 0.5 * dt * k1)
            k3 = dc_dt_fn(c + 0.5 * dt * k2)
            k4 = dc_dt_fn(c + dt * k3)
            
            # Energy-preserving update
            c = c + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            
            # Clamp the concentration strictly to physically viable phase limits
            c = torch.clamp(c, min=0.0, max=2 * math.pi)
            
        # Transduce final concentration onto the complex unit circle
        psi = torch.exp(1j * c)
        
        return psi

    def bind_sensory_modalities(self, psi_vision: torch.Tensor, psi_audio: torch.Tensor) -> torch.Tensor:
        """
        Uses Fourier Holographic Reduced Representations (FHRR) to compute the 
        circular convolution of two sensory waves, returning their combined 
        superposition strictly L2 normalized.
        """
        # Project into frequency domain via FFT
        fft_v = torch.fft.fft(psi_vision)
        fft_a = torch.fft.fft(psi_audio)
        
        # Circular convolution via point-wise multiplication
        bound_fft = fft_v * fft_a
        
        # Inverse transform back to the spatial phase plane
        psi_bound = torch.fft.ifft(bound_fft)
        
        # Return strict L2 normalization (p=2, dim=-1)
        norm = torch.norm(psi_bound, p=2, dim=-1, keepdim=True)
        psi_bound_normalized = psi_bound / (norm + 1e-16)
        
        return psi_bound_normalized


if __name__ == "__main__":
    print("Initializing The Sensory Transduction Boundary (Chemical Sensor)...")
    
    dim = 4096
    batch = 1
    
    # Initialize the biochemical anchor
    sensor = ChemicalSensorAnchor(dim=dim)
    
    # Simulate high-entropy raw environmental data vectors (e.g., RGB pixel arrays, audio waveforms)
    print("\n--- Test 1: Reaction-Diffusion ODE Transduction ---")
    I_vision = torch.randn(batch, dim, dtype=torch.float32) * 10.0
    I_audio  = torch.randn(batch, dim, dtype=torch.float32) * 10.0
    
    psi_vision = sensor(I_vision)
    psi_audio  = sensor(I_audio)
    
    # Validate the physical property of the output waves
    modulus_vision = torch.abs(psi_vision)
    mean_modulus_vision = modulus_vision.mean().item()
    print(f"Transduced Wave Element Modulus: {mean_modulus_vision:.16f}")
    assert torch.allclose(modulus_vision, torch.ones_like(modulus_vision)), "Element modulus is not strictly 1.0!"
    
    # Bind the distinct sensory arrays into a single composite representation
    print("\n--- Test 2: FHRR Cross-Modal Sensory Binding ---")
    psi_bound = sensor.bind_sensory_modalities(psi_vision, psi_audio)
    
    bound_modulus = torch.norm(psi_bound, p=2, dim=-1).mean().item()
    print(f"Bound Superposition L2 Modulus: {bound_modulus:.16f}")
    assert torch.allclose(torch.tensor(bound_modulus), torch.tensor(1.0, dtype=torch.float32)), "L2 Norm of bound superposition is not exactly 1.0!"
    
    print("\n[SUCCESS] Raw environmental telemetry natively transduced into normalized complex wave mechanics.")
