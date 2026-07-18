import torch

class BackwardEpistemicTransducer:
    """
    Project HENRI: Backward Epistemic Transducer
    Maps discrete spatial failures from the physical sandbox back into 
    continuous phase-gradients to constrain the Anisotropic Thermostat.
    """
    def __init__(self, target_dim: int = 65536):
        self.target_dim = target_dim

    def transduce_discrete_error_to_gradient(self, error_mask: torch.Tensor) -> torch.Tensor:
        """
        Converts a spatial boolean error mask into a continuous phase-gradient using iDFT.
        error_mask: [seq_len] boolean or float mask of mismatches.
        Returns: continuous phase-gradient of shape [target_dim].
        """
        error_signal = error_mask.to(torch.float32)
        
        if error_signal.size(-1) < self.target_dim:
            # Flatten if multi-dimensional
            error_signal = error_signal.view(-1)
            padded_signal = torch.nn.functional.pad(error_signal, (0, self.target_dim - error_signal.size(0)))
        else:
            error_signal = error_signal.view(-1)
            padded_signal = error_signal[:self.target_dim]
            
        complex_signal = torch.complex(padded_signal, torch.zeros_like(padded_signal))
        
        # Map to continuous phase-gradient using Inverse Discrete Fourier Transform (iDFT)
        phase_gradient = torch.fft.ifft(complex_signal)
        
        # We return the magnitude projection, flattened to act as a localized error_mask for the AnisotropicThermostat
        # Expected shape is [dimension] (e.g. 65536) to broadcast with experts_A[idx] shape [16, 65536]
        return torch.abs(phase_gradient).view(-1)

    def generate_thermodynamic_repeller(self, falsified_wave: torch.Tensor) -> torch.Tensor:
        """
        Computes the circular convolution of the complex conjugate to act as a 
        thermodynamic repeller in the Hierarchical Growing Memory Cache.
        falsified_wave: The wave state that led to the sandbox failure.
        """
        if not falsified_wave.is_complex():
            falsified_wave_complex = torch.complex(falsified_wave, torch.zeros_like(falsified_wave))
        else:
            falsified_wave_complex = falsified_wave
            
        fft_w = torch.fft.fft(falsified_wave_complex)
        repeller_field = torch.fft.ifft(fft_w * torch.conj(fft_w))
        
        # Ensure it matches original dtype
        if not falsified_wave.is_complex():
            repeller_field = repeller_field.real
            
        return repeller_field / (torch.norm(repeller_field, p=2, dim=-1, keepdim=True) + 1e-9)
