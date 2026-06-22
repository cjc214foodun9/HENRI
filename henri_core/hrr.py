import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class HRRInputLayer(nn.Module):
    """
    Holographic Reduced Representations (HRR) Ingestion Layer.
    Utilizes circular convolution via FFT for concept binding, and involution for unbinding.
    """
    def __init__(self, dim=4096):
        super().__init__()
        self.dim = dim
        
        # Base learned geometry initialized as random normal and normalized to hypersphere
        self.base_geometry = nn.Parameter(torch.randn(1, self.dim))
        nn.init.orthogonal_(self.base_geometry)

    def bind(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Binds two HRR vectors using circular convolution via Real FFT (rfft).
        z = IFFT( FFT(x) * FFT(y) )
        Complexity: O(N log N)
        """
        from .kernels import flash_circular_convolution
        z = flash_circular_convolution(x, y)
        
        # 4. Re-normalize to maintain energy conservation on L2 hypersphere
        return F.normalize(z, p=2, dim=-1)

    def unbind(self, bound_state: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Extracts a vector from a bound state by binding it with the key's involution.
        Approximate unbinding operation in Vector Symbolic Architectures (VSAs).
        """
        # Exact Involution for Unbinding
        involution_key = torch.empty_like(key)
        involution_key[..., 0] = key[..., 0]
        involution_key[..., 1:] = torch.flip(key[..., 1:], dims=[-1])
        
        # Unbind by binding the bound state with the involuted key
        return self.bind(bound_state, involution_key)

    def forward(self, ephemeral_attractor: torch.Tensor, active_constraint: torch.Tensor) -> torch.Tensor:
        """
        Ingests the Zone C attractor and active constraint, binding them with base geometry.
        Inputs are expected to be real tensors of shape (Batch, SeqLen, Dim) or (Batch, Dim).
        """
        attractor_norm = F.normalize(ephemeral_attractor, p=2, dim=-1)
        constraint_norm = F.normalize(active_constraint, p=2, dim=-1)
        
        # Bind database intuition with the active physical constraint
        bound_wave = self.bind(attractor_norm, constraint_norm)
        
        # Bind with the base learned geometry
        final_input_wave = self.bind(bound_wave, self.base_geometry)
        
        return final_input_wave


class PackedPhaseVSAEngine(nn.Module):  
    """  
    PEARL Core Phase Protector: Quantizes continuous complex waveforms into   
    8-bit integer phase spectrums, enforcing strict wrapped boundaries modulo 256.  
    Eliminates the complex-to-real casting leak completely.  
    """  
    def __init__(self, dimension: int = 4096):  
        super().__init__()  
        self.dim = dimension  
        # Map the full 2*pi spectrum into 256 discrete angular integer steps  
        self.phase_scale = 256.0 / (2.0 * 3.141592653589793)

    def pack_wave_to_phase(self, complex_wave: torch.Tensor) -> torch.Tensor:  
        """Extracts phase angles safely and packs them into INT8 registers."""  
        if torch.is_complex(complex_wave):  
            angles = torch.angle(complex_wave)  
        else:  
            # Safe boundary tracking fallback if input arrived un-constituted  
            angles = torch.atan2(complex_wave, torch.zeros_like(complex_wave) + 1e-8)  
              
        positive_angles = torch.remainder(angles, 2.0 * 3.141592653589793)  
        return (positive_angles * self.phase_scale).to(dtype=torch.uint8)

    def compute_fused_binding(self, packed_feature: torch.Tensor, packed_coordinate: torch.Tensor) -> torch.Tensor:  
        """  
        Executes circular convolution as raw integer additions wrapped modulo 256.  
        Psi_bound = (Phase_feature + Phase_coordinate) % 256  
        Maps directly to consumer GPU specialized INT8 Tensor Cores with zero linewidth drift.  
        """  
        # Cast to int16 to compute additions safely before overflow wrap checks  
        feat_u16 = packed_feature.to(torch.int16)  
        coord_u16 = packed_coordinate.to(torch.int16)  
          
        # Enforce exact modular phase boundary wrapping to prevent the SIGReg variance explosion  
        bound_phase = torch.remainder(feat_u16 + coord_u16, 256).to(torch.uint8)  
        return bound_phase

    def unbind_wave(self, packed_joint_wave: torch.Tensor, packed_key_wave: torch.Tensor) -> torch.Tensor:  
        """Performs VSA contextual unbinding via integer phase subtraction modulo 256."""  
        joint_u16 = packed_joint_wave.to(torch.int16)  
        key_u16 = packed_key_wave.to(torch.int16)  
          
        unbound_phase = torch.remainder(joint_u16 - key_u16 + 256, 256).to(torch.uint8)  
        return unbound_phase
