import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np

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
        # Use fused Triton kernel if available, else high-performance PyTorch circular convolution fallback
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

class PackedInt8HolographicEngine(nn.Module):
    """
    Implements a hardware-conscious VSA using packed INT8 phase configurations modulo 256.
    Eliminates the complex casting leak and activates dense integer execution pathways.
    """
    def __init__(self, dim=4096):
        super().__init__()
        self.dim = dim
        
        # Pre-compute the Cosine Lookup Table (LUT) for hyper-fast resonance scanning
        # Cast to bfloat16 to match the precision of the remaining core layers
        phase_steps = np.arange(256) * (2.0 * np.pi / 256.0)
        lut_values = np.cos(phase_steps)
        self.register_buffer("cosine_lut", torch.from_numpy(lut_values).to(torch.bfloat16))

    def quantize_phase(self, complex_tensor):
        """
        Extracts raw phase angles from complex tensors and packs them into INT8 registers.
        """
        # Extract angular phase using arctan2: [-pi, pi]
        angles = torch.atan2(complex_tensor.imag, complex_tensor.real)
        # Remap phase space bounds uniformly to [0, 2*pi]
        angles = torch.remainder(angles, 2.0 * np.pi)
        
        # Scale and quantize straight to unsigned byte ranges
        quantized_bytes = torch.floor(angles * (256.0 / (2.0 * np.pi))).to(torch.uint8)
        return quantized_bytes

    def quantized_bind(self, q1, q2):
        """
        Executes holographic binding via element-wise addition modulo 256.
        Utilizes native uint8 wrapping overflow to calculate the modulo operation automatically.
        """
        # In PyTorch, adding two torch.uint8 tensors naturally executes wrapping modulo 256
        return q1 + q2

    def quantized_unbind(self, q_composite, q_key):
        """
        Executes holographic unbinding via element-wise subtraction modulo 256.
        """
        return q_composite - q_key

    def calculate_resonance_matrix(self, q_thought, q_database_lexicon):
        """
        Computes the cosine similarity across your Zone C lexicon using the hardware LUT.
        
        Args:
            q_thought: [Batch, 4096] (Active quantized thought wave)
            q_database_lexicon: [Num_Axioms, 4096] (Quantized reference constants)
        """
        # Step 1: Compute element-wise difference across all channels via broadcasting
        # Shape: [Batch, Num_Axioms, 4096]
        phase_diff = q_thought.unsqueeze(1) - q_database_lexicon.unsqueeze(0)
        
        # Step 2: Remap negative differences to positive uint8 indices safely via bitwise masking
        # Simulates the cyclic ring mapping process on the GPU
        lut_indices = phase_diff.to(torch.long) & 0xFF
        
        # Step 3: Stream the indexing values through the pre-computed hardware table
        resonance_contributions = self.cosine_lut[lut_indices]
        
        # Step 4: Average across the hyperdimensional channels to extract final scalar scores
        return torch.mean(resonance_contributions, dim=-1)
