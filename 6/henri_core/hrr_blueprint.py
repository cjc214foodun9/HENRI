import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class QuantizedFHRREngine(nn.Module):
    """
    qFHRR: Quantized Fourier Holographic Reduced Representation Engine.
    Replaces slow complex64 floating-point operations with quantized phase angles 
    packed into discrete integer spaces [0, 255] modulo 256.
    Enforces strict mathematical invariants of Vector Symbolic Architectures (VSAs) 
    while preventing representation saturation and casting leaks.
    """
    def __init__(self, dim: int = 4096, num_phases: int = 256):
        super().__init__()
        self.dim = dim
        self.num_phases = num_phases
        # Register a constant phase-wheel to map integer indices back to complex coordinates
        phase_angles = torch.linspace(-math.pi, math.pi - (2 * math.pi / num_phases), num_phases)
        self.register_buffer("phase_cos", torch.cos(phase_angles))
        self.register_buffer("phase_sin", torch.sin(phase_angles))

    def generate_unitary_key(self, num_vectors: int = 1, device: torch.device = None) -> torch.Tensor:
        """
        Generates pristine, orthogonal phase hypervectors on the S^{4095} unit hypersphere.
        Each phase coordinate is randomly distributed across the 256-phase integer wheel.
        """
        if device is None:
            device = self.phase_cos.device
        # Phase indices are selected uniformly from [0, 255]
        return torch.randint(0, self.num_phases, (num_vectors, self.dim), dtype=torch.uint8, device=device)

    def bind(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Executes holographic binding in phase space via element-wise addition modulo 256.
        Natively reproduces O(N log N) circular convolution without floating-point drift.
        """
        # Element-wise addition modulo self.num_phases (256)
        # Casting to int32 to prevent unsigned integer overflow truncation during addition
        return ((x.to(torch.int32) + y.to(torch.int32)) % self.num_phases).to(torch.uint8)

    def unbind(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Executes holographic unbinding (decoding) via element-wise subtraction modulo 256.
        Inverse operation of binding: x * y^-1
        """
        # Element-wise subtraction modulo self.num_phases (256)
        return ((x.to(torch.int32) - y.to(torch.int32) + self.num_phases) % self.num_phases).to(torch.uint8)

    def bundle(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Combines an array of hypervectors via hyperspherical superposition.
        To preserve similarity invariants without amplitude blowup, the integer phases 
        are mapped to the complex plane, summed vectorially, and re-quantized.
        """
        if vectors.dim() < 2:
            return vectors
        
        # Map quantized integers back to continuous 2D coordinates (cos, sin)
        cos_components = self.phase_cos[vectors.to(torch.long)]
        sin_components = self.phase_sin[vectors.to(torch.long)]
        
        # Execute vectorial sum along the batch dimension
        sum_cos = torch.sum(cos_components, dim=0)
        sum_sin = torch.sum(sin_components, dim=0)
        
        # Compute the continuous phase angle using arc tangent
        continuous_phases = torch.atan2(sum_sin, sum_cos) # Bounded in [-pi, pi]
        
        # Map continuous phases back to the nearest discrete integer index
        quantized_phases = torch.round(((continuous_phases + math.pi) / (2 * math.pi)) * self.num_phases) % self.num_phases
        return quantized_phases.to(torch.uint8)

    def similarity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the cosine similarity between two phase hypervectors.
        Converts integer indices back to 2D Unitary Wave Representations and computes dot-product.
        """
        # Ensure correct dimensions
        x_idx = x.to(torch.long)
        y_idx = y.to(torch.long)
        
        x_cos, x_sin = self.phase_cos[x_idx], self.phase_sin[x_idx]
        y_cos, y_sin = self.phase_cos[y_idx], self.phase_sin[y_idx]
        
        # Compute dot product of reconstituted real and imaginary wave planes
        dot_product = (x_cos * y_cos) + (x_sin * y_sin)
        return torch.mean(dot_product, dim=-1)

    def continuous_reconstitution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts quantized integer phase vectors into active torch.complex64 waves 
        ready to propagate through the Zone B optical emulator.
        """
        idx = x.to(torch.long)
        real_plane = self.phase_cos[idx]
        imag_plane = self.phase_sin[idx]
        return torch.complex(real_plane, imag_plane)
