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
        # 1. Project to Frequency Domain (Real FFT)
        X_freq = torch.fft.rfft(x, dim=-1)
        Y_freq = torch.fft.rfft(y, dim=-1)
        
        # 2. Complex Element-wise Multiplication (Binding)
        Z_freq = X_freq * Y_freq
        
        # 3. Inverse FFT back to spatial/wave domain
        z = torch.fft.irfft(Z_freq, n=self.dim, dim=-1)
        
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
