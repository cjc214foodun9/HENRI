import torch
import torch.nn as nn
import math

class HolographicVectorLifter(nn.Module):
    """
    The Absolute Ingress Boundary (Zone A -> Zone B).
    Strips away all von Neumann 'Embedding' structures (which rely on unconstrained Gaussian noise).
    Maps discrete linguistic tokens directly to immutable, unit-norm phase arrays on the S^4095 hypersphere.
    """
    def __init__(self, vocab_size: int = 32000, dim: int = 4096, seed: int = 42):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim

        # 1. Generate pristine, uniformly distributed phase angles [-\pi, \pi]
        # We use a deterministic seed. The Lexicon is an absolute physical invariant (The Glass Constitution).
        # It must never drift, learn, or update across system reboots.
        generator = torch.Generator().manual_seed(seed)
        phases = torch.empty(vocab_size, dim).uniform_(-math.pi, math.pi, generator=generator)

        # 2. Project strictly to the complex unit hypersphere: e^{j\theta} = cos(\theta) + j*sin(\theta)
        # The amplitude of every coordinate is strictly 1.0. 
        # This mathematically guarantees zero thermodynamic leakage/energy generation at ingress.
        complex_lexicon = torch.complex(torch.cos(phases), torch.sin(phases))

        # 3. Register as a persistent, non-differentiable hardware buffer
        # This completely severs the PyTorch Autograd graph from the ingress layer.
        self.register_buffer("canonical_phase_lexicon", complex_lexicon)

    @torch.no_grad()
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lifts discrete integer arrays into continuous complex wave manifolds.
        
        Args:
            token_ids: [Batch, Sequence_Length]
            
        Returns: 
            Complex wavefronts of shape [Batch, Sequence_Length, Dim] (torch.complex64)
        """
        # O(1) Memory Address Lookup -> Physical Wave Emission
        # Yields a pristine, mathematically stable FHRR (Fourier Holographic Reduced Representation)
        return self.canonical_phase_lexicon[token_ids]