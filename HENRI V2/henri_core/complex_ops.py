"""Complex helpers. One term per meaning.

cgelu  complex GeLU: gelu on the real part, gelu on the imaginary part.
        doc p5 names "Complex-valued GeLU (C-GeLU)" for the Gram transformer.
"""
from __future__ import annotations

import torch


def cgelu(z: torch.Tensor) -> torch.Tensor:
    """Complex GeLU, applied per component."""
    re = torch.nn.functional.gelu(z.real)
    im = torch.nn.functional.gelu(z.imag)
    return torch.complex(re, im)


def cabs2(z: torch.Tensor) -> torch.Tensor:
    return z.real ** 2 + z.imag ** 2


def as_real_pair(z: torch.Tensor) -> torch.Tensor:
    """Stack complex as a trailing real/imag pair: [..., D] -> [..., 2, D]."""
    return torch.stack([z.real, z.imag], dim=-2)
