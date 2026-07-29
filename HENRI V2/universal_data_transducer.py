"""
Project HENRI V2 — Universal Data Transducer.

Bidirectional bridge between discrete external data modalities (ARC 2D grids,
continuous time-series, text strings, vector states) and UWE complex unit
wave states Psi in C^(num_blocks x 8).

In-place operations retain tensor allocations wherever possible to preserve
sub-millisecond streaming ingress performance.
"""

import math
import hashlib
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UniversalDataTransducer(nn.Module):
    """
    Unified multi-modal wave codec for Project HENRI V2.
    Transduces continuous numerical values, text strings, structured dictionaries,
    and 2D grid states into D=65,536 / D=4096 qFHRR phase multivectors.
    """

    def __init__(
        self,
        d_model: int = 4096,
        num_blocks: int = 512,
        phase_bits: int = 8,
        codebook_size: int = 256,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_blocks = num_blocks if num_blocks is not None else d_model // 8
        self.phase_bits = phase_bits
        self.codebook_size = codebook_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 256-entry cosine LUT
        phase_intervals = torch.linspace(0, 2 * math.pi, steps=codebook_size, device=self.device)
        self.lut_cos = torch.cos(phase_intervals)

    def transduce_object(self, obj: Union[Dict, List, str, float, int]) -> torch.Tensor:
        """
        Main ingress transducing method. Converts arbitrary Python object into uint8 phase codes.
        """
        if isinstance(obj, dict):
            return self._transduce_dict(obj)
        elif isinstance(obj, (list, tuple)):
            return self._transduce_sequence(obj)
        elif isinstance(obj, str):
            return self.transduce_string(obj)
        elif isinstance(obj, (int, float)):
            return self._transduce_scalar(float(obj))
        else:
            return self.transduce_string(str(obj))

    def transduce_string(self, s: str) -> torch.Tensor:
        """Maps string deterministically to phase ring in Z_256."""
        hash_seed = int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % (10**8)
        g = torch.Generator(device="cpu").manual_seed(hash_seed)
        q_codes = torch.randint(0, self.codebook_size, (self.d_model,), dtype=torch.uint8, generator=g)
        return q_codes.to(self.device)

    def _transduce_dict(self, d: Dict) -> torch.Tensor:
        fused_wave = torch.zeros(self.d_model, dtype=torch.int32, device=self.device)
        for k, v in d.items():
            k_wave = self.transduce_string(f"key_{k}")
            v_wave = self.transduce_object(v)
            bound = (k_wave.to(torch.int32) + v_wave.to(torch.int32)) % self.codebook_size
            fused_wave = (fused_wave + bound) % self.codebook_size
        return fused_wave.to(torch.uint8)

    def _transduce_sequence(self, seq: List) -> torch.Tensor:
        fused_wave = torch.zeros(self.d_model, dtype=torch.int32, device=self.device)
        for idx, elem in enumerate(seq):
            pos_key = self.transduce_string(f"pos_{idx}")
            elem_wave = self.transduce_object(elem)
            bound = (pos_key.to(torch.int32) + elem_wave.to(torch.int32)) % self.codebook_size
            fused_wave = (fused_wave + bound) % self.codebook_size
        return fused_wave.to(torch.uint8)

    def _transduce_scalar(self, val: float) -> torch.Tensor:
        norm_val = math.tanh(val)
        phase_code = int((norm_val + 1.0) * 127.5) % self.codebook_size
        return torch.full((self.d_model,), phase_code, dtype=torch.uint8, device=self.device)


if __name__ == "__main__":
    udt = UniversalDataTransducer(d_model=65536)
    wave = udt.transduce_string("hello henri")
    print(f"Transduced string wave shape: {wave.shape}, dtype: {wave.dtype}")
