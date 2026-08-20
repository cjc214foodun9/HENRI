"""Flag-gated structured character-position qFHRR codec for Run 21.

This module is an experiment. It does not replace qFHRREpistemicCodec.

Representation for a string of length ``n``:

* character ``c`` -> deterministic atomic ring ``q_c`` in Z_256^D;
* normalized token position ``p_i = i / max(1, n - 1)``;
* position ring ``q_pos(i,d) = round(p_i * q_P[d]) mod 256``;
* bound ring ``q_bound = (q_pos + q_c) mod 256``;
* bundle the bound phases in the complex plane and quantize the resulting
  phase back to one uint8 ring.

The normalized position is the finite qFHRR realization of a fractional power
of a position rotor. The position frequency ring ``q_P`` is deterministic and
fixed. The implementation is stateless with respect to MBPP items: it does not
read Zone C, fit a vocabulary on evaluation strings, or persist task data.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import math
from typing import Optional

import torch
import torch.nn as nn


class StructuredCharPositionCodec(nn.Module):
    """Deterministic structured qFHRR text codec.

    The public attributes and ``encode_text`` contract match the legacy codec
    used by the rank probe: ``d_model``, ``k_bins``, ``device``, and a flat
    uint8 ring in ``[0, k_bins - 1]``.
    """

    codec_name = "structured_char_position_qfhrr"
    codec_version = "run21-v1"

    def __init__(
        self,
        d_model: int = 65536,
        k_bins: int = 256,
        device: Optional[str] = None,
        max_cache_entries: int = 512,
        position_mode: str = "full",
    ):
        super().__init__()
        if k_bins != 256:
            raise ValueError("Run21 structured codec requires k_bins=256")
        if position_mode not in ("full", "none", "shuffled", "independent"):
            raise ValueError(
                f"position_mode must be full|none|shuffled|independent, got {position_mode!r}")
        self.d_model = int(d_model)
        self.k_bins = int(k_bins)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_cache_entries = int(max_cache_entries)
        self.position_mode = position_mode
        self._shuffle_cache: dict[int, list[int]] = {}
        self._position_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._token_cpu: dict[str, torch.Tensor] = {}
        self._ring_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        # q_P is a fixed phase-frequency ring. Position powers are generated
        # from this ring instead of allocating one independent ring per index.
        self._position_base_cpu = self._random_ring_cpu("position_base", "P")

    @staticmethod
    def _seed(namespace: str, value: str | int) -> int:
        raw = f"henri-run21:{namespace}:{value}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)

    def _random_ring_cpu(self, namespace: str, value: str | int) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(
            self._seed(namespace, value)
        )
        return torch.randint(
            0,
            self.k_bins,
            (self.d_model,),
            dtype=torch.uint8,
            generator=generator,
            device="cpu",
        )

    def _token_ring(self, token: str) -> torch.Tensor:
        ring = self._token_cpu.get(token)
        if ring is None:
            ring = self._random_ring_cpu("token", token)
            self._token_cpu[token] = ring
        return ring

    def _position_ring(self, position: int, length: Optional[int] = None) -> torch.Tensor:
        """Return the quantized fractional position phase ring.

        ``length`` is part of the position convention. If omitted, the
        diagnostic uses ``p_0=0`` and ``p_1=1`` for the first two positions.
        The ring is calculated on demand from one 65,536-element base ring, so
        long or varied source strings cannot grow an unbounded position cache.
        """
        position = int(position)
        if position < 0:
            raise ValueError("position must be non-negative")
        if self.position_mode == "none":
            # Match token-ring device (CPU): _bundle sums token and position
            # rings before the explicit device transfer.
            return torch.zeros(self.d_model, dtype=torch.uint8, device="cpu")
        if self.position_mode == "independent":
            # Compositional carrier fix (8.39, Evolution I remedy):
            # an INDEPENDENT random position ring per absolute index, seeded
            # by (position, length). Positions become approximately orthogonal
            # instead of collinear on a single scaled rotor. Shared characters
            # at different positions no longer superimpose coherently.
            key = (int(position), int(length if length is not None else -1))
            ring = self._position_cache.get(key)
            if ring is None:
                ring = self._random_ring_cpu(
                    "position_independent", f"{position}|{length}"
                )
                self._position_cache[key] = ring
            return ring
        if length is None:
            denominator = max(1, position)
        else:
            length = int(length)
            denominator = max(1, length - 1)
            if self.position_mode == "shuffled":
                perm = self._shuffle_cache.get(length)
                if perm is None:
                    generator = torch.Generator(device="cpu").manual_seed(
                        self._seed("position_shuffle", length))
                    perm = torch.randperm(length, generator=generator).tolist()
                    self._shuffle_cache[length] = perm
                position = perm[position]
        fraction = float(position) / float(denominator)
        phase_codes = torch.round(
            self._position_base_cpu.to(torch.float32) * fraction
        )
        return torch.remainder(phase_codes, self.k_bins).to(torch.uint8)

    def _bundle(self, text: str) -> torch.Tensor:
        if not text:
            return torch.zeros(self.d_model, dtype=torch.uint8, device=self.device)

        # Keep source rings on CPU and transfer bounded chunks. This avoids a
        # [len(text), D] allocation for the full source string.
        real = torch.zeros(self.d_model, dtype=torch.float32, device=self.device)
        imag = torch.zeros_like(real)
        text_length = len(text)
        for start in range(0, text_length, 32):
            chunk = text[start : start + 32]
            tokens = torch.stack([self._token_ring(ch) for ch in chunk], dim=0)
            positions = torch.stack(
                [
                    self._position_ring(i, text_length)
                    for i in range(start, start + len(chunk))
                ],
                dim=0,
            )
            bound = (tokens.to(torch.int32) + positions.to(torch.int32)) % self.k_bins
            angles = bound.to(self.device, dtype=torch.float32)
            angles = angles * (2.0 * math.pi / self.k_bins)
            real.add_(torch.cos(angles).sum(dim=0))
            imag.add_(torch.sin(angles).sum(dim=0))

        phase = torch.remainder(torch.atan2(imag, real), 2.0 * math.pi)
        q = torch.round(phase * (self.k_bins / (2.0 * math.pi))) % self.k_bins
        return q.to(torch.uint8)

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode a string into a deterministic structured Z_256 ring."""
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        cached = self._ring_cache.get(text)
        if cached is not None:
            self._ring_cache.move_to_end(text)
            return cached
        ring = self._bundle(text)
        self._ring_cache[text] = ring
        while len(self._ring_cache) > self.max_cache_entries:
            self._ring_cache.popitem(last=False)
        return ring

    def bind_hadamard(self, q_key: torch.Tensor, q_val: torch.Tensor) -> torch.Tensor:
        return (q_key.to(torch.int32) + q_val.to(torch.int32)) % self.k_bins

    def unbind_hadamard(self, q_bound: torch.Tensor, q_key: torch.Tensor) -> torch.Tensor:
        return (q_bound.to(torch.int32) - q_key.to(torch.int32)) % self.k_bins

    def compute_similarity(self, q1: torch.Tensor, q2: torch.Tensor) -> float:
        """Compute the phase cosine, not the integer-code cosine."""
        q1 = q1.to(self.device, dtype=torch.int32)
        q2 = q2.to(self.device, dtype=torch.int32)
        phase = (q1 - q2) % self.k_bins
        return float(
            torch.cos(phase.to(torch.float32) * (2.0 * math.pi / self.k_bins))
            .mean()
            .item()
        )

    def geometry_metadata(self) -> dict[str, object]:
        return {
            "codec_name": self.codec_name,
            "codec_version": self.codec_version,
            "tokenizer": "character_codepoint",
            "position_binding": "round((i/max(1,n-1))*q_P) plus q_token mod 256",
            "position_index": "normalized_fractional_character_index",
            "position_mode": self.position_mode,
            "bundling": "complex_phase_sum_then_quantize",
            "d_model": self.d_model,
            "k_bins": self.k_bins,
            "zone_c_dependency": False,
        }


def make_structured_codec(device: Optional[str] = None) -> StructuredCharPositionCodec:
    """Factory used by the explicitly selected Run 21 probe path."""
    return StructuredCharPositionCodec(device=device)


def ring_to_real(codec, ring: torch.Tensor) -> torch.Tensor:
    """Map a codec ring to the real wave coordinate consumed by the probe.

    The legacy path preserves its historical linear diagnostic map exactly.
    Structured qFHRR rings use a phase coordinate, so downstream EDMD and
    candidate ranking do not silently treat integer phase labels as amplitudes.
    The output remains length D to preserve the Run 20 tensor contract.
    """
    if isinstance(codec, StructuredCharPositionCodec):
        phase = ring.to(torch.float32) * (2.0 * math.pi / codec.k_bins)
        return torch.cos(phase)
    return ring.to(torch.float32) / (codec.k_bins - 1) * 2.0 - 1.0
