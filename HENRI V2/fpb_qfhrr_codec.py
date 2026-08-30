"""Carrier F5: FPB structured compositional qFHRR codec (default-OFF).

Pre-registration: docs/spec/f5_structured_codec_preregistration.md
Selected by HENRI_F5_CODEC=1; never imported by the production runner.
Kernels (additive, pure functions): qfhrr_kernels.py (make_fpb_base_ring,
fpb_power_wave, fhrr_bind, fhrr_unbind, fpb_ring_from_wave).

Construction (calibrated 2026-08-30, deterministic CPU -> exact on CUDA):
- char/role rings: full-range random Z_256 rings (token identity,
  discrimination), SHA-256-seeded, stateless wrt evaluation strings.
- position ring: iid narrow-band FPB comb, amplitude A = 0.6 rad
  (continuity: rho_wave 0.932, rho_ring 1.000 measured; far-string
  orthogonality ~ -0.001 measured). Position coordinate x = i (0-indexed),
  NOT normalized (the FPB orbit is translation-invariant).
- encoding: bundle in the complex domain
      w = sum_c exp(i * (theta_char(c) + x_c * theta_pos))     complex64 [D]
  then either keep the wave (encode_wave) or quantize to a uint8 ring
  (encode_text, consumer-compatible).
- W_task: compiled in the CONTINUOUS phase domain (directive eq. 1):
      W = sum_pairs Y_i * conj(X_i)   (unit-modulus FHRR sum)
  retrieval: X_test * W. G2 inner products are wave-domain cosines; ring
  quantization is a consumer-boundary convenience only (ring-domain modular
  arithmetic reproduces the F4 random-ring crosstalk pattern -> do not use
  ring-domain W_task for scoring).

Mechanism vs Run21 (FALSIFIED_AT_SCALE, commit 440f11d): Run21's position
binding was quantized collinear ring scaling round(p_i * q_P) mod 256 (a
single ring orbit, integer arithmetic, no sub-bin translation, no metric
anchor). F5 is the exact Fourier-domain FPB: continuous phase rotation per
dimension, exact homomorphism (cos 1.00000), narrow-band continuity, and a
separate full-range char channel for discrimination.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import math
from typing import Optional

import torch
import torch.nn as nn

from qfhrr_kernels import (
    K_PHASE,
    fpb_ring_from_wave,
    fhrr_bind,
    fhrr_unbind,
    make_fpb_base_ring,
)


class FPBStructuredCodec(nn.Module):
    """Deterministic FPB structured qFHRR text codec (Carrier F5)."""

    codec_name = "fpb_structured_qfhrr"
    codec_version = "f5-v1"

    def __init__(
        self,
        d_model: int = 65536,
        k_bins: int = 256,
        device: Optional[str] = None,
        position_amplitude: float = 0.6,
        position_seed: int = 20260830,
        max_cache_entries: int = 512,
        role: Optional[str] = None,
    ):
        super().__init__()
        if k_bins != 256:
            raise ValueError("F5 codec requires k_bins=256")
        if not (0.0 < position_amplitude <= 1.0):
            raise ValueError(f"position_amplitude must be in (0, 1], got {position_amplitude}")
        self.d_model = int(d_model)
        self.k_bins = int(k_bins)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.position_amplitude = float(position_amplitude)
        self.max_cache_entries = int(max_cache_entries)
        self._char_cpu: dict[str, torch.Tensor] = {}
        self._role_cpu: dict[str, torch.Tensor] = {}
        self._ring_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._wave_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        # Position comb: narrow-band FPB base ring (continuity channel).
        self._pos_ring = make_fpb_base_ring(
            d_model=self.d_model,
            k_bins=self.k_bins,
            seed=position_seed,
            amplitude=self.position_amplitude,
        )
        self._default_role = role

    # ------------------------------------------------------------------ #
    # Deterministic seeded rings
    # ------------------------------------------------------------------ #
    @staticmethod
    def _seed(namespace: str, value: str | int) -> int:
        raw = f"henri-f5:{namespace}:{value}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)

    def _full_ring_cpu(self, namespace: str, value: str | int) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(self._seed(namespace, value))
        return torch.randint(
            0, self.k_bins, (self.d_model,), dtype=torch.uint8, generator=g, device="cpu"
        )

    def _char_ring(self, c: str) -> torch.Tensor:
        ring = self._char_cpu.get(c)
        if ring is None:
            ring = self._full_ring_cpu("char", c)
            self._char_cpu[c] = ring
        return ring

    def _role_ring(self, role: str) -> torch.Tensor:
        ring = self._role_cpu.get(role)
        if ring is None:
            ring = self._full_ring_cpu("role", role)
            self._role_cpu[role] = ring
        return ring

    # ------------------------------------------------------------------ #
    # Encoding (wave domain primary; ring domain consumer-compatible)
    # ------------------------------------------------------------------ #
    def encode_wave(self, text: str, role: Optional[str] = None) -> torch.Tensor:
        """Bundle text into a complex64 [D] unit-ish wave.

        w = sum_i exp(i * (theta_char(c_i) + i * theta_pos)) [* role ring].
        Position coordinate x = i (0-indexed). Role phase added when given.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if not text:
            return torch.zeros(self.d_model, dtype=torch.complex64, device=self.device)
        role_phase = 0.0
        role = role if role is not None else self._default_role
        if role is not None:
            role_phase = self._role_ring(role).to(torch.float32) * (2.0 * math.pi / self.k_bins)
        theta_pos = self._pos_ring.to(torch.float32) * (2.0 * math.pi / self.k_bins)
        real = torch.zeros(self.d_model, dtype=torch.float32, device=self.device)
        imag = torch.zeros_like(real)
        for i, c in enumerate(text):
            theta_c = self._char_ring(c).to(torch.float32) * (2.0 * math.pi / self.k_bins)
            theta = theta_c + float(i) * theta_pos + role_phase
            real.add_(torch.cos(theta))
            imag.add_(torch.sin(theta))
        w = torch.complex(real, imag).to(self.device)
        # normalize to unit norm (bundles of unit phasors; keeps scoring stable)
        return w / (w.norm() + 1e-12)

    def encode_text(self, text: str, role: Optional[str] = None) -> torch.Tensor:
        """Ring-domain form (consumer-compatible uint8 [D]). Role-aware cache."""
        key = (text, role if role is not None else self._default_role)
        cached = self._ring_cache.get(key)
        if cached is not None:
            self._ring_cache.move_to_end(key)
            return cached
        ring = fpb_ring_from_wave(self.encode_wave(text, role=role), self.k_bins)
        self._ring_cache[key] = ring
        while len(self._ring_cache) > self.max_cache_entries:
            self._ring_cache.popitem(last=False)
        return ring

    # ------------------------------------------------------------------ #
    # Task functor (continuous phase domain) + retrieval
    # ------------------------------------------------------------------ #
    def compile_w_task(self, pairs: list[tuple[str, str]]) -> torch.Tensor:
        """W = sum_i Y_i * conj(X_i) in the complex phase domain (unit modulus)."""
        if not pairs:
            raise ValueError("compile_w_task requires >= 1 pair")
        w = torch.zeros(self.d_model, dtype=torch.complex64, device=self.device)
        for x, y in pairs:
            wx = self.encode_wave(x)
            wy = self.encode_wave(y)
            w = w + fhrr_bind(wy, torch.conj(wx))
        return w / (w.norm() + 1e-12)

    def retrieve(self, x: str, w_task: torch.Tensor) -> torch.Tensor:
        """Wave-domain retrieval: Psi_x * W. Returns complex64 [D] wave."""
        return fhrr_bind(self.encode_wave(x), w_task)

    # ------------------------------------------------------------------ #
    # Ring-domain compatibility (bind/unbind/sim) — Run21/legacy contract
    # ------------------------------------------------------------------ #
    def bind_hadamard(self, q_key: torch.Tensor, q_val: torch.Tensor) -> torch.Tensor:
        return (q_key.to(torch.int32) + q_val.to(torch.int32)) % self.k_bins

    def unbind_hadamard(self, q_bound: torch.Tensor, q_key: torch.Tensor) -> torch.Tensor:
        return (q_bound.to(torch.int32) - q_key.to(torch.int32)) % self.k_bins

    def compute_similarity(self, q1: torch.Tensor, q2: torch.Tensor) -> float:
        phase = (q1.to(torch.int32) - q2.to(torch.int32)) % self.k_bins
        return float(
            torch.cos(phase.to(torch.float32) * (2.0 * math.pi / self.k_bins))
            .mean()
            .item()
        )

    def wave_cosine(self, w1: torch.Tensor, w2: torch.Tensor) -> float:
        """Wave-domain cosine (directive G2 metric)."""
        num = float(torch.abs(torch.vdot(w1, w2)).item())
        den = float(w1.norm().item()) * float(w2.norm().item()) + 1e-12
        return num / den

    def geometry_metadata(self) -> dict[str, object]:
        return {
            "codec_name": self.codec_name,
            "codec_version": self.codec_version,
            "tokenizer": "character_codepoint",
            "position_binding": "fractional_power_binding_exp(i*x*theta_pos)",
            "position_amplitude_rad": self.position_amplitude,
            "position_index": "raw_character_index_0_based",
            "bundling": "complex_phase_sum_then_normalize",
            "w_task_domain": "continuous_phase",
            "d_model": self.d_model,
            "k_bins": self.k_bins,
            "zone_c_dependency": False,
        }


def make_fpb_codec(device: Optional[str] = None, **kw) -> FPBStructuredCodec:
    """Factory for the explicitly selected F5 path (HENRI_F5_CODEC=1)."""
    return FPBStructuredCodec(device=device, **kw)


def ring_to_real(codec, ring: torch.Tensor) -> torch.Tensor:
    """Map a codec ring to the real wave coordinate consumed by downstream
    probes. F5 rings are phase coordinates -> cos(phase) (Run21 convention);
    legacy codecs keep their historical linear diagnostic map exactly."""
    if isinstance(codec, FPBStructuredCodec):
        phase = ring.to(torch.float32) * (2.0 * math.pi / codec.k_bins)
        return torch.cos(phase)
    return ring.to(torch.float32) / (codec.k_bins - 1) * 2.0 - 1.0
