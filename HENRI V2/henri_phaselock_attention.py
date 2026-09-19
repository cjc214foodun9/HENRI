#!/usr/bin/env python3
"""Continuous phase-locking attention sidecar -- amendment reconciliation carrier.

SOURCE
    HENRI-ARCH-2026-11-COMPUTE-ATTENTION-DYNAMICS (amendment document 3),
    "The Selective Attention Head" (CPL-AE) and its production module
    HENRIPhaseLockingAttention.

CLAIM UNDER TEST
    Awareness is the broad uncollapsed high-dimensional phase space
    (Psi in S^{D-1}); attention is a non-linear low-dimensional phase-lock that
    crystallizes a small context-critical slice (<1%) into actionable state. The
    "99% noise" is not garbage: it is dormant superposed associations, and it can
    be mined by (a) O(D log D) holographic unbinding, (b) a phase-coherence mask,
    (c) a Modern Hopfield cleanup at the calibrated temperature.

TWO DEFECTS FOUND IN THE DOCUMENT'S MODULE (both measured below)
    D4  MASK-THEN-NORMALIZE PRODUCES NaN.
        The module computes local coherence over tiles of 64 and masks at
        r_threshold = 0.707, then calls F.normalize(z * mask).
        For independent phases the local order parameter is ~1/sqrt(64) = 0.125,
        FAR below 0.707, so the mask zeroes essentially everything and
        normalize(0) yields NaN. The threshold is asserted, never calibrated
        against the null distribution. This carrier measures the occupancy at
        the asserted threshold and at a calibrated one, and fails closed instead
        of returning NaN.
    D5  DOC/CODE MISMATCH ON DTYPE.
        The prose says the codebook is stored "in FP16 complex representation to
        conserve VRAM (2 GB footprint)", but the code does
        `.to(torch.complex64)`. A reader checking memory would be misled.

DESIGN
    Default-OFF diagnostic sidecar. It is NOT wired into any planner, has no
    trainable parameters, and returns a typed fail-closed state instead of NaN.
    A dead-input control (all-zero awareness) MUST fail it -- an attention
    mechanism that "attends" to nothing is not attending.

SCOPE
    This measures an ENCODER-LEVEL retrieval property (can a buried attractor be
    recovered from superposed cross-talk?). It is not a capability claim, not a
    planner change, and not evidence about benchmark performance.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_DIM = 4096
DEFAULT_TILE = 64
# The document's asserted threshold. Kept as a NAMED CONSTANT so the measurement
# can show what it does, rather than silently replacing it.
DOC_THRESHOLD = 0.707
DOC_TEMPERATURE = 0.038316

FAIL_CLOSED_OCCUPANCY = 0.0


def _phasors(shape: Tuple[int, ...], seed: int, device: str = "cpu") -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    ph = torch.rand(*shape, generator=g, device="cpu") * (2.0 * math.pi)
    return torch.polar(torch.ones_like(ph), ph).to(device)


class PhaseLockAttention(nn.Module):
    """Two-stage extraction: holographic unbinding -> coherence mask -> Hopfield.

    Default-OFF: `enabled=False` performs no masking and no Hopfield step, so the
    sidecar cannot influence anything unless a caller deliberately enables it.
    """

    def __init__(self, dim: int = DEFAULT_DIM, tile: int = DEFAULT_TILE,
                 temperature: float = DOC_TEMPERATURE,
                 coherence_threshold: Optional[float] = None,
                 enabled: bool = False, seed: int = 20260918,
                 zero_field_eps: float = 1e-12,
                 device: str = "cpu") -> None:
        super().__init__()
        if dim % tile != 0:
            raise ValueError(f"dim {dim} must be divisible by tile {tile}")
        self.dim = int(dim)
        self.tile = int(tile)
        self.tau = float(temperature)
        # None => calibrate from the null distribution at run time (correct).
        # A float => use it verbatim (this is how the document's 0.707 is tested).
        self.coherence_threshold = coherence_threshold
        self.enabled = bool(enabled)
        self.zero_field_eps = float(zero_field_eps)
        self.device = device

        self.register_buffer(
            "memory_codebook",
            _phasors((256, self.dim), seed + 3, device))

    # ------------------------------------------------------------ primitives
    def unbind(self, psi_aware: torch.Tensor,
               psi_query: torch.Tensor) -> torch.Tensor:
        """Holographic unbinding: z = Psi_aware (x) conj(Psi_query), O(D log D)."""
        fa = torch.fft.fft(psi_aware, dim=-1)
        fq = torch.fft.fft(psi_query, dim=-1)
        return torch.fft.ifft(fa * torch.conj(fq), dim=-1)

    def local_coherence(self, z: torch.Tensor) -> torch.Tensor:
        """Kuramoto order parameter over tiles: r = |mean(exp(i theta))|."""
        b = z.shape[0]
        ph = torch.angle(z).view(b, -1, self.tile)
        phasors = torch.polar(torch.ones_like(ph), ph)
        r = torch.abs(torch.mean(phasors, dim=-1, keepdim=True))
        return r  # [B, dim//tile, 1]

    def null_coherence(self, n: int = 32) -> torch.Tensor:
        """r distribution for INDEPENDENT phases -- the calibration reference.

        Analytic expectation is 1/sqrt(tile); this measures it so the calibrated
        threshold is derived from data on the same dimension, not asserted.
        """
        g = torch.Generator(device="cpu").manual_seed(4242)
        ph = torch.rand(n, self.dim, generator=g, device="cpu") * (2.0 * math.pi)
        z = torch.polar(torch.ones_like(ph), ph).to(self.device)
        return self.local_coherence(z.reshape(n, self.dim)).reshape(-1)

    # -------------------------------------------------------------- forward
    def forward(self, psi_aware: torch.Tensor, psi_query: torch.Tensor,
                quantile: float = 0.99) -> Dict[str, object]:
        """Return a telemetry dict. Never returns NaN: fails closed instead."""
        if psi_aware.shape != psi_query.shape:
            raise ValueError(f"shape mismatch: {tuple(psi_aware.shape)} vs "
                             f"{tuple(psi_query.shape)}")
        if psi_aware.dim() != 2:
            raise ValueError("expected [batch, dim]")
        # DIM VALIDATION (defect D9, exposed by my own contract test). The first
        # version checked only that awareness and query matched EACH OTHER, so a
        # correctly-matched pair of the WRONG width (e.g. 4096-dim input into a
        # 1024-dim module) passed validation and then failed deep inside the tile
        # reshape with "shape '[1, 1024]' is invalid for input of size 4096" --
        # an unactionable error from the wrong layer. A fail-closed module must
        # reject at its own boundary with a message naming the contract.
        if psi_aware.shape[-1] != self.dim:
            raise ValueError(
                f"last-dim mismatch: input has {psi_aware.shape[-1]}, module "
                f"contract is dim={self.dim} (tile={self.tile}, "
                f"tiles={self.dim // self.tile})")

        z_raw = self.unbind(psi_aware, psi_query)

        # ZERO-FIELD VACUOUS-COHERENCE GUARD (defect D6, MEASURED 2026-09-18).
        # torch.angle(0.0) == 0.0 for every entry, so an all-zero field has
        # PERFECTLY ALIGNED phases: the local Kuramoto order parameter is 1.0 and
        # the coherence mask passes every tile. Measured on a dead input:
        # occupancy = 1.0 with fail_closed = False -- the mechanism "succeeded"
        # on nothing. A coherence test that a null field passes is VACUOUS, so
        # the mechanism must reject on negligible power before it measures
        # coherence at all. This guard is the dead-input negative control the
        # document's module lacks.
        power = float(z_raw.abs().pow(2).mean())
        if not math.isfinite(power) or power <= self.zero_field_eps:
            return {
                "enabled": True,
                "occupancy": None,
                "threshold": self.coherence_threshold,
                "fail_closed": True,
                "reason": "ZERO_FIELD_VACUOUS_COHERENCE",
                "input_power": power,
                "note": ("a null field has angle==0 everywhere, so coherence is "
                         "1.0 and the mask would pass everything"),
            }

        r = self.local_coherence(z_raw)

        if not self.enabled:
            # Default-OFF: no mask, no retrieval. Report the null calibration so
            # the sidecar is still informative in diagnosis mode.
            null = self.null_coherence()
            return {
                "enabled": False,
                "occupancy": None,
                "threshold": self.coherence_threshold,
                "null_r_mean": float(null.mean()),
                "null_r_std": float(null.std()),
                "null_analytic_1_over_sqrt_tile": 1.0 / math.sqrt(self.tile),
                "z_raw_norm": float(z_raw.norm(dim=-1).mean()),
                "fail_closed": False,
                "reason": "DISABLED_DEFAULT_OFF",
            }

        if self.coherence_threshold is None:
            null = self.null_coherence()
            thr = float(torch.quantile(null, quantile))
            thr_source = f"calibrated_null_q{quantile:g}"
        else:
            thr = float(self.coherence_threshold)
            thr_source = "asserted_literal"

        mask = (r >= thr)
        occupancy = float(mask.to(torch.float32).mean())

        if occupancy <= FAIL_CLOSED_OCCUPANCY:
            # THE DOCUMENT'S BUG, HANDLED: mask-then-normalize would now be
            # normalize(0) -> NaN. Fail closed with a typed reason instead.
            return {
                "enabled": True,
                "occupancy": occupancy,
                "threshold": thr,
                "threshold_source": thr_source,
                "fail_closed": True,
                "reason": "MASK_EMPTY_WOULD_DIVIDE_BY_ZERO",
                "null_r_mean": float(self.null_coherence().mean()),
                "z_raw_norm": float(z_raw.norm(dim=-1).mean()),
            }

        mask_full = mask.expand(-1, -1, self.tile).reshape(
            z_raw.shape[0], self.dim)
        z_filt = torch.where(mask_full, z_raw,
                             torch.zeros_like(z_raw))
        z_filt = F.normalize(z_filt, p=2, dim=-1)

        sim = torch.real(z_filt @ self.memory_codebook.conj().T)
        w = F.softmax(sim / self.tau, dim=-1)
        crystallized = F.normalize(w.to(torch.complex64) @ self.memory_codebook,
                                   p=2, dim=-1)

        sig = crystallized.norm(dim=-1) ** 2
        noi = (z_raw - crystallized).norm(dim=-1) ** 2
        snr_db = (10.0 * torch.log10(sig / (noi + 1e-8))).mean()

        return {
            "enabled": True,
            "occupancy": occupancy,
            "threshold": thr,
            "threshold_source": thr_source,
            "fail_closed": False,
            "snr_db": float(snr_db),
            "retrieved_index": int(torch.argmax(w[0]).item()),
            "max_weight": float(w.max()),
            "z_raw_norm": float(z_raw.norm(dim=-1).mean()),
        }


# --------------------------------------------------------------- calibration
def superposed_field(key_value_pairs, seed: int = 7, device: str = "cpu"):
    """Build Psi_aware = normalize( (1/sqrt(M)) SUM_k X_k (x) Y_k ).

    This is the document's own superposition model (section 1.1): a single
    awareness state holding M bound key-value engrams, where a probe with X_k
    should retrieve Y_k plus cross-talk.
    """
    xs, ys = [], []
    for i, (x, y) in enumerate(key_value_pairs):
        xs.append(x)
        ys.append(y)
    X = torch.stack(xs)
    Y = torch.stack(ys)
    M = X.shape[0]
    acc = torch.zeros_like(X[0])
    for k in range(M):
        acc = acc + torch.fft.ifft(torch.fft.fft(X[k]) * torch.fft.fft(Y[k]))
    return F.normalize(acc / math.sqrt(M), p=2, dim=-1), X, Y
