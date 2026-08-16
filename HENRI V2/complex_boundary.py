"""Phase 8.14 — Complex boundary wiring (default-OFF, additive).

VARIANT B (corrected): un-realify the encoder's NATIVE complex wave.

The production encoder (henri_vision_encoder.HENRIVisionEncoder.encode_grid)
is already a complex encoder: it accumulates a D/2-dim complex
superposition and projects to real D via concat([real, imag], -1) + L2
normalization. The complex boundary simply RE-PAIRS that projection:

    z = w[:D/2] + 1j * w[D/2:]           (exact, up to float rounding)

so the 8.11 NativeComplexWaveTransition acts on the TRUE carrier phases,
and the real representation is rebuilt (concat real/imag) ONLY at the
environment edge. Egress is exact by construction; the complex inner
product on re-paired waves equals the legacy real cosine on the same
waves (no discrimination loss).

VARIANT A (value-lift, FALSIFIED locally 2026-08-16): mapping real values
to surrogate phases (pi/2)*w/||w||_inf re-encodes instead of wiring; color
cos 0.4555 vs legacy 0.1911 (2.4x worse), shared-support 0.3199 vs 0.0016
(200x worse). Killed by the pre-registered local probe; functions kept for
the audit trail.
"""

import os

import torch

_ENABLED = os.environ.get("HENRI_ARC_COMPLEX_BOUNDARY", "0") == "1"


def complex_boundary_enabled() -> bool:
    """Return whether the complex boundary path is active (default-OFF)."""
    return _ENABLED


@torch.no_grad()
def un_realify(w: torch.Tensor) -> torch.Tensor:
    """Re-pair the encoder's concat([real, imag]) projection.

    Args:
        w: real tensor [..., D] (even D; first D/2 = real parts, last
           D/2 = imaginary parts of the native complex wave).

    Returns:
        Complex tensor [..., D/2] = w[..., :D/2] + 1j * w[..., D/2:].
    """
    half = w.numel() // 2
    wf = w.reshape(-1).to(torch.float32)
    re_part = wf[:half]
    im_part = wf[half:]
    return torch.complex(re_part, im_part)


@torch.no_grad()
def re_realify(z: torch.Tensor) -> torch.Tensor:
    """Rebuild the real [..., 2*D] representation at the environment edge.

    Args:
        z: complex tensor [..., D].

    Returns:
        Real tensor [..., 2*D] = concat([z.real, z.imag], -1), L2-normalized
        per the production encoder convention.
    """
    return torch.nn.functional.normalize(
        torch.cat([z.real, z.imag], dim=-1), p=2, dim=-1
    )


@torch.no_grad()
def complex_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Complex-aware cosine similarity (8.12 runner convention)."""
    a, b = a.reshape(-1), b.reshape(-1)
    denom = (a.abs().norm() * b.abs().norm()).clamp_min(1e-12)
    return float(torch.real(torch.dot(a, torch.conj(b))) / denom)


@torch.no_grad()
def complex_cycle(
    w: torch.Tensor,
    transition,
    action_idx: int,
) -> torch.Tensor:
    """Full complex-boundary cycle: re-pair -> rotate -> re-realify.

    Args:
        w: real wave [1, 8192, 8] (production representation, D even).
        transition: NativeComplexWaveTransition with dimension == D/2.
        action_idx: integer action index for the phase rotation.

    Returns:
        Real wave [1, 8192, 8] after the complex transition.
    """
    z = un_realify(w.reshape(-1))              # [D/2] complex
    z_next = transition.forward_complex(z, action_idx)
    return re_realify(z_next).reshape(w.shape)


# -- VARIANT A (FALSIFIED, audit trail) --------------------------------------
@torch.no_grad()
def value_lift(w: torch.Tensor) -> torch.Tensor:
    """FALSIFIED variant A: surrogate-phase lift (kept for audit trail)."""
    wf = w.to(torch.float32)
    norm_inf = wf.abs().amax().clamp_min(1e-12)
    phi = (torch.pi / 2.0) * (wf / norm_inf)
    return torch.polar(wf.abs(), phi)


@torch.no_grad()
def value_egress(z: torch.Tensor) -> torch.Tensor:
    """FALSIFIED variant A egress (kept for audit trail)."""
    return torch.sign(z.imag) * z.abs()
