"""Phase 7.5 CPX: read-only complex third-family diagnostic sidecar.

Default-OFF diagnostic projection of the production real [num_blocks, 8]
Cl(3,0) UWE wave family into a complex flat [num_blocks] unit-modulus
phasor family. The typed container machinery (ComplexPhaseState,
PhaseProvenance, contract_summary) is REUSED from the production
phase_codec_adapter module. The one-way mapping is the corpus-prescribed
norm-preserving boundary (bank ca4bb787, manifest CPX):

    theta_k = atan2(sqrt(a_{k,4}^2 + a_{k,5}^2 + a_{k,6}^2), a_{k,0})
    q_k = floor((theta_k + pi) * 256 / (2 pi)) mod 256
    Psi_k = exp(i * 2 pi q_k / 256)          => |Psi_k| = 1

Cl(3,0) basis (verified invariant): indices 4, 5, 6 are the bivectors,
index 0 is the scalar. The sidecar NEVER writes into the action path,
NEVER influences policy, and emits telemetry only. NO reverse conversion
exists in this module.

Typed statuses: CPX_SIDECAR_OK | CPX_SIDECAR_DEGENERATE | CPX_SIDECAR_UNAVAILABLE.
Fail-closed: any anomaly -> UNAVAILABLE, no crash, no fallback.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Optional, Tuple

import torch

from phase_codec_adapter import (
    ComplexPhaseState,
    PhaseLayout,
    PhaseProvenance,
    PhaseRingState,
)

CPX_OK = "CPX_SIDECAR_OK"
CPX_DEGENERATE = "CPX_SIDECAR_DEGENERATE"
CPX_UNAVAILABLE = "CPX_SIDECAR_UNAVAILABLE"

_CLIFFORD_SCALAR_IDX = 0
_CLIFFORD_BIVECTOR_IDX = (4, 5, 6)
_PHASE_BINS = 256


def adapt_uwe_to_complex(wave: torch.Tensor) -> ComplexPhaseState:
    """One-way UWE -> complex phasor projection (per-block, unit modulus).

    Args:
        wave: real [num_blocks, 8] Cl(3,0) wave with unit-norm rows.

    Returns:
        ComplexPhaseState with complex64 values [num_blocks] (unit modulus).

    Raises:
        TypeError / PhaseCodecError on malformed input (never silent).
    """
    if not isinstance(wave, torch.Tensor):
        raise TypeError("wave must be a torch.Tensor")
    if wave.ndim != 2 or wave.shape[-1] != 8:
        raise TypeError(f"wave must be [K, 8] real, got shape {tuple(wave.shape)}")
    if wave.is_complex():
        raise TypeError("wave must be real-valued")
    if not torch.isfinite(wave).all():
        raise ValueError("wave contains NaN/Inf")

    scalar = wave[:, _CLIFFORD_SCALAR_IDX]
    biv = wave[:, list(_CLIFFORD_BIVECTOR_IDX)]
    theta = torch.atan2(torch.sqrt((biv ** 2).sum(dim=-1)), scalar)
    # Map [-pi, pi] -> [0, 256) bins.
    q = torch.floor((theta + math.pi) * (_PHASE_BINS / (2.0 * math.pi)))
    q = torch.remainder(q, _PHASE_BINS).to(torch.uint8)
    phase = q.to(torch.float64) * (2.0 * math.pi / _PHASE_BINS)
    values = torch.polar(torch.ones_like(phase), phase).to(torch.complex64)

    provenance = PhaseProvenance(
        encoder="phase_codec_adapter",
        encoder_version="stage1",
        source_representation="clifford_k8_real",
        transform="atan2_bivector_norm_to_unit_phasor",
        information_loss="lossy_one_way",
        reconstruction_error=0.0,
        source_uri="",
        notes="CPX per-block phasor projection; no reverse conversion",
    )
    return ComplexPhaseState(
        values=values,
        layout=PhaseLayout.FLAT_D,
        provenance=provenance,
        modulus_tolerance=1e-5,
    )


def evaluate_complex_sidecar(
    wave: Optional[torch.Tensor],
) -> Tuple[Dict[str, Any], str]:
    """Compute read-only complex-sidecar diagnostics from the live wave.

    Returns:
        (diagnostics_dict, status). status is CPX_OK, CPX_DEGENERATE, or
        CPX_UNAVAILABLE. UNAVAILABLE never crashes and never mutates state.
    """
    if wave is None:
        return {"status": CPX_UNAVAILABLE}, CPX_UNAVAILABLE
    try:
        flat_view = wave.reshape(-1)
        flat_sha = hashlib.sha256(
            flat_view.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        z = adapt_uwe_to_complex(wave)
        values = z.values
        coherence_r = float(torch.abs(values.mean()).item())
        phase = torch.angle(values)
        # Histogram entropy of the quantized phase bins (max ln(256) nats).
        q = torch.floor((phase + math.pi) * (_PHASE_BINS / (2.0 * math.pi)))
        q = torch.remainder(q, _PHASE_BINS).to(torch.int64)
        hist = torch.bincount(q.flatten(), minlength=_PHASE_BINS).to(torch.float64)
        probs = hist / hist.sum()
        nz = probs[probs > 0]
        entropy = float(-(nz * torch.log(nz)).sum().item())
        modulus_error = float(
            torch.max(torch.abs(torch.abs(values) - 1.0)).item()
        )
        diag = {
            "status": CPX_OK,
            "layout": z.layout.value,
            "dtype": str(values.dtype),
            "device": str(values.device),
            "num_blocks": int(values.numel()),
            "coherence_r": round(coherence_r, 6),
            "phase_entropy_nats": round(entropy, 6),
            "modulus_error": round(modulus_error, 8),
            "flat_real_sha256": flat_sha,
            "read_only": True,
        }
        status = CPX_OK
        if coherence_r > 0.999:
            diag["status"] = CPX_DEGENERATE
            status = CPX_DEGENERATE
        return diag, status
    except Exception:
        return {"status": CPX_UNAVAILABLE}, CPX_UNAVAILABLE
