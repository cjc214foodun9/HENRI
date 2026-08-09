"""Explicit phase-state contracts for the HENRI digital-twin boundary.

This module is an uncalled experimental adapter. It does not alter the HENRI
planner, Sagnac score, Zone C, decoder, Hopfield path, or benchmark path.

The adapter keeps these representations distinct:

* ``PhaseRingState``: values in Z_256, with either a flat [D] layout or a
  Clifford-channel [K, 8] layout.
* ``ComplexPhaseState``: one unit-modulus complex phasor per declared channel.
* projective states [B, N, d]: unsupported here and never flattened silently.

A unit modulus per channel is not a global unit-vector norm. For D channels,
that norm is sqrt(D). This distinction is part of the public contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Mapping

import torch


DEFAULT_MODULUS = 256
CLIFFORD_COMPONENTS = 8
HALF_BIN_RADIANS = math.pi / DEFAULT_MODULUS


class PhaseCodecError(ValueError):
    """Base class for invalid phase-state contracts."""


class UnsupportedProjectiveStateError(PhaseCodecError):
    """Raised instead of flattening a projective [B, N, d] state."""


class AmbiguousCliffordProjectionError(PhaseCodecError):
    """Raised when no explicit Clifford-to-phase policy is supplied."""


class LossyCliffordProjectionError(PhaseCodecError):
    """Raised when a lossy Clifford projection is requested without approval."""


class PhaseLayout(str, Enum):
    """Supported storage layouts for phase channels."""

    FLAT_D = "flat_d"
    CLIFFORD_CHANNELS_K8 = "clifford_channels_k8"


@dataclass(frozen=True)
class PhaseProvenance:
    """Immutable provenance and information-loss record."""

    encoder: str
    encoder_version: str
    source_representation: str
    transform: str
    information_loss: str = "none"
    reconstruction_error: float | None = None
    source_uri: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class PhaseRingState:
    """A validated tensor of integer phase codes in Z_modulus."""

    values: torch.Tensor
    layout: PhaseLayout
    provenance: PhaseProvenance
    modulus: int = DEFAULT_MODULUS
    normalization: str = "ring_codes"
    quantization: str = "integer_modular"

    def __post_init__(self) -> None:
        if not isinstance(self.values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor")
        if self.modulus < 2 or self.modulus > 256:
            raise ValueError("modulus must be in [2, 256]")
        if self.values.ndim == 3:
            raise UnsupportedProjectiveStateError(
                "projective [B, N, d] tensors must not be flattened into phase rings"
            )
        if self.layout is PhaseLayout.FLAT_D and self.values.ndim != 1:
            raise PhaseCodecError("FLAT_D requires a rank-1 [D] tensor")
        if self.layout is PhaseLayout.CLIFFORD_CHANNELS_K8:
            if self.values.ndim != 2 or self.values.shape[-1] != CLIFFORD_COMPONENTS:
                raise PhaseCodecError("CLIFFORD_CHANNELS_K8 requires a [K, 8] tensor")
        if self.values.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("phase-ring values must use an integer tensor dtype")
        if self.values.numel() and (
            int(torch.min(self.values).item()) < 0
            or int(torch.max(self.values).item()) >= self.modulus
        ):
            raise PhaseCodecError("phase-ring values must be within the declared modulus")

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.values.shape)

    @property
    def channel_count(self) -> int:
        return int(self.values.numel())

    @property
    def device(self) -> torch.device:
        return self.values.device

    def describe(self) -> dict[str, Any]:
        """Return contract metadata without copying tensor data."""
        return {
            "representation": "PhaseRingState",
            "shape": list(self.shape),
            "layout": self.layout.value,
            "dtype": str(self.values.dtype),
            "device": str(self.device),
            "normalization": self.normalization,
            "quantization": self.quantization,
            "modulus": self.modulus,
            "channel_count": self.channel_count,
            "information_loss": self.provenance.information_loss,
            "reconstruction_error": self.provenance.reconstruction_error,
            "encoder": self.provenance.encoder,
            "encoder_version": self.provenance.encoder_version,
            "transform": self.provenance.transform,
        }


@dataclass(frozen=True)
class ComplexPhaseState:
    """A validated unit-modulus complex state with an explicit layout."""

    values: torch.Tensor
    layout: PhaseLayout
    provenance: PhaseProvenance
    normalization: str = "per_channel_unit_modulus"
    quantization: str = "continuous_phase"
    modulus_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if not isinstance(self.values, torch.Tensor):
            raise TypeError("values must be a torch.Tensor")
        if not self.values.is_complex():
            raise TypeError("complex phase values must use a complex tensor dtype")
        if self.values.ndim == 3:
            raise UnsupportedProjectiveStateError(
                "projective [B, N, d] tensors must not be flattened into complex phase states"
            )
        if self.layout is PhaseLayout.FLAT_D and self.values.ndim != 1:
            raise PhaseCodecError("FLAT_D requires a rank-1 [D] tensor")
        if self.layout is PhaseLayout.CLIFFORD_CHANNELS_K8:
            if self.values.ndim != 2 or self.values.shape[-1] != CLIFFORD_COMPONENTS:
                raise PhaseCodecError("CLIFFORD_CHANNELS_K8 requires a [K, 8] tensor")
        if self.modulus_tolerance < 0:
            raise ValueError("modulus_tolerance must be non-negative")
        if self.values.numel():
            modulus_error = torch.max(torch.abs(torch.abs(self.values) - 1.0)).item()
            if modulus_error > self.modulus_tolerance:
                raise PhaseCodecError(
                    f"complex phase values must have unit modulus; error={modulus_error}"
                )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.values.shape)

    @property
    def channel_count(self) -> int:
        return int(self.values.numel())

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def global_l2_norm(self) -> float:
        """Return sqrt(channel_count) for unit-modulus channels."""
        return math.sqrt(self.channel_count)

    def describe(self) -> dict[str, Any]:
        """Return contract metadata without copying tensor data."""
        return {
            "representation": "ComplexPhaseState",
            "shape": list(self.shape),
            "layout": self.layout.value,
            "dtype": str(self.values.dtype),
            "device": str(self.device),
            "normalization": self.normalization,
            "quantization": self.quantization,
            "channel_count": self.channel_count,
            "per_channel_modulus": 1.0,
            "global_l2_norm": self.global_l2_norm,
            "information_loss": self.provenance.information_loss,
            "reconstruction_error": self.provenance.reconstruction_error,
            "encoder": self.provenance.encoder,
            "encoder_version": self.provenance.encoder_version,
            "transform": self.provenance.transform,
        }


def _coerce_ring(value: PhaseRingState | torch.Tensor) -> PhaseRingState:
    if isinstance(value, PhaseRingState):
        return value
    if isinstance(value, torch.Tensor):
        if value.ndim == 3:
            raise UnsupportedProjectiveStateError(
                "projective [B, N, d] tensors must not be flattened"
            )
        raise TypeError(
            "raw tensors require an explicit PhaseRingState layout and provenance"
        )
    raise TypeError("expected PhaseRingState")


def _check_compatible(left: PhaseRingState, right: PhaseRingState) -> None:
    if left.layout is not right.layout or left.shape != right.shape:
        raise PhaseCodecError("phase-ring operands must have identical layout and shape")
    if left.modulus != right.modulus:
        raise PhaseCodecError("phase-ring operands must have identical moduli")
    if left.device != right.device:
        raise PhaseCodecError("phase-ring operands must be on the same device")


def phase_ring_to_complex(
    state: PhaseRingState | torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> ComplexPhaseState:
    """Decode Z_modulus codes to unit-modulus complex phasors."""
    ring = _coerce_ring(state)
    if output_dtype is None:
        output_dtype = torch.complex64
    if output_dtype not in {torch.complex64, torch.complex128}:
        raise TypeError("output_dtype must be torch.complex64 or torch.complex128")
    phase = ring.values.to(torch.float64) * (2.0 * math.pi / ring.modulus)
    values = torch.polar(torch.ones_like(phase, dtype=torch.float64), phase)
    values = values.to(output_dtype)
    provenance = PhaseProvenance(
        encoder=ring.provenance.encoder,
        encoder_version=ring.provenance.encoder_version,
        source_representation="PhaseRingState",
        transform="Z_modulus_to_unit_phasor",
        information_loss="none",
        reconstruction_error=0.0,
        source_uri=ring.provenance.source_uri,
        notes=ring.provenance.notes,
    )
    return ComplexPhaseState(
        values=values,
        layout=ring.layout,
        provenance=provenance,
    )


def complex_to_phase_ring(
    state: ComplexPhaseState,
    *,
    modulus: int = DEFAULT_MODULUS,
) -> PhaseRingState:
    """Quantize unit-modulus complex phasors to Z_modulus codes."""
    if not isinstance(state, ComplexPhaseState):
        raise TypeError("expected ComplexPhaseState")
    if modulus < 2 or modulus > 256:
        raise ValueError("modulus must be in [2, 256]")
    phase = torch.remainder(torch.angle(state.values), 2.0 * math.pi)
    codes = torch.remainder(
        torch.round(phase * (modulus / (2.0 * math.pi))), modulus
    ).to(torch.uint8)
    provenance = PhaseProvenance(
        encoder=state.provenance.encoder,
        encoder_version=state.provenance.encoder_version,
        source_representation="ComplexPhaseState",
        transform="unit_phasor_to_Z_modulus",
        information_loss="quantization_only",
        reconstruction_error=HALF_BIN_RADIANS if modulus == DEFAULT_MODULUS else math.pi / modulus,
        source_uri=state.provenance.source_uri,
        notes=state.provenance.notes,
    )
    return PhaseRingState(
        values=codes,
        layout=state.layout,
        provenance=provenance,
        modulus=modulus,
    )


def bind_phase_rings(left: PhaseRingState, right: PhaseRingState) -> PhaseRingState:
    """Perform exact element-wise modular addition in Z_modulus."""
    _check_compatible(left, right)
    values = torch.remainder(
        left.values.to(torch.int32) + right.values.to(torch.int32), left.modulus
    ).to(torch.uint8)
    provenance = PhaseProvenance(
        encoder="phase_codec_adapter",
        encoder_version="stage1",
        source_representation="PhaseRingState",
        transform="modular_bind",
        information_loss="none",
        reconstruction_error=0.0,
        notes="element-wise Z_modulus addition",
    )
    return PhaseRingState(values, left.layout, provenance, left.modulus)


def unbind_phase_rings(bound: PhaseRingState, key: PhaseRingState) -> PhaseRingState:
    """Perform exact element-wise modular subtraction in Z_modulus."""
    _check_compatible(bound, key)
    values = torch.remainder(
        bound.values.to(torch.int32) - key.values.to(torch.int32), bound.modulus
    ).to(torch.uint8)
    provenance = PhaseProvenance(
        encoder="phase_codec_adapter",
        encoder_version="stage1",
        source_representation="PhaseRingState",
        transform="modular_unbind",
        information_loss="none",
        reconstruction_error=0.0,
        notes="element-wise Z_modulus subtraction",
    )
    return PhaseRingState(values, bound.layout, provenance, bound.modulus)


def circular_phase_error(
    actual: PhaseRingState,
    reconstructed: PhaseRingState,
) -> torch.Tensor:
    """Return per-channel circular phase error in radians."""
    _check_compatible(actual, reconstructed)
    diff = torch.remainder(
        actual.values.to(torch.int32) - reconstructed.values.to(torch.int32),
        actual.modulus,
    )
    circular_bins = torch.minimum(diff, actual.modulus - diff)
    return circular_bins.to(torch.float64) * (2.0 * math.pi / actual.modulus)


def reject_projective_state(value: torch.Tensor) -> None:
    """Raise a typed error for a projective [B, N, d] tensor."""
    if not isinstance(value, torch.Tensor):
        raise TypeError("expected a torch.Tensor")
    if value.ndim == 3:
        raise UnsupportedProjectiveStateError(
            "projective states [B, N, d] require a separate approved adapter"
        )
    raise PhaseCodecError("value is not a projective [B, N, d] tensor")


def project_clifford_to_phase(*_: Any, projection_policy: str | None = None, **__: Any) -> None:
    """Reject implicit or unapproved Clifford-to-phase dimensional reduction."""
    if projection_policy is None:
        raise AmbiguousCliffordProjectionError(
            "provide an explicitly approved channel-wise Clifford phase policy"
        )
    raise LossyCliffordProjectionError(
        f"projection policy {projection_policy!r} is lossy and is not approved in Stage 1"
    )


def contract_summary(state: PhaseRingState | ComplexPhaseState) -> Mapping[str, Any]:
    """Return a stable, serializable contract summary."""
    if not isinstance(state, (PhaseRingState, ComplexPhaseState)):
        raise TypeError("expected a PhaseRingState or ComplexPhaseState")
    return state.describe()


__all__ = [
    "AmbiguousCliffordProjectionError",
    "ComplexPhaseState",
    "DEFAULT_MODULUS",
    "LossyCliffordProjectionError",
    "PhaseCodecError",
    "PhaseLayout",
    "PhaseProvenance",
    "PhaseRingState",
    "UnsupportedProjectiveStateError",
    "bind_phase_rings",
    "circular_phase_error",
    "complex_to_phase_ring",
    "contract_summary",
    "phase_ring_to_complex",
    "project_clifford_to_phase",
    "reject_projective_state",
    "unbind_phase_rings",
]
