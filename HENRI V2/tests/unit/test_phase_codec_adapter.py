"""Contract tests for the isolated Stage 1 phase codec adapter."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from phase_codec_adapter import (
    AmbiguousCliffordProjectionError,
    ComplexPhaseState,
    LossyCliffordProjectionError,
    PhaseLayout,
    PhaseProvenance,
    PhaseRingState,
    UnsupportedProjectiveStateError,
    bind_phase_rings,
    circular_phase_error,
    complex_to_phase_ring,
    phase_ring_to_complex,
    project_clifford_to_phase,
    reject_projective_state,
    unbind_phase_rings,
)


@pytest.fixture
def provenance() -> PhaseProvenance:
    return PhaseProvenance(
        encoder="test",
        encoder_version="1",
        source_representation="test_ring",
        transform="fixture",
    )


def make_ring(values: torch.Tensor, provenance: PhaseProvenance) -> PhaseRingState:
    return PhaseRingState(
        values=values.to(torch.uint8),
        layout=PhaseLayout.FLAT_D if values.ndim == 1 else PhaseLayout.CLIFFORD_CHANNELS_K8,
        provenance=provenance,
    )


def test_exact_modular_bind_unbind(provenance: PhaseProvenance) -> None:
    left = make_ring(torch.tensor([0, 1, 127, 255]), provenance)
    key = make_ring(torch.tensor([1, 255, 129, 2]), provenance)
    bound = bind_phase_rings(left, key)
    recovered = unbind_phase_rings(bound, key)
    assert torch.equal(recovered.values, left.values)
    assert bound.values.tolist() == [1, 0, 0, 1]


def test_ring_complex_ring_half_bin_bound(provenance: PhaseProvenance) -> None:
    values = torch.arange(256, dtype=torch.uint8)
    ring = make_ring(values, provenance)
    complex_state = phase_ring_to_complex(ring)
    assert torch.max(torch.abs(torch.abs(complex_state.values) - 1.0)).item() <= 1e-6
    recovered = complex_to_phase_ring(complex_state)
    error = circular_phase_error(ring, recovered)
    assert float(error.max()) <= math.pi / 256.0 + 1e-6


def test_layout_and_global_norm_are_explicit(provenance: PhaseProvenance) -> None:
    ring = PhaseRingState(
        torch.zeros((4, 8), dtype=torch.uint8),
        PhaseLayout.CLIFFORD_CHANNELS_K8,
        provenance,
    )
    state = phase_ring_to_complex(ring)
    assert state.shape == (4, 8)
    assert state.global_l2_norm == math.sqrt(32.0)
    assert state.describe()["per_channel_modulus"] == 1.0


def test_metadata_preserves_device_dtype_shape_and_loss(provenance: PhaseProvenance) -> None:
    ring = make_ring(torch.zeros(16, dtype=torch.uint8), provenance)
    state = phase_ring_to_complex(ring, output_dtype=torch.complex128)
    description = state.describe()
    assert description["shape"] == [16]
    assert description["dtype"] == "torch.complex128"
    assert description["device"] == "cpu"
    assert description["information_loss"] == "none"


def test_projective_tensor_is_rejected() -> None:
    with pytest.raises(UnsupportedProjectiveStateError):
        reject_projective_state(torch.zeros((2, 3, 4)))
    with pytest.raises(UnsupportedProjectiveStateError):
        PhaseRingState(
            torch.zeros((2, 3, 4), dtype=torch.uint8),
            PhaseLayout.FLAT_D,
            PhaseProvenance("test", "1", "x", "y"),
        )


def test_ambiguous_and_lossy_clifford_projection_are_rejected() -> None:
    with pytest.raises(AmbiguousCliffordProjectionError):
        project_clifford_to_phase(torch.zeros((4, 8)))
    with pytest.raises(LossyCliffordProjectionError):
        project_clifford_to_phase(torch.zeros((4, 8)), projection_policy="block_angle")


def test_invalid_complex_modulus_is_rejected(provenance: PhaseProvenance) -> None:
    with pytest.raises(ValueError):
        ComplexPhaseState(
            values=torch.ones(8, dtype=torch.complex64) * 2,
            layout=PhaseLayout.FLAT_D,
            provenance=provenance,
        )


def test_dense_production_square_is_not_constructed() -> None:
    adapter_path = Path(__file__).resolve().parents[2] / "phase_codec_adapter.py"
    source = adapter_path.read_text(encoding="utf-8")
    assert "65536 * 65536" not in source
    assert "torch.outer" not in source
    assert "torch.mm" not in source
