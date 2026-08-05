"""Contract tests for the Run 21 structured qFHRR codec (STRUCTURED_CODEC_KILL_V1).

Local CPU-only. These verify representation geometry and the ring_to_real
boundary; they do NOT exercise the remote ranking pipeline.
"""

from __future__ import annotations

import math

import pytest
import torch

from qfhrr_structured_codec import (
    StructuredCharPositionCodec,
    ring_to_real,
)


@pytest.fixture(scope="module")
def codec_full() -> StructuredCharPositionCodec:
    return StructuredCharPositionCodec(device="cpu", position_mode="full")


@pytest.fixture(scope="module")
def codec_nopos() -> StructuredCharPositionCodec:
    return StructuredCharPositionCodec(device="cpu", position_mode="none")


@pytest.fixture(scope="module")
def codec_shuffled() -> StructuredCharPositionCodec:
    return StructuredCharPositionCodec(device="cpu", position_mode="shuffled")


def _sim(c: StructuredCharPositionCodec, a: str, b: str) -> float:
    ra = ring_to_real(c, c.encode_text(a)).view(1, -1)
    rb = ring_to_real(c, c.encode_text(b)).view(1, -1)
    return float(torch.nn.functional.cosine_similarity(ra, rb).item())


def test_output_contract(codec_full: StructuredCharPositionCodec) -> None:
    ring = codec_full.encode_text("f(3, 3, 3)")
    assert ring.dtype == torch.uint8
    assert ring.shape == (65536,)
    assert int(ring.min()) >= 0 and int(ring.max()) <= 255
    assert not torch.isnan(ring.to(torch.float32)).any()
    real = ring_to_real(codec_full, ring)
    assert real.shape == (65536,)
    assert real.dtype == torch.float32


def test_identical_deterministic(codec_full: StructuredCharPositionCodec) -> None:
    assert _sim(codec_full, "f(3, 3, 3)", "f(3, 3, 3)") >= 0.999


def test_position_binding_active_full(codec_full: StructuredCharPositionCodec) -> None:
    # Under position binding, "ab" and "ba" must differ measurably.
    assert _sim(codec_full, "ab", "ba") < 0.90


def test_position_binding_inactive_nopos(codec_nopos: StructuredCharPositionCodec) -> None:
    # Without position, "ab" and "ba" are commutative-equivalent bundles.
    assert _sim(codec_nopos, "ab", "ba") > 0.95


def test_shuffle_degrades_order_sensitivity(
    codec_full: StructuredCharPositionCodec,
    codec_shuffled: StructuredCharPositionCodec,
) -> None:
    d_full = 1.0 - _sim(codec_full, "abc", "cba")
    d_shuf = 1.0 - _sim(codec_shuffled, "abc", "cba")
    assert d_full > d_shuf


def test_nearby_output_continuity(codec_full: StructuredCharPositionCodec) -> None:
    baseline = 1.0 / math.sqrt(codec_full.d_model)
    assert _sim(codec_full, "27", "28") > 10.0 * baseline


def test_unrelated_low(codec_full: StructuredCharPositionCodec) -> None:
    baseline = 1.0 / math.sqrt(codec_full.d_model)
    # Character-disjoint strings: a character-token codec represents shared
    # characters as real overlap, so the unrelated control must share none.
    assert _sim(codec_full, "abcd", "1234") <= 10.0 * baseline


def test_position_mode_validation() -> None:
    with pytest.raises(ValueError):
        StructuredCharPositionCodec(device="cpu", position_mode="bogus")


def test_legacy_ring_to_real_preserves_linear_map() -> None:
    from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec

    legacy = qFHRREpistemicCodec(device="cpu")
    ring = legacy.encode_text("f(3, 3, 3)")
    mapped = ring_to_real(legacy, ring)
    expected = ring.to(torch.float32) / (legacy.k_bins - 1) * 2.0 - 1.0
    assert torch.allclose(mapped, expected)
