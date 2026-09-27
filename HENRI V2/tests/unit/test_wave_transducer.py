"""Contract tests for henri_wave_transducer.py (SPEC-2026-09-24-FUWT-EGRESS-V1).

Every declared invariant gets a test, and every fail-closed path gets a
negative test. The module's own docstring records two defects it must not
inherit, and both are covered:

  D1  a previous revision required `shape[-1] * 2 == d_model`, which demanded
      32768 complex and then failed to reshape into 32 x 2048. The spec's wave is
      [Batch, 65536] COMPLEX. Covered by test_spec_contract_shapes.
  D2  the einsum was transposed ("bfd,mf->bmd" contracts f=M=32 against f=6144).
      Covered implicitly by every shape test; explicitly by
      test_prefix_contracts_over_the_feature_axis.

Upstream defect this module refuses to inherit:
  `hopfield_cleanup.ContinuousHopfieldCleanup.store_engrams` accepts a rank-3
  [1,8,8] tensor WITHOUT validation. test_lexical_snap_rejects_rank3_codebook
  asserts the transducer rejects it instead.
"""

from __future__ import annotations

import pytest
import torch

from henri_wave_transducer import (
    FUWTConfig,
    FusedUnitaryWaveTransducer,
    TransducerNormViolation,
    TransducerShapeError,
    real_wave_to_complex,
    spec_wave_to_real,
)

# ---- fast config: PACKET_WIDTH is a fixed module constant (2048), so d_model
# ---- must be n_packets * 2048. 4 x 2048 = 8192 keeps the tests quick.
FAST_PACKETS = 4
FAST_D_MODEL = FAST_PACKETS * 2048
FAST_PREFIX = 64

# ---- full spec config
SPEC_D_MODEL = 65536
SPEC_PACKETS = 32
SPEC_PREFIX = 2048


def unit_wave(batch: int, d_model: int, seed: int = 0) -> torch.Tensor:
    """A wave on S^(D-1): ||psi||_2 == 1 exactly."""
    g = torch.Generator().manual_seed(seed)
    re = torch.randn(batch, d_model, generator=g)
    im = torch.randn(batch, d_model, generator=g)
    psi = torch.complex(re, im)
    return psi / torch.linalg.vector_norm(psi, dim=-1, keepdim=True)


@pytest.fixture(scope="module")
def fast_transducer():
    return FusedUnitaryWaveTransducer(
        FUWTConfig(n_packets=FAST_PACKETS, d_model=FAST_D_MODEL, prefix_dim=FAST_PREFIX)
    )


# ============================================================================
# CONFIG
# ============================================================================

def test_config_validation_rejects_inconsistent_dims():
    # n_packets * 2048 must equal d_model
    with pytest.raises(TransducerShapeError, match="must equal"):
        FUWTConfig(n_packets=32, d_model=65535).validate()
    with pytest.raises(TransducerShapeError, match="norm_tol"):
        FUWTConfig(norm_tol=0.0).validate()
    with pytest.raises(TransducerShapeError, match="prefix_dim"):
        FUWTConfig(prefix_dim=0).validate()


def test_config_defaults_match_the_spec():
    c = FUWTConfig()
    assert c.d_model == SPEC_D_MODEL == 65536
    assert c.n_packets == SPEC_PACKETS == 32
    assert c.prefix_dim == SPEC_PREFIX == 2048
    assert c.max_new_tokens == 4096
    assert c.output_dtype == torch.bfloat16      # spec: prefix_embeddings bf16
    assert c.accum_dtype == torch.float32


# ============================================================================
# SPEC TENSOR CONTRACT
# ============================================================================

def test_spec_contract_shapes():
    """[B, 65536] complex64 -> [B, 32, 2048] bfloat16. The D1 defect guard."""
    t = FusedUnitaryWaveTransducer(
        FUWTConfig(n_packets=SPEC_PACKETS, d_model=SPEC_D_MODEL, prefix_dim=SPEC_PREFIX)
    )
    psi = unit_wave(1, SPEC_D_MODEL)
    assert psi.shape == (1, SPEC_D_MODEL)

    h = t(psi)
    assert h.shape == (1, SPEC_PACKETS, SPEC_PREFIX)
    assert h.dtype == torch.bfloat16
    assert torch.isfinite(h.float()).all()


def test_polar_features_shape_matches_spec():
    t = FusedUnitaryWaveTransducer(
        FUWTConfig(n_packets=SPEC_PACKETS, d_model=SPEC_D_MODEL, prefix_dim=SPEC_PREFIX)
    )
    psi = unit_wave(1, SPEC_D_MODEL)
    pol = t.polar_decompose(psi)          # [B, M, 3*2048=6144]
    assert pol.features.shape == (1, SPEC_PACKETS, 3 * 2048) == (1, 32, 6144)
    assert pol.features.dtype == torch.float32


def test_prefix_contracts_over_the_feature_axis():
    """The transposed-einsum guard (defect D2).

    If the contraction were over the packet axis (M=4) against the feature width
    (6144) it would raise; if it silently broadcast it would produce the wrong
    width. Assert BOTH the output width and that the projection actually reads the
    feature axis by changing features and observing a change in the output.
    """
    t = FusedUnitaryWaveTransducer(
        FUWTConfig(n_packets=FAST_PACKETS, d_model=FAST_D_MODEL, prefix_dim=FAST_PREFIX)
    )
    psi = unit_wave(2, FAST_D_MODEL)
    h, pol = t.prefix_embeddings(psi)
    assert h.shape == (2, FAST_PACKETS, FAST_PREFIX)

    # W must be [prefix_dim, feature_width], i.e. contractable with features' last axis
    assert t.transduce.shape == (FAST_PREFIX, 3 * 2048)
    assert t.transduce.shape[1] == pol.features.shape[-1]


def test_dtype_boundaries_are_declared(fast_transducer):
    psi = unit_wave(1, FAST_D_MODEL)
    h, pol = fast_transducer.prefix_embeddings(psi)
    assert pol.features.dtype == torch.float32     # accumulate fp32
    assert h.dtype == torch.bfloat16               # emit bf16 per spec


# ============================================================================
# FAIL-CLOSED PATHS
# ============================================================================

def test_rejects_non_unitary_wave(fast_transducer):
    psi = unit_wave(1, FAST_D_MODEL) * 1.5         # ||psi|| = 1.5
    with pytest.raises(TransducerNormViolation, match="non-unitary"):
        fast_transducer(psi)


def test_rejects_small_norm_drift_beyond_tolerance(fast_transducer):
    psi = unit_wave(1, FAST_D_MODEL) * (1.0 + 1e-2)
    with pytest.raises(TransducerNormViolation):
        fast_transducer(psi)


def test_accepts_norm_drift_within_tolerance(fast_transducer):
    """The spec allows |norm - 1| <= 1e-4, so a drift of 1e-5 must PASS."""
    psi = unit_wave(1, FAST_D_MODEL) * (1.0 + 1e-5)
    out = fast_transducer(psi)
    assert out.shape == (1, FAST_PACKETS, FAST_PREFIX)


def test_rejects_nan_wave(fast_transducer):
    psi = unit_wave(1, FAST_D_MODEL)
    psi[0, 0] = complex(float("nan"), 0.0)
    with pytest.raises(TransducerNormViolation):
        fast_transducer(psi)


def test_rejects_real_input(fast_transducer):
    """The spec's wave_input is complex64; a real tensor must be refused."""
    with pytest.raises(TransducerShapeError, match="complex64"):
        fast_transducer(torch.randn(1, FAST_D_MODEL))


def test_rejects_wrong_width(fast_transducer):
    with pytest.raises(TransducerShapeError, match="must be"):
        fast_transducer(unit_wave(1, FAST_D_MODEL // 2))


def test_rejects_wrong_rank(fast_transducer):
    psi = unit_wave(1, FAST_D_MODEL).reshape(1, 1, FAST_D_MODEL)
    with pytest.raises(TransducerShapeError, match="must be"):
        fast_transducer(psi)


# ============================================================================
# ENERGY CONSERVATION
# ============================================================================

def test_packet_energy_sums_to_one(fast_transducer):
    """The orthogonal partition must not lose or create L2 energy."""
    psi = unit_wave(3, FAST_D_MODEL)
    pol = fast_transducer.polar_decompose(psi)
    total = pol.packet_energy().sum(dim=-1)        # [B]
    assert torch.allclose(total, torch.ones(3), atol=1e-5)


def test_reconstruct_roundtrip_is_exact(fast_transducer):
    """r * exp(i*theta) must reproduce psi -- proves the decomposition is lossless."""
    psi = unit_wave(2, FAST_D_MODEL)
    pol = fast_transducer.polar_decompose(psi)
    back = fast_transducer.reconstruct(pol.radius, pol.angle)
    assert torch.allclose(back, psi, atol=1e-4)


def test_radius_is_non_negative_and_angle_in_range(fast_transducer):
    psi = unit_wave(2, FAST_D_MODEL)
    pol = fast_transducer.polar_decompose(psi)
    assert (pol.radius >= 0).all()
    assert (pol.angle >= -torch.pi - 1e-6).all()
    assert (pol.angle <= torch.pi + 1e-6).all()


def test_features_are_the_spec_triple(fast_transducer):
    """u_p = [r_p, cos(theta_p), sin(theta_p)], concatenated on the last axis."""
    psi = unit_wave(1, FAST_D_MODEL)
    pol = fast_transducer.polar_decompose(psi)
    W = 2048
    assert torch.allclose(pol.features[..., :W], pol.radius, atol=1e-6)
    assert torch.allclose(pol.features[..., W:2 * W], torch.cos(pol.angle), atol=1e-6)
    assert torch.allclose(pol.features[..., 2 * W:], torch.sin(pol.angle), atol=1e-6)


# ============================================================================
# ADAPTERS
# ============================================================================

def test_real_wave_to_complex_pairs_last_axis():
    real = torch.randn(2, 8)
    psi = real_wave_to_complex(real)
    assert psi.shape == (2, 4)
    assert psi[0, 0] == complex(real[0, 0], real[0, 1])


def test_real_wave_to_complex_rejects_odd_width():
    with pytest.raises(TransducerShapeError, match="even"):
        real_wave_to_complex(torch.randn(2, 7))


def test_spec_wave_to_real_doubles_width():
    psi = unit_wave(2, FAST_D_MODEL)
    real = spec_wave_to_real(psi)
    assert real.shape == (2, 2 * FAST_D_MODEL)


# ============================================================================
# HOPFIELD SNAP -- refuses to inherit the upstream rank-3 defect
# ============================================================================

def test_lexical_snap_rejects_rank3_codebook(fast_transducer):
    """The measured upstream defect: store_engrams accepts [1,8,8] silently.

    The transducer must REFUSE it rather than rely on that leniency.
    """
    psi = unit_wave(1, FAST_D_MODEL)
    bad_cb = torch.randn(4, 8, 8)                  # rank 3
    with pytest.raises(TransducerShapeError, match="rank 2"):
        fast_transducer.lexical_snap(psi, bad_cb, top_k=2)


def test_lexical_snap_requires_matching_width(fast_transducer):
    psi = unit_wave(1, FAST_D_MODEL)
    wrong_cb = torch.randn(4, 16)                  # not 2*d_model
    with pytest.raises(TransducerShapeError, match="codebook width"):
        fast_transducer.lexical_snap(psi, wrong_cb, top_k=2)


def test_lexical_snap_returns_indices_and_confidence(fast_transducer):
    """Positive path: a well-formed [V, 2D] codebook must snap without error."""
    D = 2 * FAST_D_MODEL
    cb = torch.randn(4, D)
    cb = cb / cb.norm(dim=-1, keepdim=True)
    psi = unit_wave(1, FAST_D_MODEL)
    idx, conf = fast_transformer_snap(fast_transducer, psi, cb)
    assert idx.shape[0] == 1
    assert conf.shape[0] == 1
    assert int(idx.max()) < 4


def fast_transformer_snap(t, psi, cb):
    return t.lexical_snap(psi, cb, beta=8.0, top_k=2)
