"""Contract tests: the two amendment carriers.

WHY THESE EXIST
    Two amendment documents supply encoder designs. Both were measured before being
    believed, and BOTH measurements found defects -- including one in my own probe.
    These tests pin the corrected behaviour so a later edit cannot silently restore
    any of the three defects.

THE THREE DEFECTS, AND WHICH TEST PINS EACH
    D-SCALE (mine): the random_control arm returned raw unit-modulus phasors
        (norm sqrt(D)) while the other arms were L2-normalized, so its inner
        products were ~D times larger and it showed a fabricated "+18 semantic
        gap". Fixed by normalizing. Pinned by test_all_arms_are_unit_norm and
        test_random_control_has_no_gap.
    D4 (supplied code): r_threshold=0.707 exceeds the entire null distribution of
        the local Kuramoto order parameter, so the coherence mask is empty and
        F.normalize(0) yields NaN. Pinned by
        test_document_threshold_fails_closed_without_nan.
    D6 (found here): torch.angle(0.0)==0.0 everywhere, so an ALL-ZERO field has
        perfectly aligned phases, scores r=1.0, and passed a naive coherence mask
        with occupancy 1.0. A gate a null field passes is vacuous. Pinned by
        test_dead_input_is_rejected.
"""
from __future__ import annotations

import math

import pytest
import torch

import sys
import pathlib

R = pathlib.Path(__file__).resolve().parents[2]
if str(R) not in sys.path:
    sys.path.insert(0, str(R))

from henri_grounded_lexical_codec import (  # noqa: E402
    GATE_GAP, GATE_RECOVERY, ONTOLOGY, GroundedLexicalCodec,
)
from henri_phaselock_attention import (  # noqa: E402
    DOC_THRESHOLD, PhaseLockAttention, superposed_field,
)

DIM = 4096          # small for speed. NOT 1024: see test_slice_claim_is_resolution_bounded.
TILE = 64
SEED = 20260918


# --------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def codec() -> GroundedLexicalCodec:
    return GroundedLexicalCodec(dim=DIM, seed=SEED)


@pytest.fixture(scope="module")
def audit() -> PhaseLockAttention:
    return PhaseLockAttention(dim=DIM, tile=TILE, enabled=True, seed=SEED)


def _superposed(n: int = 8, seed: int = SEED + 11, dim: int = DIM):
    g = torch.Generator(device="cpu").manual_seed(seed)
    keys = torch.polar(torch.ones(n, dim),
                       torch.rand(n, dim, generator=g) * 2 * math.pi)
    vals = torch.polar(torch.ones(n, dim),
                       torch.rand(n, dim, generator=g) * 2 * math.pi)
    return superposed_field(list(zip(keys, vals)))


# ============================================================ codec: scale guard
def test_all_arms_are_unit_norm(codec):
    """D-SCALE regression. Every arm must emit L2-normalized keys.

    Without this, one arm's inner products scale by sqrt(D) relative to the
    others and the comparison fabricates a gap.
    """
    for arm in GroundedLexicalCodec.ARMS:
        K = codec.encode(arm)
        norms = K.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), \
            f"arm {arm!r} not unit-norm: min={norms.min():.6f} max={norms.max():.6f}"


def test_random_control_has_no_gap(codec):
    """The live trap: independent phasors must show NO semantic proximity."""
    rep = codec.gap_report("random_control")
    assert abs(rep["semantic_gap"]) < GATE_GAP, \
        f"random control shows a gap of {rep['semantic_gap']:+.6f}"


def test_literal_convolution_chain_has_no_gap(codec):
    """D-LITERAL. The documents' remedy, read literally, does not create structure.

    Circular convolution is elementwise multiplication in the frequency domain, so
    a concept built as a PRODUCT of attribute phasors stays unit-modulus and its
    correlation with a sibling is the MEAN of a random ratio vector ~ 0.
    """
    rep = codec.gap_report("literal_convolution")
    assert abs(rep["semantic_gap"]) < GATE_GAP, \
        f"literal convolution produced a gap of {rep['semantic_gap']:+.6f}; if this " \
        "is now real, re-derive the mechanism before trusting it"


def test_role_filler_bundle_creates_the_gap(codec):
    """The corrected construction: shared (role,value) terms ADD constructively."""
    rep = codec.gap_report("role_filler_bundle")
    assert rep["semantic_gap"] >= GATE_GAP, \
        f"gap {rep['semantic_gap']:+.6f} below gate {GATE_GAP}"
    assert rep["mean_shared"] > 0 > rep["mean_disjoint"], \
        "shared pairs must correlate positively and disjoint pairs must not"


def test_role_filler_beats_literal_by_the_gate(codec):
    rf = codec.gap_report("role_filler_bundle")["semantic_gap"]
    lit = codec.gap_report("literal_convolution")["semantic_gap"]
    assert rf - lit >= GATE_GAP, f"margin only {rf - lit:+.6f}"


def test_unbinding_recovers_every_attribute(codec):
    """Corrected construction must support exact role-filler unbinding."""
    hits = total = 0
    for concept in codec.concepts:
        for role in codec.roles:
            idx, _ = codec.recover_attribute(concept, role)
            hits += int(codec.values[role][idx] == ONTOLOGY[concept][role])
            total += 1
    rate = hits / total
    assert rate >= GATE_RECOVERY, f"recovery {hits}/{total} = {rate:.6f}"


def test_shared_pair_count_is_the_ontology_not_the_data(codec):
    """Guard against a fixture that encodes the answer on a preserved axis.

    The shared/disjoint split must follow from the ontology itself, so the counts
    are fixed and inspectable rather than emergent.
    """
    assert codec.shared_attribute_values("lion", "lion") == len(codec.roles)
    assert codec.shared_attribute_values("lion", "teaspoon") < len(codec.roles)
    rep = codec.gap_report("role_filler_bundle")
    assert rep["n_shared_pairs"] + rep["n_disjoint_pairs"] == 15  # C(6,2)


# ======================================================== attention: fail-closed
def test_document_threshold_fails_closed_without_nan(audit):
    """D4. The asserted 0.707 sits above the whole null distribution.

    The supplied code would compute F.normalize(z * mask) on an empty mask and
    return NaN. Ours must fail closed with a typed reason instead.
    """
    psi, X, _ = _superposed()
    audit.coherence_threshold = DOC_THRESHOLD
    res = audit.forward(psi.unsqueeze(0), X[0].unsqueeze(0))
    assert res["fail_closed"] is True
    assert res["reason"] == "MASK_EMPTY_WOULD_DIVIDE_BY_ZERO"
    assert res["occupancy"] == 0.0


def test_null_calibration_matches_the_analytic_scale(audit):
    """The calibrated threshold must come from the null on the SAME dimension."""
    null = audit.null_coherence(n=64)
    analytic = 1.0 / math.sqrt(TILE)
    assert abs(float(null.mean()) - analytic) < 0.05, \
        f"null mean {float(null.mean()):.6f} far from 1/sqrt(tile)={analytic:.6f}"


def test_calibrated_threshold_yields_a_small_slice(audit):
    """Attention is a <1%-order slice. Measure it rather than asserting a figure.

    At D=4096 with tile=64 there are 64 tiles, so the finest expressible occupancy
    is 1/64 = 1.56%. The measured slice is exactly that one tile.
    """
    psi, X, _ = _superposed()
    audit.coherence_threshold = None
    res = audit.forward(psi.unsqueeze(0), X[0].unsqueeze(0))
    assert res["fail_closed"] is False
    assert res["threshold_source"] == "calibrated_null_q0.99"
    assert 0.0 < res["occupancy"] < 0.10, f"occupancy {res['occupancy']}"


def test_slice_claim_is_resolution_bounded():
    """The <1% slice is NOT expressible below ~100 tiles. Fail-closed is correct.

    MEASURED (my own test bug, kept as a regression): at D=1024 with tile=64 there
    are only 16 tiles, so 1% of the space is 0.16 tiles -- unrepresentable. A
    q0.99-calibrated threshold then excludes every tile and the module returns
    MASK_EMPTY_WOULD_DIVIDE_BY_ZERO. That is the CORRECT outcome: refusing to
    report a slice the resolution cannot support beats fabricating a "<1%" figure
    from 16 samples. Any future claim of a sub-percent attention slice must state a
    tile count that can express it (at D=65536, tile=64 -> 1024 tiles, 1% ~ 10).
    """
    small = PhaseLockAttention(dim=1024, tile=TILE, enabled=True, seed=SEED)
    psi, X, _ = _superposed(dim=1024)
    res = small.forward(psi.unsqueeze(0), X[0].unsqueeze(0))
    n_tiles = 1024 // TILE
    assert n_tiles == 16
    assert res["fail_closed"] is True, \
        "a 16-tile field must not report an occupancy it cannot resolve"
    assert res["reason"] == "MASK_EMPTY_WOULD_DIVIDE_BY_ZERO"


def test_dim_mismatch_is_rejected_at_the_boundary():
    """D9: a matched pair of the WRONG width must be rejected at the boundary.

    My own test found this: awareness and query agreed with each other but not
    with the module, so validation passed and the tile reshape raised an
    unactionable RuntimeError from the wrong layer.
    """
    m = PhaseLockAttention(dim=1024, tile=TILE, enabled=True, seed=SEED)
    big = torch.zeros(1, 4096, dtype=torch.complex64)
    with pytest.raises(ValueError, match="last-dim mismatch"):
        m.forward(big, big)


def test_dead_input_is_rejected(audit):
    """D6. A null field has angle==0 everywhere -> r=1.0 -> a naive mask passes it.

    A coherence gate that a null field passes is vacuous, so the mechanism must
    reject on negligible power BEFORE measuring coherence. This is the dead-input
    negative control the supplied module lacks.
    """
    _, X, _ = _superposed()
    audit.coherence_threshold = None
    dead = audit.forward(torch.zeros(1, DIM, dtype=torch.complex64),
                         X[0].unsqueeze(0))
    assert dead["fail_closed"] is True
    assert dead["reason"] == "ZERO_FIELD_VACUOUS_COHERENCE"
    assert dead["occupancy"] is None


def test_no_result_is_ever_nan(audit):
    """Sweep thresholds; every outcome is fail-closed or finite. Never NaN."""
    psi, X, _ = _superposed()
    for thr in (0.707, 0.5, 0.3, 0.15, 0.05, 0.0):
        audit.coherence_threshold = thr
        res = audit.forward(psi.unsqueeze(0), X[0].unsqueeze(0))
        if res["fail_closed"]:
            assert res["occupancy"] in (None, 0.0)
        else:
            assert math.isfinite(res["snr_db"])
            assert 0.0 < res["occupancy"] <= 1.0


def test_default_off_does_the_work_or_none_of_it():
    """Default-OFF must perform no masking AND no retrieval."""
    off = PhaseLockAttention(dim=DIM, tile=TILE, enabled=False, seed=SEED)
    psi, X, _ = _superposed()
    res = off.forward(psi.unsqueeze(0), X[0].unsqueeze(0))
    assert res["enabled"] is False
    assert res["occupancy"] is None
    assert res["reason"] == "DISABLED_DEFAULT_OFF"
    assert "snr_db" not in res and "retrieved_index" not in res


def test_constructor_rejects_indivisible_tile():
    with pytest.raises(ValueError):
        PhaseLockAttention(dim=1000, tile=64)


def test_forward_rejects_shape_mismatch(audit):
    with pytest.raises(ValueError):
        audit.forward(torch.zeros(2, DIM, dtype=torch.complex64),
                      torch.zeros(3, DIM, dtype=torch.complex64))
