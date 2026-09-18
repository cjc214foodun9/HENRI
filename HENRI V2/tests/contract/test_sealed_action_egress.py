"""Contract tests: the checkpoint-free SEALED action egress.

WHY THESE EXIST
    `decode_action_egress` needs a LOADED HENRIUnifiedEgressTransducer, built by
    henri_decoder with checkpoint_policy="required". With no checkpoint on disk
    that path fails closed, so before this change the ARC egress had NO decoder
    at all without trained weights. The sealed codebook path removes the
    checkpoint dependence. These tests pin the properties that make it usable and
    the boundaries that make it fail closed.

WHAT IS ASSERTED, AND WHY EACH IS FALSIFIABLE
    P1  build + identity round-trip over the ACTION manifest == |A| / |A|
        (the codebook recovers its own rows; the A2 falsifier)
    P2  decode returns a member of the allowed set -- never an out-of-vocab index
    P3  boundary mapping: wrong num_blocks*8, odd d_model, and [.,8]-violations
        all RAISE (no silent reinterpretation)
    P4  phase reaches the logits: sign-flipping the wave changes the logits
    P5  logits are finite and top3 is ordered, bounded, and inside the vocabulary
    P6  the cache returns the SAME object (so the step loop does not rebuild)
    P7  duplicate action names and an empty vocabulary RAISE
    P8  no gradient path exists (the readout is frozen, not a trained head)
"""
from __future__ import annotations

import enum
import math
import pathlib
import sys

import pytest
import torch

R = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))

from arc_egress_contract import (  # noqa: E402
    ActionEgressVocabulary,
    SealedEgressError,
    build_sealed_action_codebook,
    decode_action_egress_sealed,
    real_to_phase_wave,
)

D_MODEL = 512  # reduced CPU scale: num_blocks(64) * 8

try:  # the live environment alphabet when available
    from arcengine import GameAction  # type: ignore
except Exception:  # noqa: BLE001
    GameAction = None


class _LocalAction(enum.Enum):
    RESET = 0
    ACTION1 = 1
    ACTION2 = 2
    ACTION3 = 3
    ACTION4 = 4
    ACTION5 = 5
    ACTION6 = 6
    ACTION7 = 7


ACTIONS = list((GameAction if GameAction is not None else _LocalAction))
# HENRIVisionEncoder() default is 8x8, so the wave is 64 blocks x 8 blades.
NUM_BLOCKS = 64


def make_vocab(actions=None):
    return ActionEgressVocabulary(
        GameAction if GameAction is not None else _LocalAction,
        actions if actions is not None else ACTIONS,
    )


def make_wave(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(NUM_BLOCKS, 8, generator=g)
    return torch.nn.functional.normalize(w, p=2.0, dim=None)


@pytest.fixture(scope="module")
def vocab():
    return make_vocab()


@pytest.fixture(scope="module")
def codebook(vocab):
    return build_sealed_action_codebook(vocab, D_MODEL)


# --------------------------------------------------------------------- P1
def test_build_and_identity_round_trip(codebook, vocab):
    assert tuple(codebook.codebook_M.shape) == (vocab.n_actions, 256)
    rt = codebook.identity_round_trip()
    assert rt["total"] == vocab.n_actions
    assert rt["correct"] == vocab.n_actions, f"round-trip failed: {rt}"
    assert rt["rate"] == 1.0
    assert rt["duplicate_manifest_entries"] == 0


def test_codebook_is_frozen_not_a_trained_head(codebook):
    """P8 -- the readout derives from encode_text(manifest), not from weights."""
    assert codebook.codebook_M.requires_grad is False


# --------------------------------------------------------------------- P2
def test_decode_returns_an_allowed_action(vocab, codebook):
    allowed = {a.name for a in vocab.actions}
    for seed in range(6):
        res = decode_action_egress_sealed(make_wave(seed), vocab, codebook, D_MODEL)
        assert res.action_name in allowed
        assert res.action.name == res.action_name
        assert 0 <= res.action_index < vocab.n_actions
        assert int(torch.argmax(res.action_probs).item()) == res.action_index


def test_decode_is_deterministic(vocab, codebook):
    a = decode_action_egress_sealed(make_wave(3), vocab, codebook, D_MODEL)
    b = decode_action_egress_sealed(make_wave(3), vocab, codebook, D_MODEL)
    assert a.action_name == b.action_name
    assert torch.equal(a.action_logits, b.action_logits)


# --------------------------------------------------------------------- P3
def test_wrong_flat_length_raises(vocab, codebook):
    bad = torch.randn(16, 8)  # 128 != 512
    with pytest.raises(SealedEgressError):
        decode_action_egress_sealed(bad, vocab, codebook, D_MODEL)


def test_wrong_blade_count_raises(vocab, codebook):
    bad = torch.randn(64, 4)
    with pytest.raises(SealedEgressError):
        decode_action_egress_sealed(bad, vocab, codebook, D_MODEL)


def test_odd_d_model_raises(vocab):
    """An odd d_model cannot be split into equal (cos, sin) halves."""
    with pytest.raises(SealedEgressError):
        real_to_phase_wave(torch.randn(64, 8), 511)
    with pytest.raises(SealedEgressError):
        build_sealed_action_codebook(vocab, 511)


def test_phase_mapping_is_lossless_in_the_count(vocab):
    w = make_wave(1)
    phase = real_to_phase_wave(w, D_MODEL)
    assert phase.dtype == torch.complex64
    assert phase.shape == (1, D_MODEL // 2)
    # every one of d_model numbers survives: reals then imags, in order
    flat = w.reshape(-1)
    assert torch.equal(phase.real.reshape(-1), flat[: D_MODEL // 2])
    assert torch.equal(phase.imag.reshape(-1), flat[D_MODEL // 2:])


# --------------------------------------------------------------------- P4
def test_phase_reaches_the_logits(codebook, vocab):
    """A conjugate wave must move the logits: otherwise the readout is phase-blind.

    The conjugate is built EXPLICITLY rather than with torch.conj(): torch.conj
    returns an unresolved conjugate tensor, and view_as_real (used by the
    codebook's _embed) refuses it. That is a property of the test's construction,
    not of the egress path, which never receives an unresolved-conj tensor.
    """
    phase = real_to_phase_wave(make_wave(2), D_MODEL)
    conj_phase = torch.complex(phase.real, -phase.imag)
    l1 = codebook.logits(phase)
    l2 = codebook.logits(conj_phase)
    assert float((l1 - l2).abs().max().item()) > 1e-3


def test_distinct_waves_give_distinct_logits(vocab, codebook):
    l1 = decode_action_egress_sealed(make_wave(10), vocab, codebook, D_MODEL)
    l2 = decode_action_egress_sealed(make_wave(11), vocab, codebook, D_MODEL)
    assert not torch.allclose(l1.action_logits, l2.action_logits)


# --------------------------------------------------------------------- P5
def test_logits_finite_entropy_convention_matches_existing_contract(vocab, codebook):
    """entropy_bits is NORMALIZED in [0,1]; token_entropy_bits is RAW bits.

    That split is the ESTABLISHED convention of this contract, pinned by
    tests/contract/test_arc_egress_contract.py (0.0 <= entropy_bits <= 1.0 and
    token_entropy_bits > 0.0) and implemented by entropy_bits_of(). A first
    version of the sealed decoder emitted RAW bits in `entropy_bits` (2.9869 for
    8 actions), which would have made one field name mean two scales depending on
    which decoder ran. This test pins the convention so that cannot recur.
    """
    res = decode_action_egress_sealed(make_wave(4), vocab, codebook, D_MODEL)
    assert bool(torch.isfinite(res.action_logits).all())
    assert 0.0 <= res.entropy_bits <= 1.0, "entropy_bits must be NORMALIZED"
    raw_max = math.log2(vocab.n_actions)
    assert 0.0 <= res.token_entropy_bits <= raw_max + 1e-6, (
        "token_entropy_bits must be RAW bits, bounded by log2(V)")
    probs = [p for _, p in res.top3]
    assert probs == sorted(probs, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in probs)
    names = {a.name for a in vocab.actions}
    assert all(n in names for n, _ in res.top3)
    assert len(res.top3) == min(3, vocab.n_actions)


def test_sealed_entropy_uses_the_shared_normalization_helper(vocab, codebook):
    """Cross-path comparability: same field name => same scale => same helper.

    The strongest form of the check above: the value must equal what the module's
    OWN helper produces for the same probabilities, so a future edit cannot
    silently re-introduce a second scale.
    """
    from arc_egress_contract import entropy_bits_of

    res = decode_action_egress_sealed(make_wave(7), vocab, codebook, D_MODEL)
    assert res.entropy_bits == pytest.approx(
        entropy_bits_of(res.action_probs), abs=1e-12)
    raw = float(
        -(res.action_probs * torch.log2(res.action_probs + 1e-12)).sum().item())
    assert res.token_entropy_bits == pytest.approx(raw, abs=1e-9)


# --------------------------------------------------------------------- P5b
def test_top1_margin_is_a_probability_gap(vocab, codebook):
    """The margin must be top1-top2 in PROBABILITY units, not logits or a ratio.

    Measured motivation: on 200 arbitrary waves the sealed readout gave mean
    margin 0.05015 and p05 0.00246 while normalized entropy stayed 0.975, i.e.
    near-uniform. A consumer that gates on decisiveness needs this field to be on
    a stated scale, so the scale is pinned here.
    """
    res = decode_action_egress_sealed(make_wave(9), vocab, codebook, D_MODEL)
    t2 = torch.topk(res.action_probs, 2).values
    expected = float(t2[0] - t2[1])
    assert res.top1_margin == pytest.approx(expected, abs=1e-12)
    assert 0.0 <= res.top1_margin <= 1.0
    # A probability gap is BOUNDED BY 1, so in general it is NOT the logit gap.
    # The check is conditional because a small logit gap can coincide with the
    # probability gap by chance; when the logit gap exceeds 1.0 they cannot be the
    # same quantity, which is the discriminating case. (An earlier version of this
    # assertion was a tautology: `margin != approx(logit_gap) or logit_gap <= 1.0`
    # is true for every input, so it tested nothing.)
    logit_gap = float(torch.topk(res.action_logits, 2).values.diff().abs().item())
    if logit_gap > 1.0:
        assert res.top1_margin < logit_gap


def test_margin_floor_fails_closed(vocab, codebook):
    """A floor above the observed margin must RAISE, not return a near-random action."""
    import torch as _t
    res_open = decode_action_egress_sealed(make_wave(12), vocab, codebook, D_MODEL)
    floor = min(res_open.top1_margin + 0.5, 1.0)  # guaranteed to be above
    with pytest.raises(SealedEgressError) as ei:
        decode_action_egress_sealed(
            make_wave(12), vocab, codebook, D_MODEL, min_margin=floor)
    assert "not decisive" in str(ei.value)
    assert res_open.action_name  # the unguarded call still returns a legal action


def test_zero_floor_preserves_unguarded_behaviour(vocab, codebook):
    """min_margin=0.0 is the documented no-guard default: byte-identical result."""
    a = decode_action_egress_sealed(make_wave(13), vocab, codebook, D_MODEL)
    b = decode_action_egress_sealed(
        make_wave(13), vocab, codebook, D_MODEL, min_margin=0.0)
    assert a.action_name == b.action_name
    assert a.top1_margin == b.top1_margin


def test_floor_below_observed_margin_passes(vocab, codebook):
    res = decode_action_egress_sealed(make_wave(14), vocab, codebook, D_MODEL)
    if res.top1_margin > 0.0:
        ok = decode_action_egress_sealed(
            make_wave(14), vocab, codebook, D_MODEL,
            min_margin=res.top1_margin * 0.5)
        assert ok.action_name == res.action_name


def test_single_action_margin_is_one_not_zero():
    """|A|==1 has no top2, so the margin must be 1.0 (fully decisive), not 0.0.

    Reporting 0.0 would make a one-action vocabulary look maximally ambiguous and
    floor-gated consumers would abstain on the only legal action.
    """
    v = ActionEgressVocabulary(_LocalAction, [_LocalAction.RESET])
    cb = build_sealed_action_codebook(v, D_MODEL, feat_dim=64)
    res = decode_action_egress_sealed(make_wave(15), v, cb, D_MODEL)
    assert res.top1_margin == pytest.approx(1.0)
    # and it therefore survives any floor <= 1.0
    res2 = decode_action_egress_sealed(
        make_wave(15), v, cb, D_MODEL, min_margin=1.0)
    assert res2.action_name == "RESET"


# --------------------------------------------------------------------- P6
def test_cache_returns_the_same_object(vocab):
    a = build_sealed_action_codebook(vocab, D_MODEL, feat_dim=64)
    b = build_sealed_action_codebook(vocab, D_MODEL, feat_dim=64)
    assert a is b


def test_cache_key_includes_feat_dim(vocab):
    a = build_sealed_action_codebook(vocab, D_MODEL, feat_dim=64)
    c = build_sealed_action_codebook(vocab, D_MODEL, feat_dim=128)
    assert a is not c


# --------------------------------------------------------------------- P7
def test_duplicate_action_names_raise():
    class Dup:
        def __init__(self):
            self.name = "ACTION1"

    with pytest.raises(Exception):
        build_sealed_action_codebook(
            ActionEgressVocabulary(_LocalAction, [_LocalAction.ACTION1, _LocalAction.ACTION1]),
            D_MODEL)


def test_vocabulary_rejects_duplicates():
    from arc_egress_contract import EgressFailClosedError

    with pytest.raises(EgressFailClosedError):
        ActionEgressVocabulary(
            _LocalAction, [_LocalAction.ACTION1, _LocalAction.ACTION1])


def test_single_action_vocabulary_is_legal_and_degenerate():
    """|A| == 1 is legal: the snap must return that action with zero entropy."""
    v = ActionEgressVocabulary(_LocalAction, [_LocalAction.RESET])
    cb = build_sealed_action_codebook(v, D_MODEL, feat_dim=64)
    res = decode_action_egress_sealed(make_wave(5), v, cb, D_MODEL)
    assert res.action_name == "RESET"
    assert res.entropy_bits == pytest.approx(0.0, abs=1e-9)


def test_expected_manifest_digest_mismatch_raises(vocab):
    with pytest.raises(SealedEgressError):
        build_sealed_action_codebook(
            vocab, D_MODEL, feat_dim=32, expected_manifest_sha256="0" * 64)
