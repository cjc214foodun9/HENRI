"""Sealed Modern Hopfield egress codebook — the A2 closure path (contract tests).

WHY THESE EXIST
    HoloEgressCodebook is the tree's implementation of the vision map's "Lexical
    Snap". These tests pin the properties that make it a REPAIR for Defect A2
    rather than a restatement of it, and each one fails if the property is lost:

      T1 the seal can actually FAIL (the reference's check was vacuous, D-1);
      T2 identity round-trip is EXACT on a tokenizer-derived codebook;
      T3 a RANDOM codebook fails the same test -- the control that separates
         "binding works" from "some codebook can round-trip";
      T4 phase reaches the logits (the reference used torch.abs and was measured
         phase-blind, D-2);
      T5 a placeholder manifest cannot be defaulted in (D-3);
      T6 vocab_size_V / manifest disagreement is RECORDED, not silently tolerated.

    T3 is the load-bearing one: without it T2 would only show that a codebook
    exists whose own rows it can recover.
"""
import pathlib
import sys

import pytest
import torch

R = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))

from henri_vla_tokenizer import (  # noqa: E402
    HoloEgressCodebook,
    HoloManifestError,
    HoloVLAConfig,
    HoloVLATokenizer,
    ManifestSeal,
)

V = 64
CFG = HoloVLAConfig(ambient_dim_D=512, num_blocks=64, grid_size_S=8,
                    vocab_size_V=V, feat_dim=64, seed=42)
MANIFEST = [f"tok_{i}" for i in range(V)]


@pytest.fixture(scope="module")
def tokenizer():
    return HoloVLATokenizer(CFG)


@pytest.fixture(scope="module")
def egress(tokenizer):
    return HoloEgressCodebook(CFG, tokenizer, MANIFEST)


# ---------------------------------------------------------------- T5 / D-3
def test_placeholder_manifest_is_not_defaulted(tokenizer):
    """An empty/absent manifest must raise, never default to '<tok_i>'."""
    with pytest.raises(HoloManifestError):
        HoloEgressCodebook(CFG, tokenizer, [])
    with pytest.raises(HoloManifestError):
        HoloEgressCodebook(CFG, tokenizer, None)


# ---------------------------------------------------------------- T1 / D-1
def test_seal_require_raises_on_tampered_manifest():
    """A seal that can never fail is not a seal. require() must raise."""
    seal = ManifestSeal.compute(MANIFEST)
    assert seal.verify(MANIFEST) is True
    tampered = list(MANIFEST)
    tampered[0] = "tok_0_changed"
    assert seal.verify(tampered) is False
    with pytest.raises(HoloManifestError):
        seal.require(tampered)


def test_wrong_expected_seal_is_rejected_on_construction(tokenizer):
    """Passing the wrong expected seal must fail closed at build time."""
    wrong = ManifestSeal.compute([f"other_{i}" for i in range(V)])
    with pytest.raises(HoloManifestError):
        HoloEgressCodebook(CFG, tokenizer, MANIFEST, expected_seal=wrong)


def test_seal_detects_permutation():
    """The manifest is order-sensitive: a permutation must change the digest."""
    a = ManifestSeal.compute(["a", "b", "c"])
    b = ManifestSeal.compute(["b", "a", "c"])
    assert a.sha256 != b.sha256


# ---------------------------------------------------------------- T2
def test_identity_round_trip_is_exact(egress):
    """encode(manifest[k]) then argmax must return k for every k."""
    rt = egress.identity_round_trip()
    assert rt["total"] == V
    assert rt["duplicate_manifest_entries"] == 0
    assert rt["rate"] == 1.0, f"round-trip not exact: {rt}"


# ---------------------------------------------------------------- T3 (control)
def test_random_codebook_fails_the_same_test(egress, tokenizer):
    """The document's own proposed codebook (randn, seed 42) must NOT recover.

    Without this control, the exact round-trip above would only demonstrate that
    some codebook can recover its own rows.
    """
    g = torch.Generator().manual_seed(CFG.seed)
    rand_M = torch.nn.functional.normalize(
        torch.randn(V, CFG.feat_dim, generator=g), p=2.0, dim=-1)
    with torch.no_grad():
        h = egress._embed(tokenizer.encode_text(MANIFEST))
        pred = ((h @ rand_M.t()) * CFG.hopfield_inverse_temp).argmax(dim=-1).tolist()
    hits = sum(1 for k, p in enumerate(pred) if k == p)
    assert hits / float(V) < 0.5, (
        f"random control scored {hits}/{V}; the round-trip test is not "
        f"discriminating binding from chance")


# ---------------------------------------------------------------- T4 / D-2
def test_phase_is_not_discarded(egress, tokenizer):
    """A global pi rotation must CHANGE the logits.

    The reference computed torch.abs(wave) before projecting; a pi rotation then
    left logits bit-identical (measured max |delta| = 0.00000000).
    """
    with torch.no_grad():
        w = tokenizer.encode_text(MANIFEST[:4])
        base = egress.logits(w)
        minus = egress.logits(-w)          # e^{i*pi} = -1, exact global phase flip
        delta = float((base - minus).abs().max())
    assert delta > 1e-6, (
        "logits are invariant under a global sign flip: phase is being discarded")


# ---------------------------------------------------------------- T6
def test_vocab_mismatch_is_recorded_not_hidden(tokenizer):
    """cfg.vocab_size_V disagreeing with the manifest must be visible."""
    cfg_small = HoloVLAConfig(ambient_dim_D=512, num_blocks=64, grid_size_S=8,
                              vocab_size_V=32, feat_dim=64, seed=42)
    e = HoloEgressCodebook(cfg_small, tokenizer, MANIFEST)
    assert e.vocab_size == V
    assert e.vocab_size_mismatch is True


# ---------------------------------------------------------------- provenance
def test_projection_is_frozen_not_trainable(egress):
    """The codebook's OWN projection is a frozen seeded JL map, not a learned head.

    SCOPE (corrected after a first failing version): `named_parameters()` walks
    child modules, and `HoloEgressCodebook` holds a whole `HoloVLATokenizer`, whose
    `action_encoder` IS a legitimately trainable `nn.Linear`. Asserting "no
    trainable parameter anywhere in the tree" is therefore wrong, not a defect in
    the codebook. The honest claim is narrower: the codebook introduces no learned
    readout parameter, and its projection is frozen.
    """
    assert egress.proj.requires_grad is False, "projection must be frozen"
    own = [
        n for n, p in egress.named_parameters()
        if not n.startswith("tokenizer.") and p.requires_grad
    ]
    assert own == [], f"codebook introduces trainable readout params: {own}"


def test_codebook_rows_are_a_buffer_not_a_parameter_slice(egress):
    """M must be a registered buffer derived in __init__, not a trainable weight."""
    bufs = dict(egress.named_buffers())
    assert "codebook_M" in bufs
    names = dict(egress.named_parameters())
    assert "codebook_M" not in names
    assert bufs["codebook_M"].shape == (V, CFG.feat_dim)


def test_snap_is_a_probability_distribution(egress, tokenizer):
    """Hopfield readout must emit normalized weights and finite entropy."""
    with torch.no_grad():
        _, p, ent = egress.snap(tokenizer.encode_text(MANIFEST[:4]))
        assert torch.allclose(p.sum(dim=-1), torch.ones(4), atol=1e-5)
        assert torch.isfinite(ent).all()
