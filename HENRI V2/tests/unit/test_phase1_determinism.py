"""Tests for the deterministic seeding standard (doc sec. 1.2 item 2)."""

import pytest
import torch

from henri.determinism import (
    DEFAULT_MANIFEST_SEED,
    RunManifest,
    assert_deterministic,
    derive_seed,
)


def test_seed_derivation_is_stable_and_independent():
    a = derive_seed(20261001, "ingress")
    b = derive_seed(20261001, "ingress")
    c = derive_seed(20261001, "kernel")
    assert a == b, "derivation is not stable"
    assert a != c, "components share a seed stream (arms would correlate)"


def test_manifest_is_reproducible():
    m1 = RunManifest(seed_seq=7)
    m2 = RunManifest(seed_seq=7)
    assert m1.seed_for("x") == m2.seed_for("x")
    assert m1.as_dict() == m2.as_dict()


def test_manifest_apply_seeds_torch():
    m = RunManifest(seed_seq=11)
    m.apply("probe")
    first = torch.randn(4)
    m.apply("probe")
    second = torch.randn(4)
    assert torch.equal(first, second)


def test_assert_deterministic_passes_for_seeded_draw():
    m = RunManifest(seed_seq=DEFAULT_MANIFEST_SEED)

    def seeded():
        m.apply("seeded")            # draw_fn owns its seeding
        return torch.randn(8)

    assert_deterministic(seeded)


def test_assert_deterministic_catches_unseeded_draw():
    """TAUTOLOGY GUARD: an unseeded draw must be DETECTED, not passed.

    This guard caught a real vacuity: the first version of `assert_deterministic`
    re-seeded internally before both draws, so this arm passed despite having no
    seeding at all. The harness must NOT re-seed between the compared draws.
    """
    def unseeded():
        return torch.randn(8)        # no manifest application -> nondeterministic

    with pytest.raises(AssertionError, match="NONDETERMINISTIC"):
        assert_deterministic(unseeded)


def test_different_components_draw_different_streams():
    """Correlated arms would manufacture false agreement."""
    m = RunManifest(seed_seq=20261001)
    m.apply("arm_a")
    a = torch.randn(6)
    m.apply("arm_b")
    b = torch.randn(6)
    assert not torch.equal(a, b)
