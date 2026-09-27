"""Contract tests for henri_operator_promotion.py.

The gate's whole purpose is to BLOCK.  Every blocking path gets a test that must
raise, and the one allowing path gets a test that must pass.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from henri_operator_promotion import (  # noqa: E402
    FALSIFIED_FAMILIES,
    INCUMBENT_FAMILY,
    MEASURED_HELDOUT,
    TAU,
    PromotionBlocked,
    assert_promotion_allowed,
    measured_delta,
    register,
    registry,
)


def test_tau_is_the_published_constant():
    assert TAU == 0.01


def test_incumbent_is_the_diagonal_ridge():
    assert INCUMBENT_FAMILY == "diag_ls"
    assert MEASURED_HELDOUT[INCUMBENT_FAMILY] == pytest.approx(0.430607)


def test_registry_marks_the_falsified_family_non_promotable():
    reg = registry()
    assert reg["resonator_tripartite"].promotable is False
    assert reg[INCUMBENT_FAMILY].promotable is True


def test_gate_blocks_the_falsified_family():
    with pytest.raises(PromotionBlocked) as e:
        assert_promotion_allowed("resonator_tripartite", 0.05)
    assert "PROMOTION_BLOCKED" in str(e.value)


def test_gate_blocks_the_incumbent():
    with pytest.raises(PromotionBlocked):
        assert_promotion_allowed(INCUMBENT_FAMILY, 1.0)


def test_gate_blocks_a_delta_below_tau():
    with pytest.raises(PromotionBlocked):
        assert_promotion_allowed("novel_family", TAU - 1e-9)


def test_gate_allows_a_novel_family_that_beats_by_tau():
    # the one permissive path: unknown family, delta >= tau
    assert_promotion_allowed("novel_family", TAU)


def test_measured_delta_reproduces_the_receipt():
    # the receipt recorded delta -0.017296
    assert measured_delta("resonator_tripartite") == pytest.approx(-0.017296, abs=1e-6)
    assert measured_delta("resonator_tripartite") < TAU


def test_registering_a_falsified_name_cannot_make_it_promotable():
    fam = register("resonator_tripartite", 0.99, "attempt to launder it in")
    assert fam.promotable is False
    with pytest.raises(PromotionBlocked):
        assert_promotion_allowed("resonator_tripartite", 10.0)


def test_falsified_family_list_is_not_empty():
    assert "resonator_tripartite" in FALSIFIED_FAMILIES


# END
