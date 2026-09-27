"""Contract tests for henri_operator_promotion.py.

The gate's whole purpose is to BLOCK.  Every blocking path gets a test that
must raise; the one permissive path gets a test that must pass.

PROVENANCE TEST: the module's constants are asserted equal to the on-disk
receipt, so hardening the numbers in the module cannot silently diverge from
what was measured.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from henri_operator_promotion import (  # noqa: E402
    FALSIFIED_FAMILIES,
    INCUMBENT_FAMILY,
    MEASURED_DELTA,
    MEASURED_HELDOUT,
    RECEIPT_PATH,
    RECEIPT_SHA256,
    TAU,
    PromotionBlocked,
    assert_promotion_allowed,
    measured_delta,
    register,
    registry,
)

RECEIPT = os.path.join(ROOT, RECEIPT_PATH)


def _receipt():
    if not os.path.exists(RECEIPT):
        pytest.fail("provenance receipt missing: " + RECEIPT_PATH)
    with open(RECEIPT, "rb") as fh:
        raw = fh.read()
    return hashlib.sha256(raw).hexdigest(), json.loads(raw.decode("utf-8"))


def test_receipt_digest_matches_the_pinned_sha():
    sha, _ = _receipt()
    assert sha == RECEIPT_SHA256


def test_constants_equal_the_receipt_at_full_precision():
    _, r = _receipt()
    h2 = r["h2_decision"]
    assert MEASURED_HELDOUT["diag_ls"] == h2["control_mean_cos"]
    assert MEASURED_HELDOUT["resonator_tripartite"] == h2["treatment_mean_cos"]
    assert MEASURED_HELDOUT["identity"] == h2["identity_mean_cos"]
    assert TAU == h2["tau"]


def test_measured_delta_reproduces_the_receipt_exactly():
    _, r = _receipt()
    assert MEASURED_DELTA == r["h2_decision"]["delta"]
    assert measured_delta("resonator_tripartite") == r["h2_decision"]["delta"]
    assert measured_delta("resonator_tripartite") < TAU


def test_incumbent_is_the_diagonal_ridge():
    assert INCUMBENT_FAMILY == "diag_ls"
    assert MEASURED_HELDOUT[INCUMBENT_FAMILY] > MEASURED_HELDOUT["resonator_tripartite"]


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
    assert_promotion_allowed("novel_family", TAU) is None


def test_registering_a_falsified_name_cannot_make_it_promotable():
    fam = register("resonator_tripartite", 0.99, "attempt to launder it in")
    assert fam.promotable is False
    with pytest.raises(PromotionBlocked):
        assert_promotion_allowed("resonator_tripartite", 10.0)


def test_falsified_family_list_is_not_empty():
    assert "resonator_tripartite" in FALSIFIED_FAMILIES


# END
