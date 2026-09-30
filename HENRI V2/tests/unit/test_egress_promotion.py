"""Tests for the egress beta promotion gate.

These re-read the receipts, so the pinned constants cannot silently drift, and they
assert that the gate is currently CLOSED. If a future session lands a measurement at
M=10,000 / D=65,536, the closure tests must be updated deliberately -- they cannot
pass by accident.
"""

from __future__ import annotations

import json
import os

import pytest

import henri_egress_promotion as EP


def test_receipts_verify_against_pinned_digests():
    rep = EP.audit()
    assert rep["receipts_ok"] is True, rep["digests"]
    for tag, (_rel, pinned) in EP.RECEIPTS.items():
        assert rep["digests"][tag] == pinned


def test_receipts_report_the_adopt_verdict():
    rep = EP.audit()
    for tag, verdict in rep["verdicts"].items():
        assert verdict == "ADOPT_26.10_MEASURED", tag


def test_measured_tolerances_equal_the_receipts():
    """Every pinned tolerance must equal the receipt's value, exactly."""
    for tag, (rel, _pinned) in EP.RECEIPTS.items():
        path = os.path.join(os.path.dirname(os.path.abspath(EP.__file__)), rel)
        with open(path, "rb") as fh:
            d = json.loads(fh.read().decode("utf-8"))
        live = {
            float(r["beta"]): r["tolerance"]
            for r in d["real"]
            if r.get("tolerance") is not None
        }
        for beta, tol in EP.MEASURED_TOLERANCE[tag].items():
            assert live[beta] == tol, (tag, beta, live.get(beta), tol)


def test_26_10_beats_the_sealed_8_0_at_both_densities():
    """The measured claim, per receipt. This is the whole basis for the change."""
    for tag, tols in EP.MEASURED_TOLERANCE.items():
        assert tols[EP.ADOPTED_BETA] > tols[EP.SEALED_BETA], tag


def test_26_10_is_not_the_optimum():
    """The claim the record must NOT make. beta>=64 reaches 6.0 at M=64."""
    assert EP.MEASURED_TOLERANCE["m64"][64.0] > EP.MEASURED_TOLERANCE["m64"][EP.ADOPTED_BETA]
    assert "optimum" in EP.NOT_MEASURED_CLAIM


def test_status_is_provisional_and_says_so():
    st = EP.status()
    assert st["provisional"] is True
    assert "PROVISIONAL" in st["note"]
    assert st["receipts_ok"] is True


def test_gate_blocks_the_scale_actually_measured():
    """Fail-closed on the real, sub-scale measurements -- the core assertion."""
    for m, d in EP.MEASURED_AT:
        with pytest.raises(EP.PromotionGated) as e:
            EP.assert_promotion_allowed(m, d)
        assert "PROMOTION_GATED" in str(e.value)


def test_gate_blocks_even_the_production_d_at_low_m():
    """d alone is not sufficient: the M contract is the binding one here."""
    with pytest.raises(EP.PromotionGated):
        EP.assert_promotion_allowed(EP.PROMOTION_M - 1, EP.PROMOTION_D)


def test_gate_blocks_low_d_at_full_m():
    with pytest.raises(EP.PromotionGated):
        EP.assert_promotion_allowed(EP.PROMOTION_M, EP.PROMOTION_D - 1)


def test_gate_is_merely_closed_not_broken():
    """A conforming measurement must PASS, otherwise the gate can never open.

    Without this the previous tests would also pass for a gate that always raises --
    a gate that cannot open is the mirror of a gate that cannot fail.
    """
    assert EP.assert_promotion_allowed(EP.PROMOTION_M, EP.PROMOTION_D) is None


def test_default_egress_beta_matches_the_adopted_constant():
    from henri_hopfield_egress import CanonicalCodebookEgress

    e = CanonicalCodebookEgress(dim=32)
    assert abs(e.cleanup.beta - EP.ADOPTED_BETA) < 1e-9
