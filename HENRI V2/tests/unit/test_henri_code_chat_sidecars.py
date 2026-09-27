"""Contract tests for the HENRI-Code and HENRI-Chat sidecars.

Modules under test (both DEFAULT-OFF sidecars; nothing live imports them):
  henri_sagnac_type_veto.py   -- toy typed-IR phase veto (type/borrow/termination)
  henri_discourse_barrier.py  -- axiom-subspace phase barrier

Test policy: every admitted path gets a POSITIVE case, every veto path gets a
case that MUST fire, and every gate gets a DEAD-INPUT NEGATIVE CONTROL that must
NOT fire.  An untested gate is not a gate.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from henri_sagnac_type_veto import (  # noqa: E402
    DELTA_TAU,
    Ins,
    SagnacTypeVeto,
)
from henri_discourse_barrier import DiscourseBarrier, THETA_AXIOM  # noqa: E402


# ===========================================================================
# HENRI-Code: toy typed IR veto
# ===========================================================================
def _codes(v):
    return [c for _, c, _ in v.violations]


def test_type_veto_admits_a_well_typed_borrow_program():
    """POSTIVE: correct ownership + types must ADMIT with zero residual."""
    prog = [
        Ins("LET", ("a", "Int")),
        Ins("BORROW", ("a", "r")),
        Ins("USE", ("r",)),
        Ins("RELEASE", ("a", "r")),
        Ins("MOVE", ("a", "c")),
        Ins("USE", ("c",)),
        Ins("ASSERT_TYPE", ("c", "Int")),
    ]
    v = SagnacTypeVeto().check(prog)
    assert v.admitted is True, _codes(v)
    assert v.violations == []
    assert v.delta_phi <= v.tau
    assert v.bypass is False


def test_type_veto_catches_use_after_move():
    prog = [
        Ins("LET", ("a", "Int")),
        Ins("MOVE", ("a", "b")),
        Ins("USE", ("a",)),          # a was moved out
    ]
    v = SagnacTypeVeto().check(prog)
    assert v.admitted is False
    assert "USE_AFTER_MOVE" in _codes(v)


def test_type_veto_catches_type_mismatch():
    prog = [
        Ins("LET", ("a", "Int")),
        Ins("ASSERT_TYPE", ("a", "Float")),
    ]
    v = SagnacTypeVeto().check(prog)
    assert v.admitted is False
    assert "TYPE_MISMATCH" in _codes(v)
    assert v.delta_phi > v.tau


def test_type_veto_catches_move_while_borrowed():
    prog = [
        Ins("LET", ("a", "Int")),
        Ins("BORROW", ("a", "r")),
        Ins("MOVE", ("a", "c")),
    ]
    v = SagnacTypeVeto().check(prog)
    assert v.admitted is False
    assert "MOVE_WHILE_BORROWED" in _codes(v)


def test_type_veto_catches_double_borrow():
    prog = [
        Ins("LET", ("a", "Int")),
        Ins("BORROW", ("a", "r1")),
        Ins("BORROW", ("a", "r2")),
    ]
    v = SagnacTypeVeto().check(prog)
    assert v.admitted is False
    assert "BORROW_WHILE_BUSY" in _codes(v)


def test_type_veto_catches_release_not_borrowed():
    prog = [
        Ins("LET", ("a", "Int")),
        Ins("LET", ("r", "Int")),
        Ins("RELEASE", ("a", "r")),   # r is OWNED, not BORROWED
    ]
    v = SagnacTypeVeto().check(prog)
    assert v.admitted is False
    assert "RELEASE_NOT_BORROWED" in _codes(v)


def test_type_veto_catches_unbound_slot():
    v = SagnacTypeVeto().check([Ins("USE", ("ghost",))])
    assert v.admitted is False
    assert "UNBOUND_SLOT" in _codes(v)


def test_type_veto_catches_unknown_op():
    v = SagnacTypeVeto().check([Ins("FROB", ())])
    assert v.admitted is False
    assert "UNKNOWN_OP" in _codes(v)


def test_type_veto_catches_phase_neutral_loop():
    """A loop whose state digest repeats at the same label is non-terminating."""
    prog = [
        Ins("LABEL", ("top",)),
        Ins("LET", ("a", "Int")),
        Ins("JUMP", ("top",)),
    ]
    v = SagnacTypeVeto().check(prog)
    assert v.admitted is False
    assert "NON_TERMINATING" in _codes(v)


def test_type_veto_does_not_falsely_flag_a_progressing_loop():
    """A loop whose state ADVANCES must not be called non-terminating.

    It is admitted up to the step budget, and budget exhaustion is reported as
    a separate flag -- it is NOT a termination proof (documented boundary).
    """
    prog = [
        Ins("LET", ("a", "Int")),
        Ins("LABEL", ("top",)),
        Ins("ADVANCE", ("a",)),
        Ins("JUMP", ("top",)),
    ]
    v = SagnacTypeVeto(max_steps=12).check(prog)
    assert "NON_TERMINATING" not in _codes(v)
    assert v.budget_exhausted is True
    assert v.admitted is True


def test_type_veto_disabled_is_a_full_bypass():
    prog = [Ins("USE", ("ghost",)), Ins("FROB", ())]
    v = SagnacTypeVeto(enabled=False).check(prog)
    assert v.admitted is True
    assert v.bypass is True
    assert v.violations == []


def test_type_veto_dead_input_control_never_vetoes():
    """NEGATIVE CONTROL: content-blind must FAIL to veto a violating program."""
    prog = [
        Ins("LET", ("a", "Int")),
        Ins("MOVE", ("a", "b")),
        Ins("USE", ("a",)),
        Ins("FROB", ()),
    ]
    live = SagnacTypeVeto(content_blind=False).check(prog)
    blind = SagnacTypeVeto(content_blind=True).check(prog)
    assert live.admitted is False            # the real gate fires
    assert blind.admitted is True            # the control cannot
    assert blind.violations == []
    assert blind.content_blind is True


def test_type_veto_is_deterministic():
    prog = [
        Ins("LET", ("a", "Int")),
        Ins("BORROW", ("a", "r")),
        Ins("RELEASE", ("a", "r")),
        Ins("MOVE", ("a", "c")),
    ]
    a = SagnacTypeVeto().check(prog)
    b = SagnacTypeVeto().check(prog)
    assert (a.admitted, a.delta_phi, a.state_digest, a.n_steps) == \
           (b.admitted, b.delta_phi, b.state_digest, b.n_steps)


def test_type_veto_threshold_is_the_published_constant():
    assert DELTA_TAU == 1e-6
    assert SagnacTypeVeto().tau == DELTA_TAU


# ===========================================================================
# HENRI-Chat: axiom-subspace phase barrier
# ===========================================================================
def _unit(v):
    n = math.sqrt(sum(abs(z) ** 2 for z in v))
    return [z / n for z in v]


def test_axioms_are_orthonormal():
    b = DiscourseBarrier(d=64, n_axioms=4)
    norms = [math.sqrt(sum(abs(z) ** 2 for z in a)) for a in b.axioms]
    assert all(abs(n - 1.0) < 1e-12 for n in norms), norms
    for i in range(b.n_axioms):
        for j in range(i + 1, b.n_axioms):
            ip = abs(sum(x * y.conjugate() for x, y in zip(b.axioms[i], b.axioms[j])))
            assert ip < 1e-12, (i, j, ip)


def test_wellformed_query_is_admitted_with_near_zero_score():
    b = DiscourseBarrier(d=64, n_axioms=4)
    q = [0j] * b.d
    for i, s in enumerate(b.query_support):
        q[s] = complex(math.cos(i), math.sin(i))
    r = b.admit(_unit(q))
    assert r.admitted is True
    assert r.verdict == "ADMIT"
    assert r.injection_score < 1e-12


def test_injection_into_the_axiom_subspace_is_vetoed():
    for i in range(4):
        b = DiscourseBarrier(d=64, n_axioms=4)
        r = b.admit(b.axioms[i])          # axioms are already unit-norm
        assert r.admitted is False
        assert r.verdict == "INJECTION_VETO"
        assert r.injection_score > THETA_AXIOM
        assert r.injection_score == pytest.approx(1.0, abs=1e-12)


def test_axiom_store_is_unchanged_by_admission():
    b = DiscourseBarrier(d=64, n_axioms=4)
    before = b.digest()
    b.admit(b.axioms[1])
    b.admit(_unit([1j] * b.d))
    assert b.digest() == before
    r = b.admit(b.axioms[0])
    assert r.axioms_unchanged is True
    assert r.axioms_digest_before == r.axioms_digest_after


def test_discourse_disabled_is_a_full_bypass():
    b = DiscourseBarrier(d=64, n_axioms=4, enabled=False)
    r = b.admit(b.axioms[0])
    assert r.admitted is True
    assert r.bypass is True
    assert r.verdict == "BYPASS"


def test_discourse_dead_input_control_never_vetoes():
    """NEGATIVE CONTROL: content-blind must FAIL to veto a known injection."""
    live = DiscourseBarrier(d=64, n_axioms=4, content_blind=False).admit(
        DiscourseBarrier(d=64, n_axioms=4).axioms[0]
    )
    blind_b = DiscourseBarrier(d=64, n_axioms=4, content_blind=True)
    blind = blind_b.admit(blind_b.axioms[0])
    assert live.admitted is False        # the real gate fires
    assert blind.admitted is True        # the control cannot
    assert blind.injection_score == 0.0


def test_discourse_fails_closed_on_bad_input():
    b = DiscourseBarrier(d=64, n_axioms=4)
    with pytest.raises(ValueError):
        b.admit([1j] * (b.d - 1))        # wrong length
    with pytest.raises(ValueError):
        b.admit([0j] * b.d)              # zero wave has no phase
    with pytest.raises(ValueError):
        DiscourseBarrier(d=63, n_axioms=4)   # not divisible by 2*n_axioms


def test_discourse_score_is_bounded_by_cauchy_schwarz():
    b = DiscourseBarrier(d=64, n_axioms=4)
    for k in range(1, 40):
        q = [complex(math.cos(k * i), math.sin(k * i * 1.7)) for i in range(b.d)]
        r = b.admit(_unit(q))
        assert 0.0 <= r.injection_score <= 1.0


def test_discourse_theta_is_the_published_constant():
    assert THETA_AXIOM == 0.55
    assert DiscourseBarrier(d=64, n_axioms=4).theta == THETA_AXIOM


# END
