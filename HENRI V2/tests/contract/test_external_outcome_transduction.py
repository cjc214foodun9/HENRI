"""Channel tests for external-outcome transduction (blueprint Action 3).

WHAT IS BEING TESTED, AND WHAT IS NOT
    These tests verify the CHANNEL: that a measured scalar external delta can be
    moved into the wave's phase geometry per-dimension, is bounded, is invertible,
    and NEVER travels as a scalar rotor (which would be a U(1) gauge no-op).

    They do NOT show that the channel improves task performance, and a passing
    round-trip is NOT evidence that the representation encodes task structure --
    the encode and the decode share the deliberate ramp construction. That
    circularity is stated in the module and must not be relabelled as capability.

THE BLUEPRINT'S FORMULA IS NOT THE TESTED OPERATOR
    The blueprint writes an additive bundling followed by sphere projection. This
    project's measured and contract-tested operator is per-dimension PHASE
    ROTATION. These tests pin the measured operator and would fail if the code
    silently became the untested one.
"""
import math
import pathlib
import sys

import pytest
import torch

R = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
from arc_egress_contract import (  # noqa: E402
    DELTA_SCALE,
    ProbeContractViolation,
    ScalarRotorRejected,
    apply_wave_binding,
    delta_ramp,
    recover_delta_from_wave,
    reject_scalar_rotor,
    transduce_external_outcome,
)

NUM_BLOCKS = 8192
BLOCK_DIM = 8
D_MODEL = NUM_BLOCKS * BLOCK_DIM
LATENT = D_MODEL // 2


def _wave(num_blocks=NUM_BLOCKS, block_dim=BLOCK_DIM, seed=7):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(num_blocks, block_dim, generator=g)


# ------------------------------------------------------- scalar rotor rejection

def test_scalar_rotor_is_rejected():
    """A numel-1 rotor is a gauge transformation: it must raise, not apply."""
    with pytest.raises(ScalarRotorRejected):
        reject_scalar_rotor(torch.tensor([0.5]), LATENT)


def test_scalar_rotor_rejected_through_the_real_transduction_path():
    """The guard must sit ON the production path, not only in a unit call."""
    with pytest.raises(ScalarRotorRejected):
        apply_wave_binding(_wave(), torch.tensor([1.0]))


def test_mismatched_rotor_length_is_rejected():
    with pytest.raises(ScalarRotorRejected):
        reject_scalar_rotor(torch.zeros(LATENT - 1), LATENT)


def test_delta_ramp_is_never_a_scalar_rotor():
    """The delta is spread across latent dims, so the guard cannot fire on it."""
    for delta in (0.0, 1.0, -1.0, 100.0):
        r = delta_ramp(delta, LATENT, action_id=3)
        assert r.numel() == LATENT
        assert r.numel() != 1
        reject_scalar_rotor(r, LATENT)  # must not raise


# --------------------------------------------------------------- ramp geometry

def test_delta_ramp_is_bounded_by_scale_times_delta():
    for delta in (0.25, 1.0, -2.0):
        r = delta_ramp(delta, LATENT, action_id=1, key_seed=101)
        assert float(r.abs().max().item()) <= DELTA_SCALE * abs(delta) + 2 * math.pi + 1e-5


def test_delta_ramp_rejects_non_finite_delta():
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ProbeContractViolation):
            delta_ramp(bad, LATENT, action_id=0)


def test_delta_ramp_rejects_tiny_latent():
    with pytest.raises(ProbeContractViolation):
        delta_ramp(1.0, 1, action_id=0)


def test_delta_ramp_is_deterministic_and_action_keyed():
    a = delta_ramp(0.5, LATENT, action_id=2)
    b = delta_ramp(0.5, LATENT, action_id=2)
    c = delta_ramp(0.5, LATENT, action_id=3)
    assert torch.equal(a, b), "same inputs must give the same ramp"
    assert not torch.allclose(a, c), "different action keys must differ"


def test_zero_delta_still_carries_the_action_key():
    """dS = 0 must not collapse to the identity: the key phase still applied."""
    r = delta_ramp(0.0, LATENT, action_id=5)
    assert float(r.abs().sum().item()) > 0.0


# -------------------------------------------------------------------- channel

def test_transduction_preserves_unit_norm_geometry():
    w = _wave()
    out = transduce_external_outcome(w, action_id=1, delta=1.0)
    assert out.shape == w.shape
    n = float(torch.linalg.vector_norm(out.reshape(-1), ord=2).item())
    assert abs(n - 1.0) < 1e-5, f"output must stay unit-normalized, got {n}"


def test_transduction_fails_closed_on_odd_wave_length():
    """An odd flat length cannot be split into (cos, sin) halves."""
    with pytest.raises(ProbeContractViolation):
        transduce_external_outcome(torch.randn(5), 0, 1.0)


def test_recovery_rejects_a_mismatched_reference():
    w = _wave()
    with pytest.raises(ProbeContractViolation):
        recover_delta_from_wave(w, 0, reference=torch.randn(4, 8))


def test_info_reports_operator_and_non_scalar_rotor():
    _, info = transduce_external_outcome(_wave(), 2, 1.0, return_info=True)
    assert info["is_scalar_rotor"] is False
    assert info["ramp_len"] == LATENT
    assert info["operator"] == "per_dimension_phase_rotation"
    assert info["action_id"] == 2


def test_transduction_changes_the_wave_for_nonzero_delta():
    """A real per-dimension rotation must move the state (unlike a gauge no-op)."""
    w = _wave(seed=11)
    out = transduce_external_outcome(w, action_id=1, delta=1.0)
    flat_in = w.reshape(-1).to(torch.float32)
    flat_out = out.reshape(-1).to(torch.float32)
    assert not torch.allclose(flat_in, flat_out, atol=1e-6)


# -------------------------------------------------------------- invertibility

def test_delta_round_trips_within_tolerance():
    """The encode must be falsifiable by its own decode.

    The reference wave is REQUIRED: the delta is carried by the phase increment,
    so a single post-transduction wave cannot determine it (see the docstring;
    the no-reference version measured a false 0.3816 for a true delta of 0.0).
    """
    w = _wave(seed=3)
    for action_id in (0, 1, 7):
        for delta in (0.0, 0.5, -0.5, 1.0, -1.0, 2.5):
            out = transduce_external_outcome(w, action_id=action_id, delta=delta)
            got = recover_delta_from_wave(out, action_id=action_id, reference=w)
            assert abs(got - delta) < 1e-3, (
                f"delta {delta} action {action_id} round-tripped to {got}"
            )


def test_no_reference_recovery_is_conditional_and_must_not_be_trusted():
    """Pins the defect that motivated the reference requirement: without a
    reference the recovery inherits the wave's own phase and is WRONG. This test
    documents the limitation rather than hiding it."""
    w = _wave(seed=3)
    out = transduce_external_outcome(w, action_id=0, delta=0.0)
    blind = recover_delta_from_wave(out, action_id=0)
    assert abs(blind) > 1e-2, "expected the unreferenced path to be unreliable"


def test_round_trip_is_scale_invariant():
    """Recovery must not depend on the chosen DELTA_SCALE (uses the same value)."""
    w = _wave(seed=5)
    out = transduce_external_outcome(w, action_id=1, delta=1.0, scale=0.01)
    got = recover_delta_from_wave(out, action_id=1, reference=w, scale=0.01)
    assert abs(got - 1.0) < 1e-3


def test_wrong_action_key_gives_a_wrong_recovery():
    """Keyed decode: decoding with the wrong action key must NOT return the delta.
    This is what makes the action id part of the channel rather than decoration."""
    w = _wave(seed=9)
    out = transduce_external_outcome(w, action_id=1, delta=1.0)
    wrong = recover_delta_from_wave(out, action_id=6, reference=w)
    assert abs(wrong - 1.0) > 1e-2
