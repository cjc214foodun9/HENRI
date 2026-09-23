"""UHR-02 contract suite: the exteroceptive transition gate.

THE CLAIM UNDER TEST
    Comparing a candidate option against the RECORDED empirical transition
    (FORM B) discriminates option CONTENT, where comparing it against the STATE
    it started from (FORM A) reduces to the option's MAGNITUDE under an
    isotropic baseplate.

    Measured anchor: `experiments/verification/uhr02_domain_control.py` using an
    EXACT equal-trace control (conjugation), |Tr U| matched to 3.58e-07:
        FORM A isotropic  sep 1.35e-04  << band 2.47e-03   NO-SEP
        FORM A structured sep 1.57e-01                     SEP
        FORM B isotropic  sep 3.64e-01                     SEP
        FORM B structured sep 4.66e-01                     SEP

NON-NEGOTIABLE CONTROLS (each must be able to FAIL)
    T3  exact-trace control: the conjugated option keeps |Tr U| but changes the
        rotation, so FORM B must separate it from truth.
    T4  magnitude-only NEGATIVE CONTROL: under an isotropic baseplate FORM A must
        NOT separate the same pair. If this test ever passes as "separated", the
        analytic derivation in the module docstring is wrong and the module must
        be withdrawn, not tuned.
    T5  dead-input control: an unnormalized operand must RAISE.
    T8  tau band: the blueprint's 0.35 must lie inside the measured
        compliant/invalid band, and the do-nothing candidate must be reported.

RUN
    python -m pytest tests/contract/test_uhr02_exteroceptive_gate.py -q
"""
from __future__ import annotations

import math
import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from uhr02_exteroceptive_gate import (  # noqa: E402
    AXIOM_BLOCK_NORM_TOL,
    DOMAIN_VIOLATION,
    FLAG_ENV,
    SELF_COMPARISON,
    TAU_BLUEPRINT,
    ad_of,
    assert_single_family,
    delta,
    exteroceptive_gate,
    flag_enabled,
    forge_edge,
    measured_bands,
    predict_next,
    relative_group_element,
    sampling_band,
)

K = 8192
_s3 = math.sqrt(3.0)
BASIS = torch.tensor([
    [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
    [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
    [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
    [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
    [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
    [[1 / _s3, 0, 0], [0, 1 / _s3, 0], [0, 0, -2 / _s3]],
], dtype=torch.complex64)
BAND = sampling_band(K)
SEP_GATE = 4.0 * BAND          # a separation must exceed 4 sigma to count


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def roles_iso(seed: int, k: int = K) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(k, 8, generator=g)
    return r / r.norm(dim=-1, keepdim=True)


def roles_struct(seed: int, coh: float = 6.0, k: int = K) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = coh * torch.randn(8, generator=g).unsqueeze(0) + torch.randn(k, 8, generator=g)
    return r / r.norm(dim=-1, keepdim=True)


def gens(seed: int, sc: float, k: int = 4) -> list:
    g = torch.Generator().manual_seed(seed)
    th = torch.randn(8, generator=g) * sc
    return [1j * torch.einsum("a,aij->ij", th.to(BASIS.dtype), BASIS) for _ in range(k)]


def conjugated_pair(gs: list, seed: int = 555):
    """U_B = V^dag U_A V : EXACTLY equal |Tr|, genuinely different rotation.

    Conjugation preserves the trace identically, so this is the only control
    that isolates CONTENT. (Do NOT binary-search a scale to match |Tr U|: that
    quantity is not monotonic in the scale for a multi-generator product, and an
    earlier draft of this work produced a confounded 'separation' that way.)
    """
    Ua = torch.eye(3, dtype=torch.complex64)
    for g in gs:
        Ua = Ua @ torch.matrix_exp(g)
    g = torch.Generator().manual_seed(seed)
    H = torch.randn(3, 3, generator=g, dtype=torch.complex64)
    H = H + H.conj().transpose(-2, -1)
    V = torch.matrix_exp(1j * H)
    Ub = V.conj().transpose(-2, -1) @ Ua @ V
    # log back to a generator sequence so the gate API is exercised unchanged
    w, Vm = torch.linalg.eig(Ub)
    L = Vm @ torch.diag(torch.log(w.to(torch.complex64))) @ torch.linalg.inv(Vm)
    return [L], float(torch.abs(Ua.trace().reshape(())))


def wave_next(state: torch.Tensor, gs: list, noise: float = 0.05, seed: int = 77) -> torch.Tensor:
    A = ad_of(gs, BASIS)
    y = predict_next(state, A)
    g = torch.Generator().manual_seed(seed)
    y = y + noise * torch.randn(y.shape, generator=g)
    return y / y.norm(dim=-1, keepdim=True)


def compose_u(gs: list) -> torch.Tensor:
    """U = prod exp(g) as (3,3) complex.

    A previous draft built this inline with a nested lambda chain and returned a
    function object, so the test failed with "'function' object has no attribute
    'trace'". Keeping it as a named helper makes the failure mode impossible.
    """
    U = torch.eye(3, dtype=torch.complex64)
    for g in gs:
        U = U @ torch.matrix_exp(g.to(torch.complex64))
    return U


def abs_trace_of(gs: list) -> float:
    U = compose_u(gs)
    assert U.shape == (3, 3), "compose_u returned %s, expected (3, 3)" % (tuple(U.shape),)
    return float(torch.abs(U.trace().reshape(())))


# --------------------------------------------------------------------------
# T1 -- family contract
# --------------------------------------------------------------------------
def test_t1_complex_operand_is_a_domain_violation():
    with pytest.raises(ValueError) as e:
        assert_single_family(torch.randn(16, 8, dtype=torch.complex64), "state_roles")
    assert DOMAIN_VIOLATION in str(e.value)


def test_t1_wrong_block_shape_is_a_domain_violation():
    with pytest.raises(ValueError) as e:
        assert_single_family(torch.randn(16, 16), "state_roles")
    assert DOMAIN_VIOLATION in str(e.value)


def test_t1_valid_family_returns_small_deviation():
    dev = assert_single_family(roles_iso(1, 64), "state_roles")
    assert dev < AXIOM_BLOCK_NORM_TOL


# --------------------------------------------------------------------------
# T2 -- adjoint composition identity
# --------------------------------------------------------------------------
def test_t2_adjoint_is_orthogonal_with_det_plus_one():
    A = ad_of(gens(4242, 0.30), BASIS)
    assert float((A @ A.T - torch.eye(8)).abs().max()) < 1e-5
    assert abs(float(torch.linalg.det(A.double())) - 1.0) < 1e-6


def test_t2_relative_group_element_of_equal_traces_is_one():
    gs = gens(4242, 0.30)
    assert abs(relative_group_element(gs, gs) - 3.0) < 1e-4  # |Tr(I)| = 3


# --------------------------------------------------------------------------
# T3 -- exact-trace control: FORM B reads CONTENT
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ref_name", ["isotropic", "structured"])
def test_t3_form_b_separates_equal_trace_options(ref_name):
    ref = roles_iso(1) if ref_name == "isotropic" else roles_struct(1, 6.0)
    gs = gens(4242, 0.30)
    other, abs_tr_other = conjugated_pair(gs)
    abs_tr_true = abs_trace_of(gs)
    # magnitudes matched by construction (conjugation preserves the trace)
    assert abs(abs_tr_true - abs_tr_other) < 1e-5

    R_rec = wave_next(ref, gs, seed=77)
    r_true = exteroceptive_gate(ref, R_rec, gs, BASIS, truth_generators=gs)
    r_other = exteroceptive_gate(ref, R_rec, other, BASIS, truth_generators=gs)

    sep = abs(r_true.delta_pred - r_other.delta_pred)
    assert sep > SEP_GATE, (
        "FORM B failed to separate an equal-|Tr U| option pair on the %s "
        "baseplate: sep=%.3e vs 4-sigma gate %.3e" % (ref_name, sep, SEP_GATE)
    )
    # the option that predicts the recorded transition must score LOWER
    assert r_true.delta_pred < r_other.delta_pred


# --------------------------------------------------------------------------
# T4 -- magnitude-only NEGATIVE CONTROL (must NOT separate under isotropic)
# --------------------------------------------------------------------------
def test_t4_form_a_is_magnitude_only_under_isotropic_baseplate():
    """The analytic prediction (9 - |Tr U|^2)/16 must reproduce, and FORM A must
    NOT separate the equal-|Tr U| pair. If this test fails, the module docstring's
    derivation is wrong."""

    iso = roles_iso(1)
    gs = gens(4242, 0.30)
    other, _ = conjugated_pair(gs)

    R_rec = wave_next(iso, gs, seed=77)
    r_true = exteroceptive_gate(iso, R_rec, gs, BASIS, truth_generators=gs)
    r_other = exteroceptive_gate(iso, R_rec, other, BASIS, truth_generators=gs)

    sep = abs(r_true.delta_state - r_other.delta_state)
    assert sep < SEP_GATE, (
        "FORM A separated an equal-|Tr U| pair on an isotropic baseplate "
        "(sep=%.3e). That contradicts Tr(Ad U) = |Tr U|^2 - 1 and means the "
        "magnitude-only derivation must be rechecked." % sep
    )


def test_t4_measured_bands_are_reproducible_shape():
    m = measured_bands(K)
    assert m["sampling_band"] == pytest.approx(BAND, rel=1e-9)
    assert m["tau_band_compliant_max"] < TAU_BLUEPRINT < m["tau_band_invalid_min"]
    assert m["formA_isotropic_sep"] < SEP_GATE < m["formB_isotropic_sep"]


# --------------------------------------------------------------------------
# T5 -- dead-input / self-comparison controls
# --------------------------------------------------------------------------
def test_t5_unnormalized_operand_raises():
    bad = torch.randn(16, 8) * 5.0
    with pytest.raises(ValueError) as e:
        assert_single_family(bad, "state_roles")
    assert DOMAIN_VIOLATION in str(e.value)


def test_t5_self_comparison_is_refused_not_scored():
    iso = roles_iso(1, 64)
    with pytest.raises(ValueError) as e:
        exteroceptive_gate(iso, iso.clone(), gens(1, 0.3), BASIS)
    assert SELF_COMPARISON in str(e.value)


def test_t5_empty_generator_sequence_raises():
    iso = roles_iso(1, 64)
    other = roles_struct(1, 6.0, 64)
    with pytest.raises(ValueError):
        exteroceptive_gate(iso, other, [], BASIS)


# --------------------------------------------------------------------------
# T6 -- write path: exteroceptive gating
# --------------------------------------------------------------------------
def test_t6_zero_ext_delta_forges_no_edge():
    s = roles_iso(1, 64)
    n = roles_struct(1, 6.0, 64)
    assert forge_edge(s, n, 0, 0.0, gens(1, 0.3), BASIS) is None


def test_t6_nonzero_ext_delta_forges_a_grounded_edge():
    s = roles_iso(1)
    gs = gens(4242, 0.30)
    n = wave_next(s, gs, noise=0.001, seed=5)
    e = forge_edge(s, n, 3, 0.25, gs, BASIS)
    assert e is not None
    assert e.action == 3
    assert e.t_priority is True
    assert e.grounded is True
    assert 0.0 <= e.delta_pred <= 1.0


def test_t6_nonfinite_ext_delta_raises():
    s = roles_iso(1, 64)
    n = roles_struct(1, 6.0, 64)
    with pytest.raises(ValueError):
        forge_edge(s, n, 0, float("nan"), gens(1, 0.3), BASIS)


# --------------------------------------------------------------------------
# T7 -- default-OFF flag
# --------------------------------------------------------------------------
def test_t7_flag_defaults_off(monkeypatch):
    monkeypatch.delenv(FLAG_ENV, raising=False)
    assert flag_enabled() is False


def test_t7_flag_enables_only_on_exact_one(monkeypatch):
    monkeypatch.setenv(FLAG_ENV, "1")
    assert flag_enabled() is True
    monkeypatch.setenv(FLAG_ENV, "true")
    assert flag_enabled() is False


# --------------------------------------------------------------------------
# T8 -- tau calibration and the do-nothing baseline
# --------------------------------------------------------------------------
def test_t8_do_nothing_baseline_is_always_reported():
    iso = roles_iso(1)
    gs = gens(4242, 0.30)
    R_rec = wave_next(iso, gs, noise=0.05, seed=77)
    r = exteroceptive_gate(iso, R_rec, gs, BASIS, truth_generators=gs)
    assert r.delta_identity > 0.0
    # the do-nothing candidate is what a failed gate would have to reject
    assert r.delta_identity > r.delta_pred


def test_t8_tau_band_contains_the_blueprint_constant():
    m = measured_bands(K)
    assert m["tau_band_compliant_max"] < TAU_BLUEPRINT < m["tau_band_invalid_min"], (
        "blueprint tau=%.2f is outside the measured compliant/invalid band "
        "(%.6f, %.6f)" % (TAU_BLUEPRINT, m["tau_band_compliant_max"], m["tau_band_invalid_min"])
    )


def test_t8_result_serializes_to_plain_floats():
    iso = roles_iso(1)
    gs = gens(4242, 0.30)
    R_rec = wave_next(iso, gs, noise=0.05, seed=77)
    d = exteroceptive_gate(iso, R_rec, gs, BASIS, truth_generators=gs).as_dict()
    import json

    json.dumps(d)  # must not raise
    assert set(d) >= {"delta_pred", "delta_state", "delta_identity", "hard_vetoed"}
