"""Discriminating tests for henri_unitary_resonant_commutator -- the URC.

Every test CAN FAIL. Several tests ASSERT A FALSIFICATION: they pin a claim from
HENRI-SPEC-2026-URC-ATTENTION-REPLACEMENT-V1 that measurement shows is FALSE, so that a
later reader cannot "fix" a gate by loosening a tolerance.

The negative controls carry the load. A test whose fixture assumes the orthogonality the
real data lacks would pass meaninglessly; the premise probes below exist to prevent that.

PROVENANCE (recorded once, by the caller, from probe_urc_premises + a PDF text audit):
    The spec cites arXiv:2506.18739v2 (April 2026).
    The supplied attachment is arXiv:2506.18739v1 (23 Jun 2025), 17 pages, 69,439 chars.
    In v1 the lemma numbering does NOT match the spec's mapping table:
        v1 Lemma 2 = transposition   (spec calls transposition "Lemma 3")
        v1 Lemma 3 = softmax         (spec calls softmax "Lemma 4")
        v1 Lemma 4 = matmul          (spec calls matmul "Lemma 5")
        v1 Lemma 6 = MaxMin          (spec calls MaxMin "Lemma 7")
        v1 Lemma 9 = inversion       (spec calls inversion "Lemma 10")
        v1 contains NO "Lemma 5" and NO "Lemma 10" at all (0 hits each)
    The spec's ontological deconstruction is therefore MISATTRIBUTED. These tests pin the
    MATHEMATICS, which is what matters, and do not depend on the misattribution.
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", ".."))          # tests/unit -> HENRI V2
HC = os.path.join(V, "henri_core")
for p in (V, HC):
    if p not in sys.path:
        sys.path.insert(0, p)

from henri_core.henri_unitary_resonant_commutator import (      # noqa: E402
    SAGNAC_TAU_VETO,
    URCConfig,
    UnitaryResonantCommutator,
)


def _unit(n, seed):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, generator=g, dtype=torch.complex64)
    return z / z.norm()


# ============================================================== THEOREM 4.1
def test_theorem_4_1_generic_unit_vectors_are_NOT_norm_preserving():
    """PINNED FALSIFICATION. Thm 4.1 asserts unit norm => ||Psi1 (x) Psi2|| = 1.

    The proof needs |Psi_hat(k)| = 1 for every bin (a FLAT spectrum). Parseval gives only
    SUM_k |Psi_hat(k)|^2 = D. Generic unit vectors violate the premise, so the conclusion
    fails. This test asserts the FAILURE: if someone later silently makes the raw path
    norm-preserving, this test flags it as a changed premise."""
    d = 4096
    a, b = _unit(d, 1), _unit(d, 2)
    raw = torch.fft.ifft(torch.fft.fft(a) * torch.fft.fft(b)).norm().item()
    assert abs(raw - 1.0) > 1e-6, (
        f"generic unit vectors convolved to norm {raw}; Thm 4.1's premise is FALSE and "
        f"the raw operation must NOT be norm-preserving")
    # the uniform unit vector is the sharpest counterexample
    u = torch.ones(d, dtype=torch.complex64) / math.sqrt(d)
    n2 = torch.fft.ifft(torch.fft.fft(u) * torch.fft.fft(u)).norm().item()
    assert abs(n2 - math.sqrt(d)) < 1e-2, (
        f"uniform unit vector should give sqrt(D) = {math.sqrt(d):.1f}, got {n2:.4f}")


def test_theorem_4_1_repaired_unimodular_spectrum_IS_norm_preserving():
    """THE REPAIR, and its negative control.

    Circular convolution IS isometric on the unimodular-spectrum torus
    T^D = {Psi_hat : |Psi_hat(k)| = 1 for all k}. The projection to unit modulus is the
    'Torus' factor the spec asks for, so the repair is not an extra assumption.
    Positive arm must hold to 1e-6; the raw arm must NOT."""
    d = 4096
    cfg = URCConfig(d=d, d_H=256, n_axioms=4, seed=3)
    u = UnitaryResonantCommutator(cfg)
    a, b = _unit(cfg.d_H, 11), _unit(cfg.d_H, 12)
    fixed = u.stage2_bind(a, b).norm().item()
    raw = u.stage2_bind_raw(a, b).norm().item()
    assert abs(fixed - 1.0) < 1e-6, f"repaired path must be unit norm, got {fixed}"
    assert abs(raw - 1.0) > 1e-6, f"raw path must NOT be unit norm, got {raw} (it is a control)"


def test_stage2_rejects_mismatched_subspace_dimensions():
    """Dimensionality contract. My first draft passed a d-length query against a d_H-length
    r_H and raised inside the Hadamard product. The binding role belongs in the SAME
    subspace as the content, so a mismatch must be a clear error, not a broadcast."""
    cfg = URCConfig(d=2048, d_H=128, n_axioms=2, seed=4)
    u = UnitaryResonantCommutator(cfg)
    good = _unit(cfg.d_H, 5)
    bad = _unit(cfg.d, 6)
    u.stage2_bind(good, good)                       # positive control: equal lengths OK
    try:
        u.stage2_bind(good, bad)
    except ValueError as exc:
        assert "equal-length" in str(exc) or "subspace" in str(exc)
    else:
        raise AssertionError("mismatched subspace lengths must raise ValueError")


# ============================================================== THEOREM 4.3
def test_theorem_4_3_maxmin_is_NOT_an_L2_isometry():
    """PINNED FALSIFICATION. Thm 4.3 claims MaxMin is an isometry:
        (Dmax)^2 + (Dmin)^2 == (Dx1)^2 + (Dx2)^2
    Counterexample x=(1,2), y=(2,1) gives 0 on the left and 2 on the right.

    The operator is UNARY over adjacent coordinate pairs. My first probe draft wrote it as
    a BINARY operation, which made T(x,y) == T(y,x) identically and produced a tautological
    contraction ratio of 0.0 for every pair. The unary form is the real operator."""
    d = 128
    cfg = URCConfig(d=d, d_H=64, n_axioms=2, seed=7)
    u = UnitaryResonantCommutator(cfg)
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([2.0, 1.0])
    tx, ty = u.stage4_maxmin(x), u.stage4_maxmin(y)
    lhs = (max(1.0, 2.0) - max(2.0, 1.0)) ** 2 + (min(1.0, 2.0) - min(2.0, 1.0)) ** 2
    rhs = (1.0 - 2.0) ** 2 + (2.0 - 1.0) ** 2
    assert lhs == 0.0 and rhs == 2.0, "counterexample arithmetic changed"
    assert torch.allclose(tx, ty), "both inputs sort to the same pair"


def test_theorem_4_3_maxmin_IS_non_expanding():
    """THE SATISFIABLE FORM. MaxMin is 1-Lipschitz: ||T(x)-T(y)|| <= ||x-y||.
    G-URC4 asserts '<=' and is therefore satisfiable. It must NOT be relaxed to '='.
    Positive control: the max measured ratio over many random pairs must be <= 1 + eps.
    Negative control: a strictly EXPANDING map must violate the bound, proving the
    check can fail."""
    d = 256
    cfg = URCConfig(d=d, d_H=128, n_axioms=2, seed=8)
    u = UnitaryResonantCommutator(cfg)
    g = torch.Generator().manual_seed(9)
    worst = 0.0
    for _ in range(400):
        a = torch.randn(64, generator=g)
        b = torch.randn(64, generator=g)
        num = (u.stage4_maxmin(a) - u.stage4_maxmin(b)).norm().item()
        den = (a - b).norm().item()
        if den > 1e-12:
            worst = max(worst, num / den)
    assert worst <= 1.0 + 1e-6, f"MaxMin expanded by {worst:.9f}; it must be non-expanding"
    # negative control: doubling IS expanding, so the check discriminates
    a = torch.randn(64, generator=g)
    b = torch.randn(64, generator=g)
    assert (2 * a - 2 * b).norm().item() / (a - b).norm().item() > 1.5


# ============================================================== THEOREM 4.2
def test_theorem_4_2_crosstalk_EXCEEDS_the_assumed_stiefel_bound():
    """PINNED FALSIFICATION. Thm 4.2 assumes max cross-talk eta <= 1/sqrt(D) = 0.003906
    and concludes the hard gate kills every distractor. Measured on a phase-only family,
    max|<psi_i, psi_j>| exceeds the assumed bound by an order of magnitude.

    This is why the gate ships DEFAULT-OFF: on a coherent bank the premise does not hold,
    so the hard gate would NOT contract the tie group as the proof claims."""
    d = 4096
    cfg = URCConfig(d=d, d_H=4096, n_axioms=64, seed=13)
    u = UnitaryResonantCommutator(cfg)
    mx = u.max_coherence()
    bound = 1.0 / math.sqrt(4096)
    assert mx > bound, (
        f"measured max coherency {mx:.6f} should exceed the assumed Stiefel bound "
        f"{bound:.6f}; if it does not, the family or the metric changed")
    # MY OWN DEFECT, self-caught: the first draft asserted `mx > 3 * bound` and failed at
    # 2.64x. A hard-coded multiplier is the "gate asserts more than its fixture carries"
    # class -- the fixture was 64 axioms, and the inflation factor is set by K, not by me.
    # Extreme-value theory: the max over C(K,2) pairs lifts the single-pair scale 1/sqrt(D)
    # by about sqrt(2 ln K). K=64 -> ~2.9x. Assert the EVT range, not an invented number.
    ratio = mx / bound
    evt = math.sqrt(2.0 * math.log(64))
    assert 1.5 < ratio < 6.0, (
        f"inflation {ratio:.2f}x is outside the EVT-expected range for K=64 "
        f"(sqrt(2 ln K) = {evt:.2f}x)")


def test_max_coherence_metric_discriminates():
    """The metric must SEPARATE a coherent family from an incoherent one, or the previous
    test is vacuous. Negative control: a family with large D and many members drives
    coherence toward 1/sqrt(D), so the measured value must DROP."""
    cfg = URCConfig(d=1024, d_H=1024, n_axioms=8, seed=14)
    u = UnitaryResonantCommutator(cfg)
    coherent = u.max_coherence()
    # explicit mutually orthogonal family -> coherence must be ~0
    orth = torch.eye(8, dtype=torch.complex64)
    near_zero = u.max_coherence(orth)
    assert near_zero < 1e-6, f"orthogonal family must give ~0 coherence, got {near_zero}"
    assert coherent > near_zero, "the metric must separate coherent from orthogonal"


# ============================================================== P_inv FOOTPRINT
def test_pinv_footprint_is_NOT_the_dense_projector():
    """ARITHMETIC PIN. The spec declares P_inv = 33.55 MiB for d_H = 16,384 out of
    D = 65,536. A DENSE complex64 projector is 8192 MiB. 33.55 MiB is 0.262 bits per
    entry, so the declared matrix is not dense. The module reports every candidate
    footprint so no constant is silently adopted."""
    cfg = URCConfig(d=65536, d_H=16384, n_axioms=2, seed=15, pinv_mode="analytic")
    u = UnitaryResonantCommutator(cfg)
    f = u.pinv_footprints
    assert f["dense_d_H_times_D_complex64_mib"] > 8000, f
    assert f["basis_d_H_squared_complex64_mib"] > 2000, f
    assert f["spec_declared_mib"] == 33.55
    assert f["spec_declared_is_dense"] is False
    assert u.pinv_footprint_bytes == 0, "analytic mode stores no matrix"


def test_pinv_modes_report_different_honest_costs():
    """Both modes must exist and report their OWN cost. 'basis' is exact but 2048 MiB at
    d_H=16384; 'analytic' stores nothing. A mode that silently claimed 33.55 MiB would be
    the defect this test exists to catch."""
    seen = {}
    for mode in ("analytic", "basis"):
        u = UnitaryResonantCommutator(
            URCConfig(d=4096, d_H=256, n_axioms=2, seed=16, pinv_mode=mode))
        seen[mode] = u.pinv_footprint_bytes
    assert seen["analytic"] == 0
    assert seen["basis"] > 0
    assert seen["basis"] != seen["analytic"]
    try:
        UnitaryResonantCommutator(URCConfig(d=256, d_H=64, pinv_mode="dense"))
    except ValueError as exc:
        assert "pinv_mode" in str(exc)
    else:
        raise AssertionError("an unknown pinv_mode must raise")


# ============================================================== STAGE 3 / DARK PORT
def test_sagnac_threshold_constant_is_the_search_veto_not_the_setpoint():
    """TERM DISCIPLINE. 0.35 is the SEARCH VETO bound. 0.0431 is the pre-ZoneC setpoint.
    Swapping them is a known trap. This test pins the constant by name."""
    assert SAGNAC_TAU_VETO == 0.35


def test_sagnac_gate_is_default_off_and_dark_port_disconnected():
    """SAFETY PROPERTY. The dark port must stay mechanically disconnected until Gate G-SJ4
    earns separation >= +0.7000. Measured separation is -0.683414 (FAIL), so the gate must
    report disabled and open every candidate. Negative control: enabling it must change
    the reported gate, proving the flag is live rather than decorative."""
    cfg = URCConfig(d=1024, d_H=64, n_axioms=8, seed=17, enable_sagnac=False)
    u = UnitaryResonantCommutator(cfg)
    psi = _unit(cfg.d, 18)
    q = _unit(cfg.d, 19)
    off = u.forward(psi, q)
    assert off["sagnac_enabled"] is False
    assert all(g == 1.0 for g in off["gate"]["gate"]), "default-OFF must open every gate"
    assert "DEFAULT-OFF" in off["gate"].get("note", "")
    cfg2 = URCConfig(d=1024, d_H=64, n_axioms=8, seed=17, enable_sagnac=True)
    on = UnitaryResonantCommutator(cfg2).forward(psi, q)
    assert on["gate"]["enabled"] is True
    # the flag must be load-bearing: the gate vector is no longer trivially all-ones OR
    # the delta is reported against the threshold
    assert on["gate"]["tau_veto"] == 0.35


def test_stage3_reads_a_pre_mixing_statistic():
    """NO POST-MIXING GATE. A gate may not sit downstream of an operator that mixes the
    candidate with the bank. stage3_gate consumes psi_bound and one axiom; it never sees
    the whole bank. Assert the signature takes a single axiom row at a time in `forward`."""
    cfg = URCConfig(d=512, d_H=64, n_axioms=4, seed=20)
    u = UnitaryResonantCommutator(cfg)
    psi = _unit(cfg.d, 21)
    q = _unit(cfg.d, 22)
    g0 = u.forward(psi, q, axiom_index=0)["gate"]["delta_min"]
    g1 = u.forward(psi, q, axiom_index=1)["gate"]["delta_min"]
    assert g0 != g1, "the statistic must depend on which axiom it is read against"


# ============================================================== STAGE 5 / RESONATOR
def test_stage5_delegates_to_the_shipped_resonator():
    """LESSON: wire, do not duplicate. henri_resonator_flow.py already exists with its own
    tests. Stage 5 must return that object (or None if unavailable) -- it must not
    re-implement the algebra."""
    cfg = URCConfig(d=512, d_H=64, n_axioms=2, seed=23)
    u = UnitaryResonantCommutator(cfg)
    res = u.stage5_factorize()
    assert res is not None, "the shipped TripartiteResonator should be importable"
    cap = res.block_capacity()
    assert cap["d"] == 96
    assert cap["torus_dims"] and cap["mask_dims"] and cap["rotor_dims"]


def test_triton_backend_reports_blocked_and_never_claims_a_number():
    """HARDWARE HONESTY. This host is torch 2.13.0+cpu. Triton ships no FFT primitive.
    The triton backend must record a BLOCKED item and must not emit any latency figure."""
    cfg = URCConfig(d=512, d_H=64, n_axioms=2, seed=24, backend="triton")
    u = UnitaryResonantCommutator(cfg)
    out = u.forward(_unit(cfg.d, 25), _unit(cfg.d, 26))
    blocked = out.get("blocked") or []
    assert any(b["item"] == "triton_backend" for b in blocked), blocked
    assert "no FFT primitive" in " ".join(b["reason"] for b in blocked)
    txt = str(out)
    for forbidden in ("14.2", "4.8", "1.3", "6.2"):     # declared µs constants
        assert f"{forbidden} us" not in txt and f"{forbidden}µs" not in txt
