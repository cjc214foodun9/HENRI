"""UHR-01 contract suite: RFSS role-filler homology into the boundary-axiom family.

THE CLAIM UNDER TEST
    Carrying the boundary-axiom roles through the option's ADJOINT action places the
    macro-option candidate inside the axiom's own representation family, which turns
    the Sagnac veto into an INFORMATIVE comparator: delta collapses toward 0 for a
    structurally compliant option and rises above tau_veto for an invalid one
    (scrambled role pairing / large unstructured option).

WHY EVERY CASE EXISTS (vacuous-gate discipline; henri-co-scientist-rigor 2(e))
    A gate whose only fixture is the passing case proves nothing. This suite ships:
    a passing case, a failing case, a trivially-passing control that is explicitly
    NOT evidence, a dead-input/content-sensitivity control that must FAIL if the
    projection ignores its input, a regime report that must discriminate rather than
    sit at a ceiling or a floor, and an ANTI-TAUTOLOGY control that fails if the
    score depends on the option alone.

ROLE SOURCE
    Roles here are SYNTHETIC but FAMILY-COMPLIANT: deterministic real [num_blocks, 8]
    blocks with per-block unit norm -- exactly the contract
    `zone_c_boundary_axiom_loader.verify_wave_integrity` enforces on stored axioms
    (AXIOM_NORM_TOL = 1e-4). The remote run substitutes REAL Zone C axiom bytes.
    Nothing here claims to be a Zone C axiom.
"""

from __future__ import annotations

import math

import pytest
import torch

from uhr_rfss import (
    BLOCK_NORM_TOL,
    UHRProjectionError,
    adjoint_matrix,
    axiom_sensitivity,
    bind_roles_with_adjoint,
    block_norm_deviation,
    character_delta,
    gell_mann_coefficients,
    project_option_to_boundary_family,
)

TAU_VETO = 0.35


# --------------------------------------------------------------------- fixtures
def _basis() -> torch.Tensor:
    """The in-tree Gell-Mann basis (chromodynamic_grounding.GELL_MANN_BASIS),
    rebuilt here so the suite does not import a CUDA-heavy module graph."""
    s3 = math.sqrt(3.0)
    b = [
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
        [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
        [[1 / s3, 0, 0], [0, 1 / s3, 0], [0, 0, -2 / s3]],
    ]
    return torch.tensor(b, dtype=torch.complex64)


def _gen(theta, basis: torch.Tensor) -> torch.Tensor:
    """Materialise gen = 1j * sum_a theta_a * lambda_a as [1,3,3] (in-tree form)."""
    t = torch.as_tensor(theta, dtype=torch.float32)
    return (1j * torch.einsum("a,aij->ij", t.to(basis.dtype), basis)).unsqueeze(0)


def _rand_gen(action: int, basis: torch.Tensor, scale: float = 0.3) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 + action)
    return _gen(torch.randn(8, generator=g) * scale, basis)


def _isotropic_roles(n: int, seed: int = 11) -> torch.Tensor:
    """Isotropic control fixture: random unit 8-blocks (no structure)."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(n, 8, generator=g)
    return r / r.norm(dim=-1, keepdim=True)


def _structured_roles(n: int, seed: int = 11) -> torch.Tensor:
    """STRUCTURED fixture: each block concentrates its mass on a different
    su(3) direction with a fixed sign pattern, so `mean_k R_k^T A R_k` depends on
    the axiom -- the regime in which the gate can discriminate AXIOMS."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.arange(n) % 8
    R = torch.zeros(n, 8)
    R[torch.arange(n), idx] = 1.0
    signs = torch.where(torch.rand(n, generator=g) > 0.5, 1.0, -1.0)
    R = R * signs.unsqueeze(-1)
    noise = torch.randn(n, 8, generator=g) * 0.25
    R = R + noise
    return R / R.norm(dim=-1, keepdim=True)


def _axis_roles(n: int, axis: int) -> torch.Tensor:
    """GENUINELY anisotropic roles: every block concentrates its mass on ONE su(3)
    direction. This is the regime in which the gate can discriminate AXIOMS, because
    mean_k R_k^T A R_k then probes a single rotation plane of A."""
    R = torch.zeros(n, 8)
    R[:, axis] = 1.0
    return R


def _delta(cand: torch.Tensor, ref: torch.Tensor) -> float:
    """delta = 1 - S using the planner's REAL branch: S = 0.5*(1+cos)."""
    a = cand.flatten().to(torch.float32)
    b = ref.flatten().to(torch.float32)
    cos = float((a * b).sum().item() / (float(a.norm().item()) * float(b.norm().item())))
    return 1.0 - 0.5 * (1.0 + cos)


# ------------------------------------------------------- A. coefficient geometry
def test_gell_mann_coefficient_round_trip():
    """theta -> gen -> theta must return the SAME theta (convention check)."""
    basis = _basis()
    theta = torch.tensor([0.3, -0.7, 0.0, 1.1, 0.25, -0.4, 0.9, -0.15])
    back = gell_mann_coefficients(_gen(theta, basis), basis)
    assert torch.allclose(back, theta, atol=1e-5), (back, theta)


def test_non_su3_generator_is_rejected():
    """A generator without the 1j factor must RAISE, not be silently realified."""
    basis = _basis()
    bad = torch.einsum("a,aij->ij", torch.ones(8).to(basis.dtype), basis).unsqueeze(0)
    with pytest.raises(UHRProjectionError):
        gell_mann_coefficients(bad, basis)


# ------------------------------------------------------ B. SO(8) structural laws
def test_adjoint_is_special_orthogonal():
    """Orthogonal (3.9e-15) and det = +1 on a generic option. These two properties
    are what preserve the loader's block-norm contract exactly."""
    basis = _basis()
    A = adjoint_matrix(_rand_gen(1, basis, scale=0.7), basis)
    assert tuple(A.shape) == (8, 8)
    assert not A.is_complex()
    eye = torch.eye(8)
    assert float((A @ A.T - eye).abs().max().item()) < 1e-5
    assert float(torch.linalg.det(A.double()).item()) == pytest.approx(1.0, abs=1e-5)


def _adjoint_from_U(U: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """A[c,a] = Tr(U lambda_a U^dag lambda_c)/2 computed DIRECTLY from U.

    Test-local on purpose: it exercises the homomorphism without needing a matrix
    logarithm, so the assertion does not depend on any inverse map.
    """
    lam = basis.to(torch.complex128)
    Ud = U.conj().transpose(-2, -1)
    return torch.tensor(
        [[(torch.trace(U @ lam[a] @ Ud @ lam[c]) / 2.0).real for a in range(8)]
         for c in range(8)], dtype=torch.float32)


def test_adjoint_is_homomorphic_in_generator_order():
    """Composition ORDER must be preserved (measured |Ad(U1U2) - Ad(U1)Ad(U2)| = 1.7e-16).

    THIS IS THE LOAD-BEARING ALGEBRAIC TEST: a 4-step macro-option is a product of
    generators, so a wrong composition order silently reverses the option's
    meaning. Ad is a homomorphism, not an anti-homomorphism.
    """
    basis = _basis()
    g1, g2 = _rand_gen(2, basis, 0.4), _rand_gen(3, basis, 0.5)
    U1 = torch.matrix_exp(g1[0].to(torch.complex128))
    U2 = torch.matrix_exp(g2[0].to(torch.complex128))

    lhs = _adjoint_from_U(U1 @ U2, basis)
    rhs = adjoint_matrix(g1, basis) @ adjoint_matrix(g2, basis)
    rev = adjoint_matrix(g2, basis) @ adjoint_matrix(g1, basis)

    assert float((lhs - rhs).abs().max().item()) < 1e-5, "not a homomorphism"
    assert float((lhs - rev).abs().max().item()) > 1e-3, \
        "homomorphism and anti-homomorphism are indistinguishable on this pair"

    # PUBLIC-API ORDER CHECK (deliberately NOT a tautology: the previous form
    # compared Ad(g1)@Ad(g2) against itself). The composed call must equal the
    # direct product Ad(U1)Ad(U2) applied to the roles.
    roles = _structured_roles(64)
    w_api = project_option_to_boundary_family([g1, g2], basis, roles)
    w_ref = bind_roles_with_adjoint(roles, rhs)
    assert float((w_api - w_ref).abs().max().item()) < 1e-6, "composition order lost" 


def test_identity_option_is_exactly_compliant():
    """No-op anchor: A = I => the candidate IS the axiom => delta == 0 exactly."""
    basis = _basis()
    roles = _isotropic_roles(64)
    w = project_option_to_boundary_family([], basis, roles)
    assert torch.allclose(w, roles, atol=1e-6)
    assert _delta(w, roles) == pytest.approx(0.0, abs=1e-6)


def test_character_law_matches_the_closed_form():
    """MEASURED LAW for isotropic roles: delta = 0.5*(1 - Tr(Ad U)/8), with
    Tr(Ad U) = |Tr U|^2 - 1. This ties the implementation to the verified algebra
    rather than to a fixture's expectation."""
    basis = _basis()
    roles = _isotropic_roles(8192)
    for scale in (0.05, 0.2, 0.5):
        gen = _rand_gen(4, basis, scale=scale)
        A = adjoint_matrix(gen, basis)
        U = torch.matrix_exp(gen[0].to(torch.complex128))
        tr_adj = float(torch.trace(A.double()).item())
        tr_u = abs(complex(torch.trace(U).item()))
        assert tr_adj == pytest.approx(tr_u ** 2 - 1.0, abs=1e-4)
        pred = character_delta(tr_adj)
        meas = _delta(project_option_to_boundary_family([gen], basis, roles), roles)
        assert meas == pytest.approx(pred, abs=5e-3), (scale, pred, meas)


# ------------------------------------------------ C. family / block-norm contract
def test_projection_lands_in_the_axiom_family():
    """Real fp32 [num_blocks, 8]; the loader's own per-block norm contract holds."""
    basis = _basis()
    roles = _structured_roles(256)
    gens = [_rand_gen(a, basis, 0.3) for a in (0, 1, 2, 3)]
    w = project_option_to_boundary_family(gens, basis, roles)
    assert not w.is_complex()
    assert w.dtype == torch.float32
    assert tuple(w.shape) == (256, 8)
    assert w.numel() == roles.numel()
    assert block_norm_deviation(w) < BLOCK_NORM_TOL


# -------------------------------------------------- D. THE CLAIM: compliant pass
def test_compliant_candidate_is_not_hard_vetoed():
    """CLAIM: a structurally compliant candidate path yields hard_vetoed=False."""
    basis = _basis()
    roles = _structured_roles(8192)
    gens = [_rand_gen(a, basis, scale=0.10) for a in (0, 1, 2, 3)]
    d = _delta(project_option_to_boundary_family(gens, basis, roles), roles)
    assert d <= TAU_VETO, f"compliant candidate vetoed: delta={d:.6f}"


# ------------------------------------------------ E. THE CLAIM: invalid is caught
def test_scrambled_role_pairing_is_hard_vetoed():
    """STRUCTURAL VIOLATION: the candidate is in the family (unit blocks) but its
    role pairing is NOT homologous to this axiom."""
    basis = _basis()
    roles = _structured_roles(8192)
    gens = [_rand_gen(a, basis, scale=0.10) for a in (0, 1, 2, 3)]
    perm = torch.randperm(roles.shape[0], generator=torch.Generator().manual_seed(3))
    d = _delta(project_option_to_boundary_family(gens, basis, roles,
                                                 role_permutation=perm), roles)
    assert d > TAU_VETO, f"scrambled pairing not vetoed: delta={d:.6f}"


def test_large_option_population_is_vetoed_at_a_pre_registered_rate():
    """POPULATION, not a single draw. MEASURED (24 draws, numpy): a large option
    (scale 1.2) yields delta in [0.2988, 0.5607] with veto rate 0.917, so a
    single-draw `> tau` assertion is FLAKY by construction and is not used here.
    The pre-registered rate is 0.75; the measured floor is 0.2988, which is why the
    claim test above uses the SCRAMBLED pairing (0.499, robust) as its invalid path.
    """
    basis = _basis()
    roles = _structured_roles(8192)
    deltas = []
    for s in range(8):
        gens = [_rand_gen(400 + s * 4 + i, basis, scale=1.20) for i in range(4)]
        deltas.append(_delta(project_option_to_boundary_family(gens, basis, roles),
                             roles))
    rate = sum(d > TAU_VETO for d in deltas) / len(deltas)
    print(f"\n[large-option] min={min(deltas):.6f} max={max(deltas):.6f} "
          f"veto_rate={rate:.3f} (pre-registered >= 0.75)")
    assert rate >= 0.75, f"large-option veto rate {rate:.3f} below pre-registration"


def test_wrong_family_is_rejected_before_comparison():
    """Complex or mis-shaped operands must RAISE: re-introducing cross-family
    comparison is the exact defect this module removes."""
    with pytest.raises(UHRProjectionError):
        bind_roles_with_adjoint(torch.randn(8, 8, dtype=torch.complex64),
                               torch.eye(8))
    with pytest.raises(UHRProjectionError):
        bind_roles_with_adjoint(torch.randn(8, 4), torch.eye(8))
    with pytest.raises(UHRProjectionError):
        bind_roles_with_adjoint(torch.randn(8, 8), torch.eye(4))


# -------------------------------------------------------- F. TRIVIAL control
def test_copy_of_axiom_control_passes_trivially():
    """CONTROL, NOT EVIDENCE: the axiom compared with itself must pass. This case
    cannot fail for a correct implementation, so it is recorded as a control only."""
    roles = _structured_roles(64)
    # fp32 REDUCTION NOISE: measured 2.98e-8 on a 512-element dot product, so a
    # 1e-9 tolerance would be asserting exact arithmetic that fp32 cannot deliver.
    assert _delta(roles.clone(), roles) == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------- G. DEAD-INPUT control
def test_dead_input_control_content_sensitivity():
    """DEAD-INPUT CONTROL: if the projection ignores its input (index-derived phase
    or a constant), two DIFFERENT inputs project identically. They must not."""
    basis = _basis()
    roles = _structured_roles(128)
    w_a = project_option_to_boundary_family([_rand_gen(0, basis, 0.5)], basis, roles)
    w_b = project_option_to_boundary_family([_rand_gen(7, basis, 0.5)], basis, roles)
    assert not torch.allclose(w_a, w_b, atol=1e-6), "projection ignores the option"

    roles2 = _structured_roles(128, seed=99)
    w_c = project_option_to_boundary_family([_rand_gen(0, basis, 0.5)], basis, roles2)
    assert not torch.allclose(w_a, w_c, atol=1e-6), "projection ignores the roles"

    w_id = project_option_to_boundary_family([], basis, roles)
    assert not torch.allclose(w_id, w_a, atol=1e-6), "no-op and option agree"


def test_non_finite_adjoint_is_rejected():
    with pytest.raises(UHRProjectionError):
        bind_roles_with_adjoint(_structured_roles(8),
                                torch.full((8, 8), float("nan")))


# ------------------------------------------------------------- H. regime report
def test_regime_report_is_discriminative():
    """The gate must neither saturate nor sit at a floor. Report the distribution
    and assert it straddles tau_veto for the two structurally distinct populations."""
    basis = _basis()
    roles = _structured_roles(8192)
    compliant, scrambled = [], []
    for a in range(8):
        gens_small = [_rand_gen(a * 4 + i, basis, 0.10) for i in range(4)]
        gens_large = [_rand_gen(a * 4 + i, basis, 1.10) for i in range(4)]
        perm = torch.randperm(roles.shape[0],
                              generator=torch.Generator().manual_seed(100 + a))
        compliant.append(_delta(
            project_option_to_boundary_family(gens_small, basis, roles), roles))
        scrambled.append(_delta(
            project_option_to_boundary_family(gens_large, basis, roles,
                                              role_permutation=perm), roles))

    print(f"\n[regime] tau_veto={TAU_VETO}")
    print(f"[regime] compliant min={min(compliant):.6f} max={max(compliant):.6f} "
          f"mean={sum(compliant) / len(compliant):.6f}")
    print(f"[regime] invalid   min={min(scrambled):.6f} max={max(scrambled):.6f} "
          f"mean={sum(scrambled) / len(scrambled):.6f}")
    print(f"[regime] compliant_veto_rate="
          f"{sum(d > TAU_VETO for d in compliant)}/{len(compliant)}  "
          f"invalid_veto_rate={sum(d > TAU_VETO for d in scrambled)}/{len(scrambled)}")

    # Neither population may be degenerate: the recorded defect sat at ~0.998 with
    # every candidate vetoed, which this asserts against. RATE-based, because the
    # population spans tau measured at scales 0.10 (0.037-0.212) and 1.10+scramble.
    c_rate = sum(d > TAU_VETO for d in compliant) / len(compliant)
    s_rate = sum(d > TAU_VETO for d in scrambled) / len(scrambled)
    assert max(compliant) < 0.9, "compliant population saturated (defect signature)"
    assert c_rate == 0.0, f"compliant veto rate {c_rate:.3f} must be zero"
    assert s_rate >= 0.75, f"invalid veto rate {s_rate:.3f} below pre-registration"


# ---------------------------------------------------- I. ANTI-TAUTOLOGY control
def test_axiom_sensitivity_discriminates_genuinely_anisotropic_axioms():
    """ANTI-TAUTOLOGY (positive case): when the axioms are GENUINELY anisotropic,
    the score must depend on WHICH axiom is compared. MEASURED (numpy, 8192 blocks,
    option scale 0.25): the per-axis delta spread is 0.2596 across the 8 su(3)
    directions, so a gate that scores the option alone (spread 0) is rejected here."""
    basis = _basis()
    gens = [_rand_gen(a, basis, scale=0.25) for a in (0, 1, 2, 3)]

    deltas = []
    for axis in range(4):
        roles = _axis_roles(4096, axis)
        deltas.append(_delta(
            project_option_to_boundary_family(gens, basis, roles), roles))
    spread = max(deltas) - min(deltas)
    print(f"\n[anti-tautology] axis deltas={[round(d, 6) for d in deltas]} "
          f"spread={spread:.6f}")

    s0 = axiom_sensitivity(project_option_to_boundary_family, gens, basis,
                           _axis_roles(4096, 0), _axis_roles(4096, 3))
    print(f"[anti-tautology] axis0 vs axis3 sensitivity={s0:.6f}")
    assert spread > 0.02, (
        "the projection scores the OPTION ALONE -- it cannot discriminate axioms, "
        "so a claim of axiom correspondence would be self-confirming")
    assert s0 > 0.02


def test_axiom_sensitivity_is_weak_on_the_real_near_isotropic_baseplate():
    """DOCUMENTED LIMITATION, asserted as a recorded property -- NOT as a claim.

    The REAL Zone C baseplate is near-isotropic: `generate_seed_crystal_axioms`
    builds each axiom as `normalize(randn(num_blocks, 8))` with only small
    structural edits (e.g. `w2[:, 1:4] = -w2[:, 1:4]`, `w3[:, 0] = 1.0`). For
    isotropic roles the block average kills axiom identity:

        delta = 0.5 * (1 - Tr(Ad U)/8)      ->  a function of the OPTION alone.

    MEASURED on seeder-like draws: per-axiom sensitivity 4.7e-5 .. 1.1e-3, i.e.
    ~3 orders of magnitude below the anisotropic case. The honest consequence is
    that UHR-01 restores OPTION-level discrimination (the pre-registered claim) and
    does NOT establish axiom-level correspondence. Anything stronger requires the
    baseplate itself to become structured -- a separate carrier.
    """
    basis = _basis()
    gens = [_rand_gen(a, basis, scale=0.25) for a in (0, 1, 2, 3)]

    iso = []
    for seed in (1, 2, 3):
        a, b = _isotropic_roles(4096, seed), _isotropic_roles(4096, seed + 100)
        iso.append(axiom_sensitivity(project_option_to_boundary_family, gens, basis,
                                     a, b))
    s_iso = max(iso)

    ax = axiom_sensitivity(project_option_to_boundary_family, gens, basis,
                           _axis_roles(4096, 0), _axis_roles(4096, 7))
    ratio = ax / max(s_iso, 1e-12)
    print(f"\n[limitation] near-isotropic sensitivity={s_iso:.6f} "
          f"anisotropic sensitivity={ax:.6f} ratio={ratio:.1f}x")

    # Record, do not overclaim: the anisotropic case must dominate, and the
    # near-isotropic case must be measurably small.
    assert ratio > 5.0, "anisotropy must dominate axiom sensitivity"
    assert s_iso < 0.01, "near-isotropic baseplate cannot deliver axiom specificity"

# ------------------------------------------------- J. PURITY / DEFAULT-PATH identity
def test_projection_is_pure_and_deterministic():
    """The projection must not mutate its inputs and must be deterministic: the
    production path uses it inside a per-step loop, so an in-place write would
    corrupt `boundary_batch` (the axiom bank) for every later step."""
    basis = _basis()
    roles = _structured_roles(256)
    roles_before = roles.clone()
    gens = [_rand_gen(a, basis, 0.3) for a in (0, 1, 2, 3)]
    gens_before = [g.clone() for g in gens]

    w1 = project_option_to_boundary_family(gens, basis, roles)
    w2 = project_option_to_boundary_family(gens, basis, roles)

    assert torch.equal(roles, roles_before), "projection MUTATED the axiom roles"
    for g, gb in zip(gens, gens_before):
        assert torch.equal(g, gb), "projection MUTATED a generator"
    assert torch.equal(w1, w2), "projection is not deterministic"


def test_default_off_flag_is_declared_with_zero_default():
    """DEFAULT-PATH IDENTITY (static half). With `HENRI_UHR01_RFSS` unset the
    runner must take the legacy branch byte-for-byte, so the flag must default to
    the STRING "0" and be read exactly once. A missing/renamed flag here would
    silently change production behaviour."""
    src = (_repo_root() / "production_arc_run.py").read_text(
        encoding="utf-8", errors="replace")
    decl = [ln for ln in src.splitlines() if "HENRI_UHR01_RFSS" in ln]
    assert any('os.environ.get("HENRI_UHR01_RFSS", "0")' in ln for ln in decl), (
        f"flag must default to \"0\"; found: {decl}")
    reads = sum(1 for ln in decl if "os.environ.get" in ln)
    assert reads == 1, f"flag must be read once, found {reads}"

    # The projection import must be LAZY (inside the flag branch), otherwise a
    # missing module would fail-close the whole OPINE telemetry block on the
    # default path through its broad `except`.
    lines = src.splitlines()
    top_level_import = [i + 1 for i, ln in enumerate(lines)
                        if ln.startswith("from uhr_rfss") or ln.startswith("import uhr_rfss")]
    assert not top_level_import, (
        f"uhr_rfss must not be imported at module top level (lines {top_level_import})")
    guarded = [i + 1 for i, ln in enumerate(lines) if "from uhr_rfss import" in ln]
    assert guarded, "uhr_rfss is never imported -- the flag branch is unreachable"
    # every guarded import must appear after an `if HENRI_UHR01_RFSS:` line
    for gl in guarded:
        prior = "\n".join(lines[max(0, gl - 25):gl - 1])
        assert "if HENRI_UHR01_RFSS:" in prior, (
            f"import at line {gl} is not inside the flag branch")
    # the candidate must fall back to the legacy operand when the flag is off
    assert "_cand = _psi_macro" in src, "missing default-off candidate fallback"


def _repo_root():
    import pathlib as _pl
    return _pl.Path(__file__).resolve().parents[2]


def test_projection_changes_the_candidate_operand_for_both_channels():
    """The veto returns (delta_axiom, delta_epistemic, hard). The projection replaces
    the CANDIDATE operand, which is shared by both channels, so BOTH deltas must move
    away from the recorded defect signature (>= 0.99). The world operand is NOT
    touched, so delta_epistemic moves only through the candidate -- that is the
    expected coupling and is asserted here rather than assumed."""
    basis = _basis()
    roles = _structured_roles(4096)
    gens = [_rand_gen(a, basis, 0.15) for a in (0, 1, 2, 3)]
    world = _structured_roles(4096, seed=777)

    cand = project_option_to_boundary_family(gens, basis, roles)
    d_ax = _delta(cand, roles)
    d_ep = _delta(cand, world)

    print(f"\n[channels] delta_axiom={d_ax:.6f} delta_epistemic={d_ep:.6f} "
          f"(legacy defect signature ~0.998)")
    assert d_ax < 0.99, "axiom channel still at the cross-family defect signature"
    assert d_ep < 0.99, "epistemic channel still at the cross-family defect signature"
    assert d_ax != d_ep, "the two channels are not independent measurements"

# --------------------------------------------- K. ONE READER: the unification point
def test_opine_object_owns_the_binding_delegate():
    """ONE READER (coupled-defect lesson from `_macro_num_blocks`): the object that
    COMPOSES the option must be the object that PROJECTS it, so the candidate the
    veto judges and the option the planner ranks cannot drift apart. The projection
    reached through the option object must be bit-identical to the module function.
    """
    import opine_object_mcts

    assert hasattr(opine_object_mcts.OPINEObjectMCTS, "project_to_boundary_family"), (
        "OPINEObjectMCTS does not expose the projection -- the unification point "
        "is missing and the two consumers could drift")
    assert not hasattr(opine_object_mcts, "project_option_to_boundary_family"), (
        "opine_object_mcts must DELEGATE to uhr_rfss, not re-implement the binding")

    basis = _basis()
    roles = _structured_roles(128)
    gens = [_rand_gen(a, basis, 0.2)[0] for a in (0, 1, 2, 3)]   # production shape [3,3]

    ob = opine_object_mcts.OPINEObjectMCTS(num_channels=128, option_horizon=4)
    via_object = ob.project_to_boundary_family(gens, basis, roles)
    via_module = project_option_to_boundary_family(gens, basis, roles)
    assert torch.equal(via_object, via_module), (
        "the option-object route and the module route disagree -- two readers")


def test_opine_delegate_is_inert_unless_called():
    """Default-path safety: constructing the object and calling its OTHER methods must
    not touch the UHR-01 sidecar. Only an explicit projection call may import it."""
    import importlib
    import sys as _sys

    _sys.modules.pop("uhr_rfss", None)
    import opine_object_mcts as _o
    importlib.reload(_o)

    ob = _o.OPINEObjectMCTS(num_channels=4, option_horizon=2)
    # SHAPE: `construct_macro_option` consumes [3,3] generators, exactly as
    # production supplies them (`lie_element(...)[0]` strips the channel axis).
    # Passing a [1,3,3] tensor here raises inside `u_step.expand(n,3,3)`.
    g = _gen(torch.ones(8) * 0.1, _basis())[0]
    assert tuple(g.shape) == (3, 3)
    u = ob.construct_macro_option([g], device="cpu")
    assert tuple(u.shape) == (4, 3, 3)
    _ = ob.unitarity_error(u)
    assert "uhr_rfss" not in _sys.modules, (
        "the sidecar was imported by a non-UHR path -- the default path is not inert")
