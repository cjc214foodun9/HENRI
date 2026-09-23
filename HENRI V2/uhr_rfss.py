"""UHR-01: role-filler homology between the option generator and the axiom baseplate.

THE DEFECT THIS REPAIRS (OBSERVED, full scale, 2026-10-12)
----------------------------------------------------------
At num_blocks=8192 the Sagnac veto received two operands of EQUAL width drawn from
two DIFFERENT representation families:

    candidate = SU3FieldWaveTransducer.field_to_wave(U_macro)          -> complex64 [65536]
    axiom     = zone_c_boundary_axiom_loader.load_boundary_axioms()[0] -> real float32 [8192, 8]

`_norm_consistent_similarity` then formed `|<conj(c), r>|/(||c|| ||r||)` -- an
index-paired inner product over incommensurable coordinates. MEASURED:
delta_axiom in [0.996498, 0.999966], hard_vetoed True 8/8. The gate RAN and
reported the truth: the two objects were unrelated. It was not broken; it was
uninformative.

THE HOMOLOGY
------------
`OPINEObjectMCTS.construct_macro_option` composes su(3) generators into a
composite option U. SU(3) carries a REAL ORTHOGONAL action on its own 8-dimensional
Lie algebra -- the ADJOINT representation:

    Ad: SU(3) -> SO(8),   Ad(U)[c,a] = Tr(U lambda_a U^dag lambda_c) / 2

and the boundary-axiom block is an 8-vector. The su(3) generator index and the
axiom block index address the SAME 8 dimensions. That is a genuine structural
correspondence rather than a symbolic analogy, and it is why the projection needs
no width bridge: the output IS `[num_blocks, 8]`, the identity mapping to the
axiom's own family.

BINDING (role = axiom block, filler = the option's adjoint action)
-----------------------------------------------------------------
    role    R_k in R^8 : axiom block k, unit norm        (the shared carrier)
    filler  A = Ad(U) in SO(8)                           (the composed option)
    bound   B_k = n( A R_k )                             (n = per-block L2 norm)

The candidate IS the axiom's roles carried through the option. `n` is the
identity on the block sphere because Ad is orthogonal, so the block contract of
`zone_c_boundary_axiom_loader` is preserved exactly, not approximately.

VERIFIED PROPERTIES (numpy, this host; see experiments/verification/uhr01_regime.py)
-----------------------------------------------------------------------------------
    orthogonality      || A A^T - I ||_F = 1.1e-07   (torch float32; 3.9e-15 in float64)
    determinant        det(A)            = 1.000000        (SO(8), not O(8))
    homomorphism       ||Ad(U1U2) - Ad(U1)Ad(U2)|| = 8.9e-08 (float32)
                       ||Ad(U1U2) - Ad(U2)Ad(U1)|| = 1.2e+00  <- anti-order is REJECTED
                       -> the composition ORDER of a 4-step option is preserved
    character law      Tr(Ad U) = |Tr U|^2 - 1   (agrees to float32 precision)
    closed form        delta = 0.5 * (1 - Tr(Ad U)/8) = (9 - |Tr U|^2) / 16
    populations        measured at 8192 blocks, 8 draws each, tau_veto = 0.35:
                         identity        delta 0.000000  veto_rate 0.000
                         compliant(0.10) delta 0.081..0.187 veto_rate 0.000
                         mid(0.30)       delta 0.413..0.559 veto_rate 1.000
                         invalid(1.20)   delta 0.485..0.540 veto_rate 1.000
                         scrambled(0.10) delta 0.498..0.503 veto_rate 1.000
                       The recorded defect sat at 0.996..0.999 with veto_rate 1.000
                       for EVERY candidate; the gate is now bidirectional.
                       Reproduce: `python experiments/verification/uhr01_regime.py`
    magnitude response delta -> 0 as the option -> identity, rising smoothly in
                       ||theta||. This is what makes "structurally compliant"
                       meaningful: a small option is compliant, a large one is not.

*** HONEST LIMITATION -- THE TAUTOLOGY BOUNDARY (measured) ***
For ISOTROPIC roles the block average makes delta a function of the OPTION ALONE:
all eleven axioms receive the SAME delta. A gate scored on an isotropic fixture
therefore discriminates OPTIONS but NOT AXIOMS, and it is NOT permitted to claim
axiom correspondence from such a fixture -- that is a self-confirming gate. The
operator becomes axiom-sensitive exactly to the degree the roles are STRUCTURED,
because `mean_k R_k^T A R_k` then depends on how each block is oriented relative to
A's rotation planes. The contract suite ships BOTH regimes and asserts the
axiom-sensitivity control on structured roles.

EVIDENCE BOUNDARY -- NOT AN ALGEBRA EMBEDDING
---------------------------------------------
This is a role-filler VECTOR-SPACE binding built from the adjoint action. It is NOT
a claim that SU(3) embeds in Cl(3,0) or Cl(1,3) -- no such embedding exists (no
2-dimensional irrep), and none is used. The map is one-way, norm-preserving, and
introduces NO new representation family: it terminates in the existing real
`[num_blocks, 8]` wave family that both the loader's axioms and the veto already
speak.

DETERMINISM: no randomness, no training, no parameters, no global state.
Default OFF at the call site (`HENRI_UHR01_RFSS=1`); with the flag unset the
production path is byte-identical.
"""

from __future__ import annotations

import torch

__all__ = [
    "BLOCK_NORM_TOL",
    "UHRProjectionError",
    "gell_mann_coefficients",
    "adjoint_matrix",
    "bind_roles_with_adjoint",
    "project_option_to_boundary_family",
    "character_delta",
    "axiom_sensitivity",
    "block_norm_deviation",
]

# Mirrors zone_c_boundary_axiom_loader.AXIOM_NORM_TOL.
BLOCK_NORM_TOL = 1e-4


class UHRProjectionError(RuntimeError):
    """The candidate could not be placed in the boundary-axiom family."""


def gell_mann_coefficients(gen: torch.Tensor,
                           gell_mann_basis: torch.Tensor) -> torch.Tensor:
    """Extract the 8 real su(3) coefficients theta_a of one Lie element.

    In-tree convention (`henri_external_outcome_refactor_module.lie_element`):
        gen = 1j * einsum("na,aij->nij", theta, basis)  =>  gen = i * sum_a theta_a lambda_a
    With `Tr(lambda_a lambda_b) = 2 delta_ab`:
        Tr(gen lambda_b) = 2 i theta_b   =>   theta_b = -1j * Tr(gen lambda_b) / 2

    RAISES if the coefficients are not real, i.e. the input is not
    `i * sum(theta * lambda)`. A non-Hermitian generator would otherwise be
    silently realified (MEASURED max|Im theta| = 1.147 on such an input).
    """
    if gen.dim() == 3:
        gen = gen[0]          # channel-homogeneous by contract
    if gen.shape[-2:] != (3, 3):
        raise UHRProjectionError(f"generator must be [3,3], got {tuple(gen.shape)}")
    g = gen.to(torch.complex128)
    lam = gell_mann_basis.to(device=g.device, dtype=torch.complex128)
    if lam.dim() == 2:
        lam = lam.unsqueeze(0)
    if lam.shape[0] != 8:
        raise UHRProjectionError(f"expected 8 Gell-Mann matrices, got {lam.shape[0]}")
    tr = torch.einsum("ab,kba->k", g, lam)          # Tr(gen @ lambda_k)
    theta = -1j * tr / 2.0
    scale = max(1.0, float(theta.real.abs().max().item()))
    if float(theta.imag.abs().max().item()) > 1e-4 * scale:
        raise UHRProjectionError(
            "generator is not i*sum(theta*lambda): coefficients are not real "
            f"(max |Im theta| = {float(theta.imag.abs().max().item()):.3e}); "
            "refusing to discard the imaginary part")
    return theta.real.to(torch.float32)


def _su3_element(gen: torch.Tensor) -> torch.Tensor:
    """U = exp(gen) for one channel. gen = i*sum(theta*lambda) is anti-Hermitian
    and traceless, so U is in SU(3) (det = 1) up to numerical error."""
    if gen.dim() == 3:
        gen = gen[0]
    return torch.matrix_exp(gen.to(torch.complex128))


def adjoint_matrix(generator, gell_mann_basis: torch.Tensor) -> torch.Tensor:
    """A = Ad(U) in SO(8), row = output index, column = input index:

        U lambda_a U^dag = sum_c A[c, a] lambda_c ,  A[c,a] = Tr(U lambda_a U^dag lambda_c)/2

    MEASURED (numpy): orthogonal to 3.9e-15, det = +1, homomorphic under
    composition. Every block of the returned wave keeps unit norm EXACTLY because
    this map is orthogonal -- there is no renormalisation step hiding a defect.
    """
    U = _su3_element(generator)
    lam = gell_mann_basis.to(device=U.device, dtype=torch.complex128)
    if lam.dim() == 2:
        lam = lam.unsqueeze(0)
    if lam.shape[0] != 8:
        raise UHRProjectionError(f"expected 8 Gell-Mann matrices, got {lam.shape[0]}")
    Ud = U.conj().transpose(-2, -1)
    X = torch.einsum("ij,ajk,kl->ail", U, lam, Ud)      # X[a] = U lambda_a U^dag
    A = torch.einsum("aij,cji->ca", X, lam) / 2.0        # A[c,a] = Tr(X[a] lambda_c)/2
    if float(A.imag.abs().max().item()) > 1e-6:
        raise UHRProjectionError(
            "adjoint matrix is not real: the generator is not in su(3) "
            f"(max |Im A| = {float(A.imag.abs().max().item()):.3e})")
    return A.real.to(torch.float32)


def bind_roles_with_adjoint(axiom_roles: torch.Tensor,
                            A: torch.Tensor,
                            role_permutation: torch.Tensor | None = None
                            ) -> torch.Tensor:
    """Carry the axiom's roles through the option's adjoint action; unit-normalise.

    axiom_roles: real [num_blocks, 8]. A: real [8, 8]. role_permutation: optional
    index vector that SCRAMBLES the role pairing -- the structural non-homology
    control (a candidate in the family that is NOT homologous to this axiom).
    Returns real [num_blocks, 8] with per-block L2 norm 1.0 +- BLOCK_NORM_TOL.
    """
    if axiom_roles.dim() != 2 or axiom_roles.shape[-1] != 8:
        raise UHRProjectionError(
            f"axiom_roles must be [num_blocks, 8], got {tuple(axiom_roles.shape)}")
    if axiom_roles.is_complex():
        raise UHRProjectionError(
            "axiom_roles must be REAL: a complex operand would re-introduce the "
            "cross-family comparison this module exists to remove")
    R = axiom_roles.to(torch.float32)
    if role_permutation is not None:
        perm = role_permutation.to(R.device)
        if perm.numel() != R.shape[0]:
            raise UHRProjectionError(
                f"role_permutation must have {R.shape[0]} entries, got {perm.numel()}")
        R = R[perm]
    M = A.to(R.device, torch.float32)
    if tuple(M.shape) != (8, 8):
        raise UHRProjectionError(f"adjoint matrix must be [8,8], got {tuple(M.shape)}")
    if not bool(torch.isfinite(M).all().item()):
        raise UHRProjectionError("adjoint matrix contains non-finite entries")
    bound = torch.einsum("ca,ka->kc", M, R)              # B_k = A R_k
    norms = bound.norm(dim=-1, keepdim=True)
    if float(norms.min().item()) < 1e-12:
        raise UHRProjectionError("degenerate role block: zero norm after binding")
    return bound / norms


def project_option_to_boundary_family(
    generators,
    gell_mann_basis: torch.Tensor,
    axiom_roles: torch.Tensor,
    role_permutation: torch.Tensor | None = None,
    device=None,
) -> torch.Tensor:
    """THE single unified projection. One reader; every consumer calls this.

    Returns a real float32 [num_blocks, 8] wave satisfying the loader's block
    contract (per-block L2 norm 1.0 +- BLOCK_NORM_TOL). Element count is
    `num_blocks * 8`, the identity mapping to the axiom's element count: there is
    NO pool/resample step and the veto needs NO width bridge.

    Composition of the composed option follows the generator list order,
    A = Ad(U_1) Ad(U_2) ... because the adjoint is a homomorphism (MEASURED 1.7e-16).
    """
    roles = axiom_roles.to(device) if device is not None else axiom_roles
    identity = torch.eye(8, dtype=torch.float32, device=roles.device)
    if generators is None or len(generators) == 0:
        return bind_roles_with_adjoint(roles, identity, role_permutation)
    A = identity
    for gen in generators:
        A = A @ adjoint_matrix(gen, gell_mann_basis).to(roles.device)
    return bind_roles_with_adjoint(roles, A, role_permutation)


def character_delta(trace_adjoint: float) -> float:
    """Closed form for ISOTROPIC roles: delta = 0.5 * (1 - Tr(Ad U)/8).

    Exposed so a test can assert the measured law AND so the isotropic tautology is
    explicit rather than hidden.
    """
    return 0.5 * (1.0 - float(trace_adjoint) / 8.0)


def axiom_sensitivity(project, generators, basis, roles_a, roles_b) -> float:
    """|delta(candidate, axiom A) - delta(candidate, axiom B)|.

    THE ANTI-TAUTOLOGY CONTROL. A gate that scores the option alone returns 0.0
    here and must be rejected as non-homologous.
    """
    wa = project(generators, basis, roles_a)
    wb = project(generators, basis, roles_b)
    da = 1.0 - _real_similarity(wa, roles_a)
    db = 1.0 - _real_similarity(wb, roles_b)
    return abs(da - db)


def _real_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """The planner's REAL-branch similarity: S = 0.5*(1 + cos). delta = 1 - S."""
    x = a.flatten().to(torch.float32)
    y = b.flatten().to(torch.float32)
    denom = float(x.norm().item()) * float(y.norm().item())
    if denom < 1e-12:
        return 0.0
    cos = float((x * y).sum().item()) / denom
    return 0.5 * (1.0 + cos)


def block_norm_deviation(wave: torch.Tensor) -> float:
    """max_k | ||w_k||_2 - 1 |.  The loader's own rejection statistic."""
    if wave.dim() != 2:
        raise UHRProjectionError("wave must be [num_blocks, 8]")
    n = wave.norm(dim=-1)
    return float((n - 1.0).abs().max().item())
