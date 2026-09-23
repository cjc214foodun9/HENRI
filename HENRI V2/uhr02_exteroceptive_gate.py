"""UHR-02: exteroceptive transition gate — the outcome-grounded comparison domain.

THE MAXIM UNDER IMPLEMENTATION
    "Fix the comparison's domain, not the comparison's threshold."

WHY THIS MODULE EXISTS (measured, not narrated)
    UHR-01's live paired A/B (instance 52189427) reported `delta_axiom == 0.0`
    for all 8 steps while the baseline reported 0.997..0.9996 with
    `hard_vetoed` 8/8. The gate did not become discriminative; it flipped from
    ALWAYS-VETO to NEVER-VETO. The cause is now measured.

    With per-block unit roles R_k in R^8 and a candidate rotation
    A = Ad(U_c) in SO(8):

        FORM A  (compare the candidate against the STATE it started from)
            cos(R_pred, R_state) = (1/K) sum_k R_k^T A R_k
                                 -> Tr(A)/8   for isotropic R_k
                                 = (|Tr U_c|^2 - 1)/8
            => delta_A = (9 - |Tr U_c|^2)/16     : OPTION MAGNITUDE ONLY

        Verified: worst |predicted - measured| = 2.95e-03 against an analytic
        1-sigma sampling band of 2.47e-03 (ratio 1.19) at K = 8192.
        See `experiments/verification/uhr02_domain_probe.py` (test T1).

    The consequence is visible with an EXACT equal-trace control (conjugation,
    U_B = V^dag U_A V: |Tr U_B| = |Tr U_A| to 3.58e-07 while the adjoint
    rotation differs by 3.43 in Frobenius norm):

        ISOTROPIC  baseplate, FORM A : separation 1.35e-04  << band   NO-SEP
        STRUCTURED baseplate, FORM A : separation 1.57e-01           SEP
        ISOTROPIC  baseplate, FORM B : separation 3.64e-01           SEP
        STRUCTURED baseplate, FORM B : separation 4.66e-01           SEP

    FORM B compares the candidate against the RECORDED transition:
        cos(R_pred, R_recorded) = (1/K) sum_k R_k^T A_c^T A_t R_k
                                -> Tr(A_c^T A_t)/8
                                = (|Tr(U_c^dag U_t)|^2 - 1)/8
    It reads the RELATIVE group element: does the candidate PREDICT what was
    empirically observed to happen. That is a DOMAIN change, not a threshold
    change, and it is the blueprint's own prescription
    (HENRI-SPEC-2026-CAUSAL-REALITY-V1 section 3.2: exteroceptive verification,
    Delta_S_ext, SOLIPSISM_VETO).

CAUSAL-TIMING CONTRACT (load-bearing; a future observation must never score a
present action)
    WRITE PATH (`forge_edge`): a directed causal edge is forged from a
        transition that has ALREADY been observed. The exteroceptive test is
        applied HERE: `ext_delta == 0` -> SOLIPSISM_VETO and NO edge is written.
    READ PATH (`exteroceptive_gate`): a candidate option is scored against the
        successor ALREADY RECORDED for the matched (state, action) pair -- past
        constant conjunction, never the future observation.

SINGLE-FAMILY CONTRACT
    Both operands are real [K, 8] block vectors. A complex operand is a domain
    violation and RAISES: comparing the complex flat [D] transducer wave against
    a real [K, 8] baseplate is the recorded defect this module exists to avoid.

FLAG
    Production wiring is gated by `HENRI_UHR02_EXTERO_GATE` (default OFF). The
    functions here are pure and carry no flag so they stay directly testable.

EVIDENCE CLASS
    The identities are DERIVED (proved by the control probe above). The
    live-loop value of the gate is HYPOTHESIS until the populated-store paired
    A/B runs.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Sequence

import torch

# --- reuse the verified adjoint machinery: one reader, not a second copy -----
try:  # pragma: no cover - import shape depends on caller packaging
    from uhr_rfss import adjoint_matrix, block_norm_deviation
except Exception:  # pragma: no cover
    _here = os.path.dirname(os.path.abspath(__file__))
    import sys

    if _here not in sys.path:
        sys.path.insert(0, _here)
    from uhr_rfss import adjoint_matrix, block_norm_deviation

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
N_BLOCKS_DEFAULT = 8192
DIM = 8
AXIOM_BLOCK_NORM_TOL = 1e-4
TAU_BLUEPRINT = 0.35
TAU_VETO = 0.35

SOLIPSISM_VETO = "SOLIPSISM_VETO"
DOMAIN_VIOLATION = "DOMAIN_VIOLATION"
SELF_COMPARISON = "SELF_COMPARISON"

FLAG_ENV = "HENRI_UHR02_EXTERO_GATE"


def flag_enabled() -> bool:
    """Production wiring gate. Default OFF: the default path stays byte-identical."""
    return os.environ.get(FLAG_ENV, "0") == "1"


def sampling_band(n_blocks: int = N_BLOCKS_DEFAULT, dim: int = DIM) -> float:
    """1-sigma band of the isotropic FORM-A identity at this block count.

    Per-block variance of R^T A R on S^{d-1} is about 2/(d+2); averaging K
    i.i.d. blocks gives sigma = sqrt(2/(d+2))/sqrt(K) on the cosine, hence half
    that on delta.
    """
    return 0.5 * math.sqrt(2.0 / (dim + 2)) / math.sqrt(n_blocks)


def measured_bands(n_blocks: int = N_BLOCKS_DEFAULT) -> dict:
    """The measured numbers this module's contracts are calibrated against.

    Every value is reproducible by `experiments/verification/uhr02_domain_control.py`.
    """
    return {
        "source": "experiments/verification/uhr02_domain_control.py",
        "exact_trace_abs_diff": 3.576e-07,
        "adjoint_frobenius_gap": 3.4280,
        "formA_isotropic_sep": 1.348e-04,
        "formA_structured_sep": 1.565e-01,
        "formB_isotropic_sep": 3.638e-01,
        "formB_structured_sep": 4.657e-01,
        "tau_band_compliant_max": 0.004355,
        "tau_band_invalid_min": 0.352165,
        "do_nothing_isotropic": 0.328260418,
        "do_nothing_structured": 0.502021222,
        "sampling_band": sampling_band(n_blocks),
    }


# ---------------------------------------------------------------------------
# family and numeric helpers
# ---------------------------------------------------------------------------
def assert_single_family(roles: torch.Tensor, name: str) -> float:
    """Enforce the real [K, 8] block family. Returns the block-norm deviation.

    RAISES on a complex operand (the recorded cross-family defect) or on a
    non-block shape, and RAISES when the block norms are not unit to tolerance.
    """
    if roles.is_complex():
        raise ValueError(
            "%s: %s complex operand %s is a domain violation; this gate compares "
            "real [K, 8] block vectors only"
            % (DOMAIN_VIOLATION, name, tuple(roles.shape))
        )
    if roles.dim() != 2 or roles.shape[1] != DIM:
        raise ValueError(
            "%s: %s has shape %s, expected [K, %d]"
            % (DOMAIN_VIOLATION, name, tuple(roles.shape), DIM)
        )
    dev = float(block_norm_deviation(roles))
    if not math.isfinite(dev) or dev > AXIOM_BLOCK_NORM_TOL:
        raise ValueError(
            "%s: %s block-norm deviation %.3e exceeds tolerance %.1e; normalize "
            "before comparing" % (DOMAIN_VIOLATION, name, dev, AXIOM_BLOCK_NORM_TOL)
        )
    return dev


def _normalize_rows(t: torch.Tensor) -> torch.Tensor:
    return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def normalize_roles(roles: torch.Tensor, name: str = "roles") -> torch.Tensor:
    """Project a real [K, 8] operand onto the UNIT block family, row by row.

    The module's strict family contract (`assert_single_family`) requires unit
    block norms. The LIVE reference `boundary_batch[0]` is the per-frame
    prediction RESIDUAL when `USE_ZONE_C_AXIOMS=0`, whose block norms are not
    unit. `delta` is a cosine, so normalizing each block is a benign,
    documented normalization of the DOMAIN (directions on S^7), not a
    threshold change. Shape/real-ness are still enforced here so a complex or
    non-block operand still raises DOMAIN_VIOLATION.
    """
    if roles.is_complex():
        raise ValueError(
            "%s: %s complex operand %s is a domain violation"
            % (DOMAIN_VIOLATION, name, tuple(roles.shape))
        )
    if roles.dim() != 2 or roles.shape[1] != DIM:
        raise ValueError(
            "%s: %s has shape %s, expected [K, %d]"
            % (DOMAIN_VIOLATION, name, tuple(roles.shape), DIM)
        )
    return _normalize_rows(roles.to(torch.float32))


def delta(a: torch.Tensor, b: torch.Tensor) -> float:
    """The gate's own residual: 0.5 * (1 - cos), on the flattened real pair."""
    x = a.detach().flatten().to(torch.float32)
    y = b.detach().flatten().to(torch.float32)
    denom = float(x.norm()) * float(y.norm())
    if denom <= 0.0:
        raise ValueError("delta: a zero-norm operand makes the residual undefined")
    return 0.5 * (1.0 - float((x * y).sum()) / denom)


def predict_next(state_roles: torch.Tensor, ad_c: torch.Tensor) -> torch.Tensor:
    """R_pred = n(Ad(U_c) R_state) in the real [K, 8] family."""
    y = torch.einsum("ij,kj->ki", ad_c.to(state_roles.dtype).to(state_roles.device),
                     state_roles)
    return _normalize_rows(y)


def ad_of(generator_sequence: Sequence[torch.Tensor], gell_mann_basis: torch.Tensor) -> torch.Tensor:
    """Ad(U) for a composed generator sequence, as an SO(8) matrix.

    DEVICE CONTRACT (measured live 2026-09-23, RFSS arm of UHR-03): the identity
    accumulator MUST be created on the SAME device as the generators and the
    basis. A CPU `torch.eye(DIM)` against a CUDA `adjoint_matrix` raises
    `RuntimeError: Expected all tensors to be on the same device, but got mat2 is
    on cuda:0, different from other tensors on cpu`. That is a CPU-only-invisible
    fault: local contract tests pass on CPU because everything is CPU.
    """
    dev = gell_mann_basis.device
    if generator_sequence:
        dev = generator_sequence[0].device
    A = torch.eye(DIM, device=dev)
    for gen in generator_sequence:
        gshape = tuple(gen.shape)
        if gshape != (3, 3):
            raise ValueError(
                "ad_of: generator must be [3,3] (one channel); got %s. A batched "
                "[N,3,3] operand must first be reduced, e.g. "
                "generators_from_displacement(disp, channel=0)." % (gshape,))
        A = A @ adjoint_matrix(gen, gell_mann_basis)
    return A


def relative_group_element(
    candidate_generators: Sequence[torch.Tensor],
    truth_generators: Sequence[torch.Tensor],
) -> float:
    """|Tr(U_c^dag U_t)| -- the analytic quantity FORM B reads.

    Exposed so a diagnostic run can show the wave-space residual next to the
    group-theoretic quantity it estimates. Both sequences must compose to SU(3).
    """
    def _U(gs):
        # DEVICE CONTRACT: same fault as ad_of (measured live, UHR-03 RFSS arm).
        # The identity must live on the generator's device.
        dev = gs[0].device
        U = torch.eye(3, dtype=torch.complex64, device=dev)
        for g in gs:
            U = U @ torch.matrix_exp(g.to(torch.complex64))
        return U

    if not candidate_generators or not truth_generators:
        raise ValueError("relative_group_element: both sequences must be non-empty")
    U_c, U_t = _U(candidate_generators), _U(truth_generators)
    return float(torch.abs((U_c.conj().transpose(-2, -1) @ U_t).trace().reshape(())))


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GateResult:
    """One candidate scored against the recorded empirical transition."""

    delta_pred: float          # FORM B: candidate vs recorded successor
    delta_state: float         # FORM A: candidate vs own starting state (diagnostic)
    delta_identity: float      # the do-nothing candidate, always reported
    hard_vetoed: bool
    relative_group_element: float | None
    n_blocks: int
    block_norm_dev: float
    reason: str | None = None
    magnitude_only_risk: bool = False

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        for k in ("delta_pred", "delta_state", "delta_identity"):
            d[k] = round(d[k], 9)
        if d["relative_group_element"] is not None:
            d["relative_group_element"] = round(d["relative_group_element"], 9)
        d["block_norm_dev"] = float("%.3e" % d["block_norm_dev"])
        return d


def exteroceptive_gate(
    state_roles: torch.Tensor,
    recorded_next_roles: torch.Tensor,
    candidate_generators: Sequence[torch.Tensor],
    gell_mann_basis: torch.Tensor,
    tau: float = TAU_BLUEPRINT,
    truth_generators: Sequence[torch.Tensor] | None = None,
) -> GateResult:
    """Score a candidate option against the RECORDED empirical successor.

    `recorded_next_roles` is the successor ALREADY observed for the matched
    (state, action) pair -- past constant conjunction, never the future. Passing
    a successor equal to the state is refused as a self-comparison, because that
    is exactly the degeneracy that produced `delta_axiom == 0.0` in the live loop.

    FORM B is `delta_pred`. FORM A (`delta_state`) is reported for diagnosis
    only and must not be used to gate: under an isotropic baseplate it collapses
    onto the option's magnitude.
    """
    dev_s = assert_single_family(state_roles, "state_roles")
    dev_n = assert_single_family(recorded_next_roles, "recorded_next_roles")
    if state_roles.shape != recorded_next_roles.shape:
        raise ValueError(
            "%s: state %s and recorded successor %s differ in shape"
            % (DOMAIN_VIOLATION, tuple(state_roles.shape), tuple(recorded_next_roles.shape))
        )
    if not candidate_generators:
        raise ValueError("exteroceptive_gate: candidate_generators is empty")

    # a self-comparison is not a transition; refuse it rather than return 0.0
    if float((recorded_next_roles - state_roles).abs().max()) == 0.0:
        raise ValueError(
            "%s: recorded successor is identical to the state; that is the "
            "self-comparison degeneracy, not an empirical transition" % SELF_COMPARISON
        )

    A_c = ad_of(candidate_generators, gell_mann_basis)
    pred = predict_next(state_roles, A_c)

    delta_pred = delta(pred, recorded_next_roles)
    delta_state = delta(pred, state_roles)
    delta_identity = delta(state_roles, recorded_next_roles)

    rge = None
    if truth_generators is not None:
        rge = relative_group_element(candidate_generators, truth_generators)

    band = sampling_band(state_roles.shape[0])
    return GateResult(
        delta_pred=delta_pred,
        delta_state=delta_state,
        delta_identity=delta_identity,
        hard_vetoed=bool(delta_pred > tau),
        relative_group_element=rge,
        n_blocks=int(state_roles.shape[0]),
        block_norm_dev=max(dev_s, dev_n),
        reason=None,
        magnitude_only_risk=bool(abs(delta_pred - delta_state) < band),
    )


# ---------------------------------------------------------------------------
# write path: forge a directed causal edge under the exteroceptive test
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EdgeRecord:
    """A directed causal edge in the Zone C engram DAG (spec section 3.2)."""

    action: int
    ext_delta: float
    delta_pred: float
    t_priority: bool
    grounded: bool

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "ext_delta": self.ext_delta,
            "delta_pred": round(self.delta_pred, 9),
            "t_priority": self.t_priority,
            "grounded": self.grounded,
        }


def forge_edge(
    state_roles: torch.Tensor,
    observed_next_roles: torch.Tensor,
    action: int,
    ext_delta: float,
    truth_generators: Sequence[torch.Tensor],
    gell_mann_basis: torch.Tensor,
    tau: float = TAU_BLUEPRINT,
) -> EdgeRecord | None:
    """Forge an edge ONLY when the environment actually moved. Else SOLIPSISM_VETO.

    The three conditions of spec section 3.2 are checked in order:
      1. temporal priority  -- this function is only reachable after observation
      2. exteroceptive verification -- `ext_delta != 0`
      3. statistical conjunction   -- the residual passes tau

    Returns an EdgeRecord, or None when the edge is refused.
    """
    dev_s = assert_single_family(state_roles, "state_roles")
    dev_n = assert_single_family(observed_next_roles, "observed_next_roles")
    if not math.isfinite(ext_delta):
        raise ValueError("forge_edge: ext_delta must be finite; got %r" % (ext_delta,))

    # condition 2: no exteroceptive movement means the 'transition' is internal
    if float(ext_delta) == 0.0:
        return None

    A_t = ad_of(truth_generators, gell_mann_basis)
    pred = predict_next(state_roles, A_t)
    dp = delta(pred, observed_next_roles)
    return EdgeRecord(
        action=int(action),
        ext_delta=float(ext_delta),
        delta_pred=dp,
        t_priority=True,
        grounded=bool(dp <= tau and max(dev_s, dev_n) <= AXIOM_BLOCK_NORM_TOL),
    )


# ---------------------------------------------------------------------------
# recorded-transition read path (the live-loop wiring, FORM B)
# ---------------------------------------------------------------------------
UNAVAILABLE_NO_RECORDED_TRANSITION = "UNAVAILABLE_NO_RECORDED_TRANSITION"


def transition_channel(store, action: int, min_norm: float = 1e-5) -> int | None:
    """The channel index whose LEARNED transition is strongest for `action`.

    CHANNEL-SEMANTICS CONTRACT (measured defect, UHR-03 kill-run #3, 2026-09-23).
    `theta_a` is CHANNEL-RESOLVED: `encode_su3_color_field(...).reshape(-1,3,3)`
    maps grid cell `(r,c)` to channel `side*r + c`, so channel 0 is ONE cell.
    My first wiring guarded on the AGGREGATE `theta_a[action].norm()` (0.750022 on
    the live store) but RETURNED `lie_element(action)[0]` -- channel 0 -- whose
    norm was 3.7e-08. The guard tested one quantity, the consumer read another, and
    16/16 telemetry records looked successful while comparing two IDENTITIES
    (`relative_group_element = 3.0` is `|Tr(I)|`). That is the signature-proof
    defect, not a mechanism failure.

    Selecting the argmax-norm channel makes the guard and the consumer read the
    SAME quantity. Ties break by index, so the choice is deterministic.
    """
    if store is None or action is None or int(action) < 0:
        return None
    with torch.no_grad():
        th = store.theta_a[int(action)]
        norms = th.norm(dim=-1) if th.dim() > 1 else th.norm().reshape(1)
        c = int(torch.argmax(norms))
        if float(norms[c]) <= min_norm:
            return None
        return c


def recorded_transition_generators(
    store, action: int, gell_mann_basis: torch.Tensor,
    min_norm: float = 1e-5, channel: int | None = None,
) -> list | None:
    """The generator of the strongest LEARNED transition for `action`.

    Returns None when no channel of `theta_a[action]` exceeds `min_norm`.

    `min_norm` CALIBRATION (measured, UHR-03 kill-run #1, 2026-09-23). The
    default was 1e-8, which sits BELOW the measurement's own noise floor: the
    store's log path (`henri_external_outcome_refactor_module._matrix_log_eig`)
    returns a norm of ~3.2e-06 for `delta_U = U U^dag` on float32, and EXACTLY
    0.0 only for an exact identity. A floor of 1e-8 therefore admits a pure
    identity's numerical residue as a "recorded transition" -- it cannot
    separate "no transition" from "a transition". Observed live: a static frame
    gave `target_theta_norm = 1.8106915149473934e-06`, ABOVE the old floor, so
    the gate would have compared against noise. 1e-5 is ~3x the measured floor
    and 5 orders below the smallest genuinely learned transition
    (`||H||=1e-03 -> 9.05e-02`). Raising it converts a spurious comparison into
    an honest UNAVAILABLE, which is `BLOCKED`, never a negative result.
    """
    if store is None or action is None or int(action) < 0:
        return None
    with torch.no_grad():
        if channel is None:
            channel = transition_channel(store, action, min_norm)
            if channel is None:
                return None
        th = store.theta_a[int(action)]
        if not 0 <= int(channel) < int(th.shape[0]):
            raise ValueError(
                "recorded_transition_generators: channel %r out of range for %d"
                % (channel, int(th.shape[0])))
        if float(th[int(channel)].norm()) <= min_norm:
            return None
        return [store.lie_element(int(action), gell_mann_basis)[int(channel)]]


def exteroceptive_residual_vs_recorded(
    state_roles: torch.Tensor,
    candidate_generators: Sequence[torch.Tensor],
    truth_generators: Sequence[torch.Tensor],
    gell_mann_basis: torch.Tensor,
    tau: float = TAU_BLUEPRINT,
) -> GateResult:
    """FORM B against a RECORDED transition. Reads the RELATIVE group element.

    Both operands are forward maps of the SAME real [K, 8] role vector, so the
    residual is
        cos -> (1/K) sum_k R_k^T A_c^T A_t R_k -> Tr(A_c^T A_t)/8
    i.e. |Tr(U_c^dag U_t)|^2, the group-theoretic relative element. It never
    reads a future observation: `truth_generators` come from a transition that
    has already been observed and stored.
    """
    dev_s = assert_single_family(state_roles, "state_roles")
    if not candidate_generators:
        raise ValueError("exteroceptive_residual_vs_recorded: candidate is empty")
    if not truth_generators:
        raise ValueError(UNAVAILABLE_NO_RECORDED_TRANSITION)
    A_c = ad_of(candidate_generators, gell_mann_basis)
    A_t = ad_of(truth_generators, gell_mann_basis)
    pred_c = predict_next(state_roles, A_c)
    pred_t = predict_next(state_roles, A_t)
    dp = delta(pred_c, pred_t)
    rge = relative_group_element(candidate_generators, truth_generators)
    band = sampling_band(state_roles.shape[0])
    return GateResult(
        delta_pred=dp,
        delta_state=delta(pred_c, state_roles),
        delta_identity=delta(state_roles, pred_t),
        hard_vetoed=bool(dp > tau),
        relative_group_element=rge,
        n_blocks=int(state_roles.shape[0]),
        block_norm_dev=dev_s,
        reason=None,
        magnitude_only_risk=bool(abs(dp - delta(pred_c, state_roles)) < band),
    )


def generators_from_displacement(disp: torch.Tensor, channel: int = 0) -> list:
    """The anti-Hermitian generator H with exp(H) == disp[channel] (a group element).

    Mirrors the production D31 path (`henri_external_outcome_refactor_module.
    _matrix_log_eig`): torch 2.12 has no `matrix_log`, and `disp` is unitary so
    the eigendecomposition form is exact.

    SHAPE CONTRACT (measured defect, UHR-03 kill-run #2 RFSS arm, 2026-09-23).
    `relative_displacement` returns the CHANNEL-BATCHED group element `[N,3,3]`.
    Returning its log verbatim produced an `[N,3,3]` "generator", while `ad_of`
    and `relative_group_element` require `[3,3]` -- the production convention is
    ONE channel (`store.lie_element(action, basis)[0]`). The mismatch surfaced
    only deep inside torch as
        `RuntimeError: trace: expected a matrix, but got tensor with dim 3`
    and was CPU-INVISIBLE, because the contract tests pass `[3,3]` operands on
    BOTH sides and so never exercise the batched path. This function now SELECTS
    one channel and ASSERTS `[3,3]` out; `ad_of` asserts `[3,3]` in. The fault now
    fails loudly at the boundary with a named message instead of inside linalg.
    """
    if disp.dim() == 2:
        single = disp
    elif disp.dim() == 3:
        if not 0 <= int(channel) < int(disp.shape[0]):
            raise ValueError(
                "generators_from_displacement: channel %r out of range for batch %d"
                % (channel, int(disp.shape[0])))
        single = disp[int(channel)]
    else:
        raise ValueError(
            "generators_from_displacement: expected [3,3] or [N,3,3], got %s"
            % (tuple(disp.shape),))
    if tuple(single.shape) != (3, 3):
        raise ValueError(
            "generators_from_displacement: selected operand is %s, expected [3,3]"
            % (tuple(single.shape),))
    evals, evecs = torch.linalg.eig(single.to(torch.complex64))
    log_single = (evecs @ torch.diag_embed(torch.log(evals))
                  @ evecs.conj().transpose(-2, -1))
    return [log_single]


def relative_displacement(u_to: torch.Tensor, u_from: torch.Tensor) -> torch.Tensor:
    """U_to @ U_from^dag -- the observed transition as a group element [N,3,3]."""
    return torch.einsum("nij,nkj->nik", u_to, u_from.conj())


__all__ = [
    "AXIOM_BLOCK_NORM_TOL",
    "DOMAIN_VIOLATION",
    "EdgeRecord",
    "FLAG_ENV",
    "GateResult",
    "N_BLOCKS_DEFAULT",
    "SELF_COMPARISON",
    "SOLIPSISM_VETO",
    "TAU_BLUEPRINT",
    "TAU_VETO",
    "UNAVAILABLE_NO_RECORDED_TRANSITION",
    "ad_of",
    "assert_single_family",
    "delta",
    "exteroceptive_gate",
    "exteroceptive_residual_vs_recorded",
    "flag_enabled",
    "forge_edge",
    "measured_bands",
    "normalize_roles",
    "predict_next",
    "recorded_transition_generators",
    "relative_group_element",
    "sampling_band",
    "transition_channel",
]
