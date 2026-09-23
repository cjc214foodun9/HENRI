"""UHR-01 degeneracy probe — the C2 falsifier (OBSERVED evidence).

WHY THIS FILE EXISTS
    The remote paired A/B (instance 52189427) reported

        BASELINE  delta_axiom = 0.997175 .. 0.999582   hard_vetoed True  8/8
        RFSS      delta_axiom = 0.0 exactly (8/8)      hard_vetoed False 8/8

    A constant ZERO is a degeneracy signature, not a pass. This probe names the
    cause with two independent controls, and is the artifact behind §5 of
    experiments/verification/uhr01_verdict.md.

CHAIN UNDER TEST
    live telemetry showed preference_store_size = 0 and gain_single = 0.0 at every
    step. Therefore:

        theta_a = 0
          -> lie_element: 1j * einsum(theta, gell_mann)      = 0 matrix
          -> construct_macro_option: matrix_exp(0)           = I
          -> adjoint action:  Ad(I)                          = I
          -> projection returns the axiom itself             = SELF-COMPARISON
          -> delta(candidate, axiom)                         = 0.0

CONTROLS
    1. ZERO generator        -> predicts delta ~ 0.0   (the live case)
    2. NON-ZERO generator    -> predicts delta > 0     (a populated store)
    3. AXIOM-CONTENT sweep   -> delta must vary with the axiom, not only the option
    4. DEAD-INPUT negative   -> a permuted candidate must FAIL the gate (must be > tau)

    Control 4 is mandatory per the vacuous-invariant defect class: every invariant
    gate ships a negative control that MUST fail it.

RUN
    python experiments/verification/uhr01_degeneracy_probe.py
Exit 0 always; the printed lines are the record.
"""
from __future__ import annotations

import math
import pathlib
import sys

import torch

_HERE = pathlib.Path(__file__).resolve()
for _cand in (_HERE.parents[1], _HERE.parents[2]):
    if (_cand / "uhr_rfss.py").exists():
        sys.path.insert(0, str(_cand))
        break

from uhr_rfss import (  # noqa: E402
    block_norm_deviation as block_norm_dev,
    project_option_to_boundary_family as project,
)

TAU = 0.35  # hard-veto threshold, matching production_arc_run.py's epsilon_hard

_s3 = math.sqrt(3.0)
GELL_MANN = [
    [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
    [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
    [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
    [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
    [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
    [[1 / _s3, 0, 0], [0, 1 / _s3, 0], [0, 0, -2 / _s3]],
]
BASIS = torch.tensor(GELL_MANN, dtype=torch.complex64)
N_BLOCKS = 8192


def axiom_roles(seed: int) -> torch.Tensor:
    """Per-block unit 8-vectors: the shape the Zone C loader returns."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(N_BLOCKS, 8, generator=g)
    return r / r.norm(dim=-1, keepdim=True)


def option_generators(theta: torch.Tensor) -> list:
    """D_a = i * sum_k theta[k] * lambda_k, exactly as lie_element emits."""
    return [1j * torch.einsum("a,aij->ij", theta.to(BASIS.dtype), BASIS)
            for _ in range(4)]


def delta(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    """The gate's similarity residual: 1 - cos, on the flattened real pair."""
    x = candidate.flatten().to(torch.float32)
    y = reference.flatten().to(torch.float32)
    cos = float((x * y).sum()) / (float(x.norm()) * float(y.norm()))
    return 0.5 * (1.0 - cos)


def main() -> int:
    roles = axiom_roles(1)
    print("torch", torch.__version__)
    print("N_BLOCKS", N_BLOCKS, "tau", TAU)
    print()

    # ---- control 1: ZERO generator (the live configuration) ----------------
    zero_theta = torch.zeros(8, dtype=BASIS.dtype)
    gens0 = option_generators(zero_theta)
    p0 = project(gens0, BASIS, roles)
    d0 = delta(p0, roles)
    print("[control 1] zero generator (live case: preference_store_size = 0)")
    print("  ||gen||                 = %.6e" % float(gens0[0].norm()))
    print("  block_norm_dev(proj)    = %.6e" % block_norm_dev(p0))
    print("  delta(proj, axiom)      = %.10f   (self-comparison)" % d0)
    print("  hard_vetoed             = %s" % (d0 > TAU))
    print()

    # ---- control 2: NON-ZERO generator (a populated store) -----------------
    g = torch.Generator().manual_seed(1000)
    theta = torch.randn(8, generator=g).to(BASIS.dtype) * 0.10
    gens1 = option_generators(theta)
    p1 = project(gens1, BASIS, roles)
    d1 = delta(p1, roles)
    print("[control 2] non-zero generator (theta ~ 0.10: populated store)")
    print("  ||gen||                 = %.6e" % float(gens1[0].norm()))
    print("  block_norm_dev(proj)    = %.6e" % block_norm_dev(p1))
    print("  delta(proj, axiom)      = %.10f" % d1)
    print("  ||proj - roles||        = %.6e  (projection MOVED the candidate)"
          % float((p1 - roles).norm()))
    print("  hard_vetoed             = %s" % (d1 > TAU))
    print()

    # ---- control 3: axiom-content sweep -----------------------------------
    print("[control 3] axiom-content sweep (non-zero generator, 4 different axioms)")
    sweep = []
    for seed in (11, 12, 13, 14):
        ax = axiom_roles(seed)
        sweep.append(delta(project(gens1, BASIS, ax), ax))
    print("  delta values            = %s" % [round(x, 9) for x in sweep])
    print("  varies with axiom       = %s" % (len({round(x, 12) for x in sweep}) > 1))
    print()

    # ---- control 4: DEAD-INPUT negative control (MUST fail the gate) --------
    perm = torch.randperm(N_BLOCKS, generator=torch.Generator().manual_seed(3))
    dp = delta(project(gens1, BASIS, roles, role_permutation=perm), roles)
    print("[control 4] dead-input negative control (permuted candidate) - MUST exceed tau")
    print("  delta(permuted, axiom)  = %.10f" % dp)
    print("  gate FAILS as required  = %s" % (dp > TAU))
    print()

    # ---- option dependence -------------------------------------------------
    gens2 = option_generators(
        torch.randn(8, generator=torch.Generator().manual_seed(2000)).to(BASIS.dtype) * 0.10)
    print("[control 5] option dependence")
    print("  ||P(G1) - P(G2)||       = %.6e" % float((p1 - project(gens2, BASIS, roles)).norm()))
    print()

    # ---- verdict -----------------------------------------------------------
    # NOTE: the zero-generator case is a SELF-COMPARISON only to float32 accuracy.
    # Measured delta = -2.28e-07 (cos = 1.0000004): the projection returns the
    # axiom itself up to float32 noise, so the tolerance is 1e-5, not 1e-9.
    self_comparison = abs(d0) < 1e-5
    nz_discriminates = d1 > 1e-6
    content_reads = len({round(x, 12) for x in sweep}) > 1
    neg_control_ok = dp > TAU
    print("VERDICT INPUTS")
    print("  empty_store_is_self_comparison = %s   (|delta| = %.2e, float32 noise)"
          % (self_comparison, abs(d0)))
    print("  populated_store_discriminates  = %s" % nz_discriminates)
    print("  channel_reads_axiom_content    = %s" % content_reads)
    print("  negative_control_fails_gate    = %s" % neg_control_ok)
    print()
    print("[calibration note] a POPULATED store at theta scale 0.10 gives")
    print("  delta = %.6f against tau = %.2f -- a margin of only %.6f."
          % (d1, TAU, abs(d1 - TAU)))
    print("  tau is therefore NOT well calibrated for the projected family: a modest")
    print("  option already lands at the threshold. Re-calibrate tau from a measured")
    print("  compliant/invalid population before enabling this gate in production.")
    print()
    print("INTERPRETATION")
    print("  The live RFSS arm reported delta_axiom = 0.0 because preference_store_size")
    print("  was 0, so theta = 0 -> U = I -> Ad(U) = I, and the projection returns the")
    print("  axiom itself. The gate therefore flipped from ALWAYS-VETO (baseline 8/8)")
    print("  to NEVER-VETO (RFSS 0/8). Both extremes are non-discriminative.")
    print("  UHR-01 made the veto COMPUTABLE; it did not establish live-loop")
    print("  discrimination. Discriminating it requires a POPULATED outcome store.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
