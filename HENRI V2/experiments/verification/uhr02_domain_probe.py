"""UHR-02 DOMAIN PROBE — the load-bearing measurement for the blueprint.

USER'S MAXIM
    "Fix the comparison's domain, not the comparison's threshold."

WHY THIS FILE
    The HENRI-SPEC-2026-CAUSAL-REALITY-V1 blueprint places Zone A ingress on the
    complex unit hypersphere S^{D-1}, D=65,536, via

        Psi_in = (1/sqrt(M)) * sum_k (R_k  conv  F_k)

    That is a COMPLEX FLAT [D] wave -- the exact family whose cross-family
    comparison against the REAL [num_blocks, 8] axiom baseplate produced the
    recorded defect (delta ~ 0.998, 8/8 vetoed). So the blueprint decision is
    governed by the same maxim: is S^{D-1} a legitimate SECOND comparison domain,
    or must it be transport-only with one-way transduction into one domain?

ANALYTIC STRUCTURE UNDER TEST (derived, then measured)
    With per-block unit roles R_k in R^8 and B_k = n(A R_k) (A = Ad(U) in SO(8)),
    the gate's residual is

        delta = 0.5 * (1 - cos(B, R)),   cos(B, R) = (1/K) * sum_k R_k^T A R_k

    For i.i.d. isotropic R_k,  E[R^T A R] = Tr(A)/8,  and Tr(Ad U) = |Tr U|^2 - 1, so

        delta_iso = (9 - |Tr U|^2) / 16

    i.e. delta becomes a function of the OPTION'S MAGNITUDE alone and stops
    reading the REFERENCE'S CONTENT. The question is whether that collapse is
    (a) universal, or (b) a property of an i.i.d. ISOTROPIC reference only.

TESTS
    T1  analytic identity, with the K=8192 sampling noise stated (not a guess)
    T2  two options matched on |Tr U|, different content, ISOTROPIC reference
    T3  two options matched on |Tr U|, different content, STRUCTURED reference
        (block-correlated roles: R_k = n(c + eps_k)) -- this is the constructive
        case: a real grid residual boundary is NOT i.i.d. across blocks
    T4  per-block alignment dispersion (does content survive BEFORE averaging?)
    T5  tau calibration populations (compliant vs invalid) at the natural scale

READINGS
    T2 no separation + T3 separation  -> S^{D-1} is transport-only; the
        comparison domain must carry CONTENT-BEARING (structured) roles.
    T2 no separation + T3 no separation -> the channel is magnitude-only
        whatever the reference -> UHR-01's live value is FALSIFIED.
"""
from __future__ import annotations

import math
import pathlib
import sys

import torch

_HERE = pathlib.Path(__file__).resolve()
for _c in (_HERE.parents[1], _HERE.parents[2], _HERE.parents[3]):
    if (_c / "uhr_rfss.py").exists():
        sys.path.insert(0, str(_c))
        break

from uhr_rfss import (  # noqa: E402
    adjoint_matrix,
    block_norm_deviation,
    project_option_to_boundary_family as project,
)

TAU_BLUEPRINT = 0.35
K = 8192          # axiom block width used by the live loader
DIM = 8           # su(3) adjoint dimension
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


# --------------------------------------------------------------------------
# reference (baseplate) constructors
# --------------------------------------------------------------------------
def roles_isotropic(seed: int, k: int = K) -> torch.Tensor:
    """i.i.d. unit 8-vectors: the 'randn + normalize' seeder pattern."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(k, DIM, generator=g)
    return r / r.norm(dim=-1, keepdim=True)


def roles_structured(seed: int, coherence: float = 6.0, k: int = K) -> torch.Tensor:
    """Block-CORRELATED roles: R_k = n(c + eps_k).

    A real ARC grid residual boundary is not i.i.d. across blocks -- spatially
    contiguous patches share colour/structure, so the block ensemble carries a
    common component. `coherence` scales that shared component.
    """
    g = torch.Generator().manual_seed(seed)
    common = torch.randn(DIM, generator=g)
    eps = torch.randn(k, DIM, generator=g)
    r = coherence * common.unsqueeze(0) + eps
    return r / r.norm(dim=-1, keepdim=True)


# --------------------------------------------------------------------------
# option (candidate) constructors
# --------------------------------------------------------------------------
def generators(theta: torch.Tensor, k: int = 4) -> list:
    """D_a = i * sum_k theta[k] * lambda_k, exactly as lie_element emits it."""
    return [1j * torch.einsum("a,aij->ij", theta.to(BASIS.dtype), BASIS)
            for _ in range(k)]


def theta_of(seed: int, scale: float, n: int = DIM) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, generator=g) * scale


def U_of(gs: list) -> torch.Tensor:
    U = torch.eye(3, dtype=torch.complex64)
    for g in gs:
        U = U @ torch.matrix_exp(g)
    return U


def abs_trace(U: torch.Tensor) -> float:
    """|Tr U| as a float, reporting the shape if it is not 3x3.

    The earlier probe crashed with 'only one element tensors can be converted to
    Python scalars' at the call site rather than here, so this helper names the
    offending shape instead of hiding it.
    """
    if U.shape != (3, 3):
        raise ValueError("U_of returned shape %s, expected (3, 3)" % (tuple(U.shape),))
    return float(torch.abs(U.trace().reshape(())))


def A_of(gs: list) -> torch.Tensor:
    A = torch.eye(DIM)
    for g in gs:
        A = A @ adjoint_matrix(g, BASIS)
    return A


def delta(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    """The gate's own residual: 0.5 * (1 - cos), on the flattened real pair."""
    x = candidate.flatten().to(torch.float32)
    y = reference.flatten().to(torch.float32)
    cos = float((x * y).sum()) / (float(x.norm()) * float(y.norm()))
    return 0.5 * (1.0 - cos)


def match_tr(target_abs_tr: float, seed: int) -> list:
    """Scale a DIFFERENT direction until |Tr U| matches the target."""
    lo, hi = 0.0, 12.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        s = abs_trace(U_of(generators(theta_of(seed, mid))))
        if s < target_abs_tr:
            lo = mid
        else:
            hi = mid
    return generators(theta_of(seed, 0.5 * (lo + hi)))


def main() -> int:
    print("torch", torch.__version__, "| K =", K, "| tau_blueprint =", TAU_BLUEPRINT)
    iso = roles_isotropic(1)
    print("block_norm_dev(isotropic ref) = %.3e" % block_norm_deviation(iso))

    # ---------------- T1: the analytic identity, with noise context ----------
    print("\n=== T1  delta_iso = (9 - |Tr U|^2)/16   (identity + sampling noise) ===")
    print("  %-6s %-14s %-14s %-12s %-12s" % ("seed", "predicted", "measured", "abs_err", "noise_band"))
    worst = 0.0
    for sd in (10, 20, 30, 40, 50, 60):
        gs = generators(theta_of(sd, 0.30))
        trU = abs_trace(U_of(gs))
        pred = (9.0 - trU ** 2) / 16.0
        meas = delta(project(gs, BASIS, iso), iso)
        # sigma of (1/2)<R,AR> over K i.i.d. isotropic blocks:
        # per-block var of R^T A R on S^7 is ~ 2/(d+2) = 2/9; averaged over K.
        band = 0.5 * math.sqrt(2.0 / (DIM + 2)) / math.sqrt(K)
        worst = max(worst, abs(pred - meas))
        print("  %-6d %-14.9f %-14.9f %-12.3e %-12.3e" % (sd, pred, meas, abs(pred - meas), band))
    pred_band = 0.5 * math.sqrt(2.0 / (DIM + 2)) / math.sqrt(K)
    print("  measured worst_err = %.3e   analytic 1-sigma band = %.3e" % (worst, pred_band))
    print("  -> identity %s (err is %.2f x the sampling band)"
          % ("CONFIRMED WITHIN SAMPLING NOISE" if worst < 5 * pred_band else "REFUTED",
             worst / pred_band))

    # ---------------- T2/T3: equal |Tr U|, different content ----------------
    ref_g = generators(theta_of(10, 0.30))
    target = abs_trace(U_of(ref_g))
    alt_g = match_tr(target, 9000)
    trA, trB = abs_trace(U_of(ref_g)), abs_trace(U_of(alt_g))
    print("\n=== T2/T3  equal |Tr U|, DIFFERENT option content ===")
    print("  option A: |Tr U| = %.9f" % trA)
    print("  option B: |Tr U| = %.9f   (matched, distinct direction)" % trB)
    print("  ||Ad(A) - Ad(B)||_F = %.4f   (genuinely different rotations)"
          % float((A_of(ref_g) - A_of(alt_g)).norm()))

    for name, ref in (("ISOTROPIC i.i.d. roles", iso),
                      ("STRUCTURED (coherence=2)", roles_structured(1, 2.0)),
                      ("STRUCTURED (coherence=6)", roles_structured(1, 6.0)),
                      ("STRUCTURED (coherence=20)", roles_structured(1, 20.0))):
        dA = delta(project(ref_g, BASIS, ref), ref)
        dB = delta(project(alt_g, BASIS, ref), ref)
        sep = abs(dA - dB)
        verdict = "SEPARATES" if sep > 4 * pred_band else "INDISTINGUISHABLE"
        print("  %-26s dA=%.9f dB=%.9f  sep=%.3e  %s"
              % (name, dA, dB, sep, verdict))

    # ---------------- T4: per-block dispersion (content before averaging) ---
    print("\n=== T4  per-block alignment dispersion (content survives pre-average?) ===")
    for name, ref in (("ISOTROPIC", iso), ("STRUCTURED c=6", roles_structured(1, 6.0))):
        pa = (project(ref_g, BASIS, ref).flatten(1).to(torch.float32)
              * ref.flatten(1).to(torch.float32)).sum(-1)
        pb = (project(alt_g, BASIS, ref).flatten(1).to(torch.float32)
              * ref.flatten(1).to(torch.float32)).sum(-1)
        gap = (pa - pb)
        print("  %-14s per-block mean |gap| = %.6f   std = %.6f   max = %.6f"
              % (name, float(gap.abs().mean()), float(gap.std()), float(gap.abs().max())))
        print("                 frac(|gap| > 1e-3) = %.4f"
              % float((gap.abs() > 1e-3).to(torch.float32).mean()))

    # ---------------- T5: tau calibration populations -----------------------
    print("\n=== T5  tau calibration: compliant vs invalid populations ===")
    comp, inval = [], []
    for sd in (101, 102, 103, 104, 105, 106, 107, 108):
        gs = generators(theta_of(sd, 0.10))
        comp.append(delta(project(gs, BASIS, iso), iso))
        perm = torch.randperm(K, generator=torch.Generator().manual_seed(sd))
        inval.append(delta(project(gs, BASIS, iso, role_permutation=perm), iso))
    print("  compliant (theta=0.10) n=%d  min=%.6f max=%.6f mean=%.6f"
          % (len(comp), min(comp), max(comp), sum(comp) / len(comp)))
    print("  invalid   (permuted)   n=%d  min=%.6f max=%.6f mean=%.6f"
          % (len(inval), min(inval), max(inval), sum(inval) / len(inval)))
    lo, hi = max(comp), min(inval)
    print("  separation gap = %.6f  -> tau must lie in (%.6f, %.6f); blueprint 0.35 %s"
          % (hi - lo, lo, hi,
             "LIES IN BAND" if lo < TAU_BLUEPRINT < hi else "IS OUTSIDE THE MEASURED BAND"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
