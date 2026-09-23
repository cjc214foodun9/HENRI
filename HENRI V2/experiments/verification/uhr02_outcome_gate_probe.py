"""UHR-02 OUTCOME-GATE PROBE — what "fix the comparison's domain" means, measurably.

USER'S MAXIM
    "Fix the comparison's domain, not the comparison's threshold."

THE TWO DOMAINS (this file measures the difference)
    Let R in R^{K x 8} be the current state roles, and let the ENVIRONMENT yield
    an observed next state

        R_obs = n( Ad(U_true) R + noise )            (exteroceptive transition)

    A candidate option U_c predicts  R_c = n( Ad(U_c) R ).

    FORMULATION A -- compare the candidate against the CURRENT STATE:
        Delta_A(U_c) = 0.5 * (1 - cos(R_c, R))

      Under an isotropic baseplate this collapses onto the OPTION'S MAGNITUDE:
          cos(R_c, R) = (1/K) sum_k R_k^T A R_k  ->  Tr(A)/8,  Tr(Ad U) = |Tr U|^2 - 1
          => Delta_A = (9 - |Tr U_c|^2)/16          [MEASURED to 1.19x sampling noise]
      Two DIFFERENT options with equal |Tr U| are therefore indistinguishable.
      This is the domain the live loop currently uses, and it is why the
      UHR-01 run reported delta_axiom = 0.0 with an empty store: with theta = 0,
      U = I, Ad(I) = I, R_c == R exactly -- a self-comparison.

    FORMULATION B -- compare the candidate against the OBSERVED TRANSITION:
        Delta_B(U_c) = 0.5 * (1 - cos(R_c, R_obs))

      Then  cos ~ R^T A_c^T A_t R / K  ->  Tr(A_c^T A_t)/8
                 = ( |Tr(U_c^dag U_t)|^2 - 1 ) / 8
      i.e. it reads the RELATIVE group element -- how well the candidate
      predicts the transition that ACTUALLY happened. This is the blueprint's
      own prescription (spec §3.2: exteroceptive verification, Delta_S_ext,
      SOLIPSISM_VETO) and it is a DOMAIN change, not a threshold change.

PRE-REGISTERED PREDICTIONS (stated before the numbers are read)
    P1  Formulation A, isotropic baseplate: c_true and c_other (equal |Tr U|)
        are indistinguishable within the sampling band  -> magnitude-only.
    P2  Formulation A, STRUCTURED baseplate (block-correlated roles): the same
        two options SEPARATE -> content survives only with a content-bearing
        reference.
    P3  Formulation B: c_true separates from c_other under BOTH baseplates,
        because it is grounded in the observed transition, not the state.
    KILL  If P3 fails, the channel cannot gate on outcome at all and the
        blueprint's Loop 1 has no usable verification signal.

Exit 0 always; the printed lines are the record.
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

from uhr_rfss import adjoint_matrix, project_option_to_boundary_family as project  # noqa: E402

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

# sampling band of the isotropic identity, measured in uhr02_domain_probe.py
BAND = 0.5 * math.sqrt(2.0 / (8 + 2)) / math.sqrt(K)


def roles_iso(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(K, 8, generator=g)
    return r / r.norm(dim=-1, keepdim=True)


def roles_struct(seed: int, coh: float = 6.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = coh * torch.randn(8, generator=g).unsqueeze(0) + torch.randn(K, 8, generator=g)
    return r / r.norm(dim=-1, keepdim=True)


def gens(seed: int, sc: float, k: int = 4) -> list:
    g = torch.Generator().manual_seed(seed)
    th = torch.randn(8, generator=g) * sc
    return [1j * torch.einsum("a,aij->ij", th.to(BASIS.dtype), BASIS) for _ in range(k)]


def A_of(gs: list) -> torch.Tensor:
    A = torch.eye(8)
    for g in gs:
        A = A @ adjoint_matrix(g, BASIS)
    return A


def abs_tr(gs: list) -> float:
    U = torch.eye(3, dtype=torch.complex64)
    for g in gs:
        U = U @ torch.matrix_exp(g)
    return float(torch.abs(U.trace().reshape(())))


def wave(R: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """n(A R) blockwise: the candidate / transition wave in the real family."""
    y = torch.einsum("ij,kj->ki", A.to(R.dtype), R)
    return y / y.norm(dim=-1, keepdim=True)


def delta(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.flatten().to(torch.float32)
    y = b.flatten().to(torch.float32)
    return 0.5 * (1.0 - float((x * y).sum()) / (float(x.norm()) * float(y.norm())))


def match_scale(target: float, seed: int) -> list:
    lo, hi = 0.0, 12.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if abs_tr(gens(seed, mid)) < target:
            lo = mid
        else:
            hi = mid
    return gens(seed, 0.5 * (lo + hi))


def main() -> int:
    print("torch", torch.__version__, "| K =", K, "| sampling band = %.3e" % BAND)
    print("PRE-REGISTERED: P1 no-sep(A,iso)  P2 sep(A,struct)  P3 sep(B,both)")
    print()

    g_true = gens(4242, 0.30)
    A_true = A_of(g_true)
    tgt = abs_tr(g_true)
    g_other = match_scale(tgt, 9000)          # equal |Tr U|, different content
    A_other = A_of(g_other)
    print("true   option: |Tr U| = %.9f" % abs_tr(g_true))
    print("other  option: |Tr U| = %.9f   (matched magnitude, distinct direction)" % abs_tr(g_other))
    print("||Ad(true) - Ad(other)||_F = %.4f" % float((A_true - A_other).norm()))
    print()

    for ref_name, ref in (("ISOTROPIC", roles_iso(1)), ("STRUCTURED c=6", roles_struct(1, 6.0))):
        # exteroceptive observation: the environment performs the TRUE transition
        gn = torch.Generator().manual_seed(77)
        R_obs = wave(ref, A_true) + 0.05 * torch.randn(K, 8, generator=gn)
        R_obs = R_obs / R_obs.norm(dim=-1, keepdim=True)

        dA_true = delta(wave(ref, A_true), ref)       # A: candidate vs CURRENT state
        dA_othr = delta(wave(ref, A_other), ref)
        dB_true = delta(wave(ref, A_true), R_obs)     # B: candidate vs OBSERVED transition
        dB_othr = delta(wave(ref, A_other), R_obs)
        # and the do(a)=identity candidate: no action performed
        dB_none = delta(ref, R_obs)

        print("=== baseplate %s ===" % ref_name)
        print("  FORM A (vs current state)  true=%.9f  other=%.9f  sep=%.3e  band=%.3e  %s"
              % (dA_true, dA_othr, abs(dA_true - dA_othr), BAND,
                 "NO-SEP (magnitude-only)" if abs(dA_true - dA_othr) < 4 * BAND else "SEP"))
        print("  FORM B (vs observed trans) true=%.9f  other=%.9f  sep=%.3e  band=%.3e  %s"
              % (dB_true, dB_othr, abs(dB_true - dB_othr), BAND,
                 "NO-SEP" if abs(dB_true - dB_othr) < 4 * BAND else "SEP"))
        print("  noise floor: identity-candidate (do nothing) delta = %.9f" % dB_none)
        print("  ordering true < other for FORM B: %s"
              % (dB_true < dB_othr))
        print()

        if ref_name.startswith("ISOTROPIC"):
            p1_ok = abs(dA_true - dA_othr) < 4 * BAND
            p3_iso = abs(dB_true - dB_othr) > 4 * BAND
        else:
            p2_ok = abs(dA_true - dA_othr) > 4 * BAND
            p3_str = abs(dB_true - dB_othr) > 4 * BAND

    print("VERDICT")
    print("  P1 no-separation under isotropic, FORM A : %s" % p1_ok)
    print("  P2 separation under structured,  FORM A : %s" % p2_ok)
    print("  P3 separation under BOTH,        FORM B : iso=%s struct=%s" % (p3_iso, p3_str))
    print()
    print("INTERPRETATION")
    print("  FORM A compares the candidate against the state it started from. Under an")
    print("  isotropic baseplate that reduces to (9 - |Tr U|^2)/16 -- option MAGNITUDE --")
    print("  so two different options with equal |Tr U| are indistinguishable. This is the")
    print("  live loop's current domain, and with theta = 0 it is exactly the self-comparison")
    print("  that reported delta_axiom = 0.0.")
    print("  FORM B compares the candidate against the transition the environment ACTUALLY")
    print("  made. That reads the relative group element |Tr(U_c^dag U_t)| -- how well the")
    print("  candidate PREDICTS what happened -- which is a DOMAIN change, not a threshold")
    print("  change. It is the blueprint's own exteroceptive comparator (spec section 3.2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
