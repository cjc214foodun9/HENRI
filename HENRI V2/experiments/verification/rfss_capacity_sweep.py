#!/usr/bin/env python3
"""RFSS capacity: MEASURE which crosstalk law holds. Do not assume either.

THE DISPUTE
    The attached spec (HENRI-SPEC-2026-THERMO-VLA-V1, sec 2.3) asserts

        E[||xi_crosstalk||^2] = (M - 1) / D          (spec's crosstalk law)

    and therefore "SNR >= 256 for M <= 256" at D = 65,536, with Gate 2 registered
    as "exact cosine >= 0.9200 at M = 32" (sec 6).

    A competing reading of the same algebra says the crosstalk terms are
    NORM-PRESERVING. Unbinding member m leaves a rotated copy:

        Psi (*) R_k^dagger = (1/sqrt(M)) [ F_k + sum_{m!=k} F_m (*) U_mk ],
        U_mk = R_m (*) R_k^dagger,  |U_mk| unit-modulus in frequency.

    Because | ||F (*) U|| - ||F|| | is tiny, each interfering term keeps the norm
    of F_m. Summing M-1 quasi-orthogonal such terms gives

        ||xi|| ~ sqrt(M - 1)      (NOT sqrt((M-1)/D))

    which is D-INDEPENDENT and suggests the spec's formula is wrong by a factor D.

    BUT the two claims are not actually in conflict, because they measure
    different things, and conflating them is the real error:

      * ||xi|| / ||signal|| ~ sqrt(M-1)  -> the RAW cosine to the correct filler
        falls like 1/sqrt(M). This is the number the spec's Gate 2 names.
      * <xi, F_j> for a codebook member j is only ~ sqrt(M-1)/sqrt(D), because xi
        lives in D dimensions and aligns with no single member. This is the
        margin codebook cleanup (argmax) uses, and it stays large.

    So exact recovery can remain perfect at M where the raw cosine is already
    tiny. Gate 2 is testable ONLY if it says which of the two it means.

WHAT THIS MODULE MEASURES  (D = 65536, float64 where it matters)
    raw_abs_cos           |<z, F_k>| / (||z|| ||F_k||)   vs prediction 1/sqrt(M)
    cleanup_exact_rate    fraction where argmax over filler codebook finds k
    margin_to_best_other  sims[k] - max_{j!=k} sims[j]
    nonmember_abs_cos     |<z, F_j>| for a filler that is NOT in the bundle
    crosstalk_norm        ||xi|| measured, vs spec sqrt((M-1)/D) and sqrt(M-1)

HONEST LIMITS
    Roles and fillers here are independent uniform phasors: the BEST case for the
    algebra. Real codebooks may be correlated, so any capacity limit measured here
    is an upper bound on real capacity. Gate 2's true number at M=32 is whatever
    cleanup_exact_rate says, not a cosine.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from typing import Dict, List

import torch

DEFAULT_DIM = 65536


def _receipt_path(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("HENRI_RECEIPT_DIR")
    if env:
        return os.path.join(env, "rfss_capacity_observed.json")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "rfss_capacity_observed.json")


def make_phasors(count: int, dim: int, gen: torch.Generator) -> torch.Tensor:
    """Unit-modulus phasors, L2-normalized so ||p||_2 = 1.

    Complex64 only. An earlier draft accepted a `dtype` argument and a caller
    passed the already-complex dtype of another tensor, which made `torch.complex`
    receive complex inputs. That class of bug is excluded by not parameterizing it.
    """
    phases = torch.rand(count, dim, generator=gen, dtype=torch.float32) * (2.0 * math.pi)
    p = torch.complex(torch.cos(phases), torch.sin(phases))
    return p / p.norm(dim=-1, keepdim=True)


def circular_bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft(torch.fft.fft(a, dim=-1) * torch.fft.fft(b, dim=-1), dim=-1)


def circular_correlate(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a (*) conj(b). b (*) conj(b) is the circular delta."""
    return torch.fft.ifft(torch.fft.fft(a, dim=-1)
                          * torch.conj(torch.fft.fft(b, dim=-1)), dim=-1)


def abs_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    num = torch.abs((torch.conj(a) * b).sum()).item()
    den = (a.norm() * b.norm()).item()
    return num / den if den > 0 else 0.0


def build_bundle(roles: torch.Tensor, fillers: torch.Tensor, m: int) -> torch.Tensor:
    bound = circular_bind(roles[:m], fillers[:m])              # [M, D]
    psi = bound.sum(dim=0) / math.sqrt(m)
    return psi / psi.norm()


def measure_depth(roles: torch.Tensor, fillers: torch.Tensor, m: int) -> Dict:
    psi = build_bundle(roles, fillers, m)
    codebook = fillers[:m]

    raw_cos: List[float] = []
    recovered = 0
    margins: List[float] = []
    crosstalk_norms: List[float] = []

    for k in range(m):
        z = circular_correlate(psi, roles[k])
        z = z / z.norm()
        raw_cos.append(abs_cos(z, fillers[k]))

        sims = torch.abs(torch.matmul(torch.conj(z), codebook.t()))   # [M]
        best = int(torch.argmax(sims).item())
        if best == k:
            recovered += 1
        mask = torch.ones(m, dtype=torch.bool); mask[k] = False
        margins.append(float(sims[k].item() - sims[mask].max().item()))

        # NOTE: this column is DEGENERATE and is retained only as a diagnostic.
        # z is unit-norm and `sig` is its projection onto fillers[k], so
        # ||z - sig|| == sqrt(1 - cos^2) BY CONSTRUCTION. It therefore cannot
        # discriminate the spec's (M-1)/D crosstalk law from the competing
        # sqrt(M-1) law; an earlier draft of this file reported a "winning law"
        # from this column, which was an artifact of the degeneracy, not a
        # measurement. The valid discriminators in this probe are raw_abs_cos
        # (falls as 1/sqrt(M)) and nonmember_abs_cos (should stay near 0).
        sig = (torch.conj(fillers[k]) * z).sum() * fillers[k]
        crosstalk_norms.append(float((z - sig).norm().item()))

    # non-member rejection: a filler absent from the bundle
    nonmember_cos = abs_cos(
        (lambda t: t / t.norm())(circular_correlate(psi, roles[0])),
        make_phasors(1, roles.shape[1], torch.Generator().manual_seed(999_001))[0])

    return {
        "M": m,
        "raw_abs_cos_mean": sum(raw_cos) / len(raw_cos),
        "prediction_1_over_sqrtM": 1.0 / math.sqrt(m),
        "cleanup_exact_recovery_rate": recovered / m,
        "margin_to_best_other_mean": sum(margins) / len(margins),
        "crosstalk_norm_mean": sum(crosstalk_norms) / len(crosstalk_norms),
        "spec_crosstalk_sqrt((M-1)/D)": math.sqrt((m - 1) / roles.shape[1]),
        "competing_crosstalk_sqrt(M-1)": math.sqrt(m - 1),
        "nonmember_abs_cos": nonmember_cos,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=DEFAULT_DIM)
    ap.add_argument("--depths", type=str, default="4,8,16,32,64,109,128,180,256")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    gen = torch.Generator().manual_seed(args.seed)
    roles = make_phasors(max(depths), args.dim, gen)
    fillers = make_phasors(max(depths), args.dim, gen)

    # ---- INSTRUMENT SELF-CHECK, in float64 so the check itself is trustworthy
    r64 = roles[0].to(torch.complex128)
    f64 = fillers[0].to(torch.complex128)
    bound_norm = circular_bind(r64, f64).norm().item()
    bind_err = abs(bound_norm - r64.norm().item())
    spike = circular_correlate(r64, r64)
    offpeak = float(spike.abs().max() / max(spike.abs().mean().item(), 1e-30))

    print("=" * 84)
    print("INSTRUMENT SELF-CHECK (float64; run before trusting any capacity number)")
    # The binding deviation is NOT zero and cannot be gated at a fixed 1e-10.
    # Measured law: dev_std * sqrt(D) = 0.50, constant from D=1024 to D=65536
    # (experiments/verification/binding_norm_algebra.py). At D=65536 that is
    # sqrt(3/(4D)) ~ 3.4e-3, so a one-sample deviation of ~2e-3 is EXPECTED, not a
    # fault. An earlier draft gated on bind_err < 1e-10 and reported
    # instrument_ok=False for a healthy instrument. The tolerance below is derived
    # from the measured scaling, not chosen to make the run pass.
    tol = 6.0 / math.sqrt(args.dim)
    print(f"  | ||R (*) F|| - ||R|| |   = {bind_err:.3e}   "
          f"(tolerance 6/sqrt(D) = {tol:.3e})")
    print(f"  R (*) conj(R) peak/mean   = {offpeak:.4g}   (expect large ~ delta)")
    instrument_ok = bind_err < tol and offpeak > 10.0
    print(f"  instrument_ok             = {instrument_ok}")
    if not instrument_ok:
        print("  !! instrument failed; capacity numbers below are NOT trustworthy")
        print("  !! check the delta identity (peak/mean) and the D-scaling first")
    print("=" * 84)

    rows: List[Dict] = []
    for m in depths:
        t0 = time.time()
        row = measure_depth(roles, fillers, m)
        row["seconds"] = round(time.time() - t0, 3)
        rows.append(row)
        print(f"  M={m:4d} raw={row['raw_abs_cos_mean']:.4f} "
              f"(1/sqrtM={row['prediction_1_over_sqrtM']:.4f}) "
              f"exact={row['cleanup_exact_recovery_rate']:.3f} "
              f"margin={row['margin_to_best_other_mean']:.2e} "
              f"|xi|={row['crosstalk_norm_mean']:.2f} "
              f"spec={row['spec_crosstalk_sqrt((M-1)/D)']:.4f} "
              f"sqrt(M-1)={row['competing_crosstalk_sqrt(M-1)']:.2f} "
              f"nonmem={row['nonmember_abs_cos']:.4f}")

    at32 = next((r for r in rows if r["M"] == 32), None)
    raw_ok = [r["M"] for r in rows if r["raw_abs_cos_mean"] >= 0.92]
    exact_ok = [r["M"] for r in rows if r["cleanup_exact_recovery_rate"] >= 1.0]
    worst_rel_1_over_sqrtM = max(
        abs(r["raw_abs_cos_mean"] - r["prediction_1_over_sqrtM"])
        / r["prediction_1_over_sqrtM"] for r in rows)
    # which crosstalk law is closer, measured vs each prediction
    spec_err = max(abs(r["crosstalk_norm_mean"] - r["spec_crosstalk_sqrt((M-1)/D)"])
                   / r["competing_crosstalk_sqrt(M-1)"] for r in rows)
    comp_err = max(abs(r["crosstalk_norm_mean"] - r["competing_crosstalk_sqrt(M-1)"])
                   / r["competing_crosstalk_sqrt(M-1)"] for r in rows)

    out = {
        "module": "rfss_capacity_sweep",
        "evidence_class": "OBSERVED",
        "dim": args.dim, "seed": args.seed,
        "platform": platform.platform(), "torch": torch.__version__,
        "instrument_selfcheck": {"binding_norm_err_float64": bind_err,
                                 "corr_peak_over_mean": offpeak,
                                 "instrument_ok": instrument_ok},
        "rows": rows,
        "verdicts": {
            "spec_gate2_raw_cos_at_M32": at32["raw_abs_cos_mean"] if at32 else None,
            "spec_gate2_raw_0.92_reachable": bool(at32 and at32["raw_abs_cos_mean"] >= 0.92),
            "cleanup_exact_at_M32": at32["cleanup_exact_recovery_rate"] if at32 else None,
            "max_M_with_exact_cleanup_recovery": max(exact_ok) if exact_ok else None,
            "max_M_with_raw_cos_ge_0.92": max(raw_ok) if raw_ok else None,
            "worst_rel_err_vs_1_over_sqrtM": worst_rel_1_over_sqrtM,
            "crosstalk_law_worst_rel_err_vs_SPEC_(M-1)/D": spec_err,
            "crosstalk_law_worst_rel_err_vs_competing_sqrt(M-1)": comp_err,
            "crosstalk_law_verdict": (
                "NOT_SETTLED_BY_THIS_PROBE. The ||xi|| column is degenerate: it "
                "equals sqrt(1-cos^2) by construction because z is unit-norm and "
                "the subtracted signal is its own projection. An earlier draft "
                "reported a winning law from it; that was an artifact. The valid "
                "evidence here is raw_abs_cos ~ 1/sqrt(M) and nonmember_abs_cos "
                "staying near 0. To settle the crosstalk law independently, "
                "correlate an unbound wave against a member filler ABSENT from the "
                "bundle, and against a cross-correlated role/filler pair."),
        },
        "interpretation": (
            "Raw unbinding cosine follows 1/sqrt(M) and does not depend on D. "
            "Codebook cleanup (argmax) still recovers exactly far beyond that, "
            "because the crosstalk aligns with no single codebook member. Spec "
            "Gate 2 must therefore measure EXACT RECOVERY, not a raw cosine."
        ),
    }

    path = _receipt_path(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print()
    print("VERDICTS")
    v = out["verdicts"]
    print(f"  raw cos at M=32                 : {v['spec_gate2_raw_cos_at_M32']:.4f}")
    print(f"  spec Gate 2 raw (>=0.92) at M=32: "
          f"{'REACHABLE' if v['spec_gate2_raw_0.92_reachable'] else 'UNREACHABLE'}")
    print(f"  exact recovery at M=32          : {v['cleanup_exact_at_M32']:.3f}")
    print(f"  max M with exact recovery       : {v['max_M_with_exact_cleanup_recovery']}")
    print(f"  max M with raw cos >= 0.92      : {v['max_M_with_raw_cos_ge_0.92']}")
    print(f"  crosstalk law that fits data    : {v['crosstalk_law_verdict']}")
    print(f"  worst rel err vs 1/sqrt(M)      : {worst_rel_1_over_sqrtM:.4f}")
    print(f"wrote {path}")
    return 0 if instrument_ok else 1


if __name__ == "__main__":
    sys.exit(main())
