"""D3-SUPPLEMENT v2 -- NON-NORMALITY AUDIT of the low-rank Koopman operator.

WHY (delegation Q2, deleg_fa46fa1e)
===================================
The digest names a rollout failure mode my D3 verification did NOT cover:
NON-NORMAL TRANSIENT GROWTH -- ||K^k||_2 can exceed rho(K)^k by orders of
magnitude even while rho(K) < 1. My `d3_truncation_probe.py` used an
ORTHOGONAL synthetic truth, and orthogonal == NORMAL, so it measured only the
benign case and I generalised from it.

DEFECT FOUND IN MY OWN PREDECESSOR PROBE (self-reported, this is why v2 exists)
==============================================================================
The v1 run called `spectral_radius()` via `torch.linalg.eigvals` and printed

    rho(K_nn) = 1.219232   (< 1 => spectrally STABLE)

for a matrix whose eigenvalues are EXACTLY 0.95 by construction. The printed
line contradicted itself. Cause: the eigenvalue problem is ILL-CONDITIONED for
non-normal matrices, so eigvals() on an orthogonally-similar copy of a highly
non-normal matrix returns numbers that can be wrong by O(1). Effect: v1
UNDERSTATED the transient by dividing by 1.219^12 instead of 0.95^12.

METHOD
======
rho = lim_k ||K^k||^(1/k)  (Gelfand). This uses only matrix norms, so it is
reliable where eigvals() is not. Both are reported, to EXPOSE the gap rather
than hide it. The operator is built so that the TRUE rho is known exactly.

SECTIONS
  A. instrument control -- a NORMAL (orthogonal) operator must give ||K^k||=1
     for every k. If not, the instrument is broken and B-D are void.
  B. rho: eigvals() vs Gelfand, against the analytic value.
  C. transient amplification A(k) = ||K^k|| / rho_true^k.
  D. does the LOW-RANK fit capture the transient? (r in {8, 32, 64})
  E. rollout error vs horizon, dense vs low-rank. NOTE: `rollout_error`
     multiplies a float32 state by the truth operator, so a float64 truth
     raises "expected scalar type Double but found Float" -- the truth is
     cast to float32 here. (v1 hit that and skipped the section.)
  F. cross-action subspace alignment: principal angles between per-action U_a.

The receipt path takes an override so a diagnostic re-run cannot clobber the
committed artifact (the 9a8fbce defect class):
    --out  >  HENRI_RECEIPT_DIR  >  committed default.
A malformed --out fails CLOSED, at entry, before any D=65,536 work.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
import henri_action_koopman as K                                    # noqa: E402

DIM = 64                       # the property is dimension-independent;
N_A = 4                        # D=65536 would cost minutes for the same answer
N_PER = 128                    # n > d, so rank is not sample-limited
LAM = 0.95                     # analytic rho
COUPLING = 0.5                 # makes it strongly non-normal
HORIZONS = (1, 3, 5, 10)
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "d3_nonnormality_probe.json")


def resolve_out() -> str:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a, _ = ap.parse_known_args()
    if a.out:
        d = os.path.dirname(os.path.abspath(a.out))
        if not os.path.isdir(d):
            print(f"MALFORMED --out: directory does not exist: {d}", file=sys.stderr)
            raise SystemExit(2)
        return os.path.abspath(a.out)
    envd = os.environ.get("HENRI_RECEIPT_DIR")
    if envd:
        os.makedirs(envd, exist_ok=True)
        return os.path.join(envd, "d3_nonnormality_probe.json")
    return DEFAULT_OUT


def rho_eigvals(M: torch.Tensor) -> float:
    """UNRELIABLE for non-normal M. Reported to expose the gap, never trusted."""
    ev = torch.linalg.eigvals(M.to(torch.float64))
    return float(ev.abs().max())


def gelfand(M: torch.Tensor, jmax: int = 7) -> list[dict]:
    """rho = lim ||M^k||^(1/k). Repeated squaring: k = 1,2,4,...,2^(jmax-1)."""
    cur = M.to(torch.float64)
    rows, k = [], 1
    for _ in range(jmax):
        nrm = float(torch.linalg.matrix_norm(cur, ord=2))
        rows.append({"k": k, "norm": nrm, "norm_pow_1_over_k": nrm ** (1.0 / k)})
        cur = cur @ cur
        k *= 2
    return rows


def power_norms(M: torch.Tensor, kmax: int) -> list[float]:
    out, cur = [], torch.eye(M.shape[0], dtype=torch.float64)
    for _ in range(kmax):
        cur = cur @ M.to(torch.float64)
        out.append(float(torch.linalg.matrix_norm(cur, ord=2)))
    return out


def make_nonnormal(dim: int, lam: float, coupling: float, seed: int) -> torch.Tensor:
    """K = lam (I + coupling * N), N = strictly-superdiagonal shift.

    N is nilpotent => I + coupling*N is UPPER TRIANGULAR with ones on the
    diagonal => all eigenvalues are exactly 1 => rho(K) = lam exactly.
    The random orthogonal similarity transform below preserves the eigenvalues
    EXACTLY in exact arithmetic; it is what makes eigvals() numerically fail.
    """
    g = torch.Generator().manual_seed(seed)
    N = torch.zeros(dim, dim, dtype=torch.float64)
    idx = torch.arange(dim - 1)
    N[idx, idx + 1] = 1.0
    M = lam * (torch.eye(dim, dtype=torch.float64) + coupling * N)
    A = torch.randn(dim, dim, generator=g, dtype=torch.float64)
    q, _ = torch.linalg.qr(A)
    return q @ M @ q.t()


def make_triples(Ktrue: dict):
    """Action-labelled one-step transitions from an explicit K per action.

    Deterministic: the generator is seeded here, so `fit_on` and the dense arm
    below see EXACTLY the same data (a mismatch there would silently compare
    two different problems).
    """
    g = torch.Generator().manual_seed(99)
    triples = []
    for a in range(N_A):
        for _ in range(N_PER):
            s = torch.randn(DIM, generator=g, dtype=torch.float64)
            s = (s / s.norm()).to(torch.float32)
            s1 = (Ktrue[a] @ s.to(torch.float64)).to(torch.float32)
            s1 = s1 + (0.01 * torch.randn(DIM, generator=g) / math.sqrt(DIM)).to(torch.float32)
            triples.append((s, a, s1))
    return triples


def fit_on(Ktrue: dict, rank):
    """Fit the low-rank path on triples generated from an explicit K per action."""
    return K.ActionConditionedKoopman(dim=DIM, n_actions=N_A, rank=rank).fit(
        make_triples(Ktrue))


def rebuilt(m, a) -> torch.Tensor:
    return (m.U[a] @ m.Vs[a].t()).to(torch.float64)


def main() -> int:
    out = resolve_out()
    R: dict = {"schema": "henri.d3-nonnormality-probe.v2",
               "purpose": "close the delegation Q2 non-normality gap; supersede v1",
               "dim": DIM, "n_per_action": N_PER, "lam_analytic_rho": LAM,
               "coupling": COUPLING}
    print("[nn] receipt:", out, flush=True)

    # ---------------------------------------------------------------- A
    print("\n=== A. INSTRUMENT CONTROL (normal operator: ||K^k|| must be 1) ===")
    _, truth_orth = K.make_synthetic_triples(N_A, DIM, N_PER, 11, rng_scale=0.02)
    pn_orth = power_norms(truth_orth[0], 6)
    ctrl_ok = all(abs(x - 1.0) < 1e-6 for x in pn_orth)
    gel_orth = gelfand(truth_orth[0], jmax=5)
    print("  orthogonal truth ||K^k|| k=1..6 =", [round(x, 6) for x in pn_orth])
    print("  Gelfand ||K^k||^(1/k)          =",
          [round(r["norm_pow_1_over_k"], 6) for r in gel_orth])
    print("  CONTROL PASSES                 =", ctrl_ok)
    R["control"] = {"power_norms": pn_orth, "gelfand": gel_orth, "passes": ctrl_ok}
    if not ctrl_ok:
        print("  INSTRUMENT BROKEN -> later sections VOID")
        json.dump(R, open(out, "w", encoding="utf-8"), indent=2)
        return 1

    # ---------------------------------------------------------------- B
    print("\n=== B. rho: eigvals() vs Gelfand, against the ANALYTIC value ===")
    Knn = {a: make_nonnormal(DIM, LAM, COUPLING, 7 + a) for a in range(N_A)}
    r_eig = rho_eigvals(Knn[0])
    gel = gelfand(Knn[0], jmax=7)
    r_gel = gel[-1]["norm_pow_1_over_k"]
    print(f"  analytic rho (by construction) = {LAM:.6f}")
    print(f"  eigvals().abs().max()          = {r_eig:.6f}"
          f"   <-- WRONG by {r_eig - LAM:+.6f}")
    print(f"  Gelfand ||K^64||^(1/64)        = {r_gel:.6f}")
    print("  Gelfand table (k, ||K^k||, ^(1/k)):")
    for r in gel:
        print(f"    k={r['k']:>3}  ||K^k||={r['norm']:>12.4f}  ^(1/k)={r['norm_pow_1_over_k']:.6f}")
    R["rho"] = {"analytic": LAM, "eigvals": r_eig, "eigvals_error": r_eig - LAM,
                "gelfand_table": gel, "gelfand_final": r_gel,
                "gelfand_converged": abs(r_gel - LAM) / LAM < 0.05,
                "gelfand_note": ("Gelfand needs k->inf; a strongly NON-NORMAL "
                                 "matrix has NOT converged by k=64, so a Gelfand "
                                 "value far ABOVE the analytic rho is itself a "
                                 "measurement of the severity of the non-normality.")}

    # ---------------------------------------------------------------- C
    print("\n=== C. TRANSIENT AMPLIFICATION A(k) = ||K^k|| / rho_true^k ===")
    pn = power_norms(Knn[0], 12)
    amps_correct = [pn[k - 1] / (LAM ** k) for k in range(1, 13)]
    amps_wrong = [pn[k - 1] / (r_eig ** k) for k in range(1, 13)]
    peak_k = int(max(range(1, 13), key=lambda k: amps_correct[k - 1]))
    print(f"  ||K^k|| k=1..12      = {[round(x,4) for x in pn]}")
    print(f"  A(k) vs TRUE rho     = {[round(x,4) for x in amps_correct]}")
    print(f"  A(k) vs eigvals rho  = {[round(x,4) for x in amps_wrong]}  (v1's wrong basis)")
    print(f"  PEAK at k={peak_k}: TRUE amplification = {amps_correct[peak_k-1]:.2f}x"
          f"   (v1 reported {amps_wrong[peak_k-1]:.2f}x)"
          f"   -> v1 UNDERSTATED by {amps_correct[peak_k-1]/amps_wrong[peak_k-1]:.1f}x")
    R["transient"] = {"power_norms": pn,
                      "amplification_true_rho": amps_correct,
                      "amplification_eigvals_rho": amps_wrong,
                      "peak_k": peak_k,
                      "peak_true": amps_correct[peak_k - 1],
                      "peak_as_v1_reported": amps_wrong[peak_k - 1]}

    # ---------------------------------------------------------------- D
    print("\n=== D. DOES THE LOW-RANK FIT CAPTURE THE TRANSIENT? ===")
    dense_pk = None
    capture = {}
    for r in (8, 32, 64):
        m = fit_on(Knn, r)
        Kr = rebuilt(m, 0)
        pn_r = power_norms(Kr, peak_k)
        pk = pn_r[peak_k - 1]
        capture[str(r)] = {"eff_rank": m.effective_rank[0], "norm_at_peak": pk,
                           "ratio_vs_dense": pk / pn[peak_k - 1]}
        if r == 64:
            dense_pk = pk
        print(f"  r={r:>3} eff={m.effective_rank[0]:>3}  ||K_r^{peak_k}||={pk:>9.4f}  "
              f"||K_dense^{peak_k}||={pn[peak_k-1]:>9.4f}  ratio={pk/pn[peak_k-1]:.4f}")
    R["lowrank_capture"] = capture
    R["lowrank_captures_transient_at_r64"] = bool(
        dense_pk is not None and abs(dense_pk - pn[peak_k - 1]) / pn[peak_k - 1] < 0.05)

    # ---------------------------------------------------------------- E
    print("\n=== E. ROLLOUT ERROR vs HORIZON (dense vs low-rank) ===")
    # `rollout_error` multiplies a float32 state by the truth operator, so the
    # truth MUST be float32 (v1 raised "expected scalar type Double but found
    # Float" and skipped this section entirely).
    truth32 = {a: Knn[a].to(torch.float32) for a in Knn}
    roll = {}
    for r in (32, 64):
        m = fit_on(Knn, r)
        roll[f"lowrank_r{r}"] = [K.rollout_error(m, truth32, DIM, horizon=h)
                                 for h in HORIZONS]
        print(f"  lowrank r={r:>3}: " +
              "  ".join(f"h{h}={e:.5f}" for h, e in zip(HORIZONS, roll[f"lowrank_r{r}"])))
    try:
        m_dense = K.ActionConditionedKoopman(dim=DIM, n_actions=N_A).fit(make_triples(Knn))
        roll["dense"] = [K.rollout_error(m_dense, truth32, DIM, horizon=h) for h in HORIZONS]
        print("  DENSE        : " +
              "  ".join(f"h{h}={e:.5f}" for h, e in zip(HORIZONS, roll["dense"])))
    except Exception as exc:                                        # noqa: BLE001
        print("  dense rollout_error failed:", repr(exc)[:110])
    R["rollout_error_by_horizon"] = roll
    R["horizons"] = list(HORIZONS)

    # ---------------------------------------------------------------- F
    print("\n=== F. CROSS-ACTION SUBSPACE ALIGNMENT (principal angles) ===")
    mm = fit_on(Knn, 32)
    basis = {}
    for a in sorted(mm.U):
        q, _ = torch.linalg.qr(mm.U[a].to(torch.float64))
        basis[a] = q
    worst, angles = 180.0, {}
    acts = sorted(basis)
    for i in range(len(acts)):
        for j in range(i + 1, len(acts)):
            s = torch.linalg.svdvals(basis[acts[i]].t() @ basis[acts[j]])
            ang = torch.rad2deg(torch.arccos(s.clamp(-1, 1)))
            angles[f"{acts[i]}-{acts[j]}"] = {"min_deg": float(ang.min()),
                                              "mean_deg": float(ang.mean())}
            worst = min(worst, float(ang.min()))
    for k, v in angles.items():
        print(f"  U_{k}: min={v['min_deg']:8.4f} deg  mean={v['mean_deg']:7.4f} deg")
    print(f"  SMALLEST angle across pairs = {worst:.4f} deg")
    R["cross_action_angles"] = angles
    R["cross_action_min_angle_deg"] = worst
    R["cross_action_note"] = (
        "Small minimum angle => the per-action U_a subspaces overlap, so a state "
        "held in one action's coordinates IS partly expressible in another's. "
        "Mean angle near 45 deg => they are otherwise close to orthogonal.")

    # ---------------------------------------------------------------- G
    print("\n=== G. PRODUCTION SCOPE OF THE TRANSIENT ===")
    print("  `step()` normalises its output when enforce_unit_norm=True, so the")
    print("  PLANNER boundary is bounded. The growth above is the RAW operator;")
    print("  any consumer reading _apply_* directly, or rolling with normalisation")
    print("  off, sees it unbounded. Estimated rho from eigvals() must not be used")
    print("  as a stability gate on a non-normal K.")
    R["production_scope"] = ("enforce_unit_norm=True bounds the transient at the "
                             "step() boundary; raw _apply_* is unbounded")

    R["verdict_inputs"] = {
        "control_passes": ctrl_ok,
        "analytic_rho": LAM,
        "rho_from_eigvals_UNRELIABLE": r_eig,
        "rho_from_gelfand": r_gel,
        "gelfand_converged": abs(r_gel - LAM) / LAM < 0.05,
        "peak_transient_k": peak_k,
        "peak_transient_amplification": amps_correct[peak_k - 1],
        "v1_understated_by_factor": amps_correct[peak_k - 1] / amps_wrong[peak_k - 1],
        "lowrank_captures_transient_at_r64": R["lowrank_captures_transient_at_r64"],
        "cross_action_min_angle_deg": worst,
    }
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)
    print("\n[nn] WROTE", out)
    print("[nn] verdict inputs:", json.dumps(R["verdict_inputs"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
