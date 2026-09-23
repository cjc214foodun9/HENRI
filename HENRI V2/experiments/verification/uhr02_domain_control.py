"""UHR-02 DOMAIN CONTROL — exact equal-trace control + tau calibration.

SUPERSEDES the binary-search matching in an earlier draft of this file.
MY DEFECT (self-caught, printed by the previous run): I matched two options on
|Tr U| by binary-searching a scale, but |Tr U|(scale) is NOT monotonic for a
4-generator product, so the search returned |Tr U| = 1.9349 vs 1.1499. The
resulting "separation" was a MAGNITUDE difference, not content. The test was
vacuous. Never binary-search a non-monotonic quantity.

THE CORRECT CONTROL (exact by construction)
    Conjugation preserves the trace identically:
        Tr(V^dag U_A V) = Tr(U_A)          for any unitary V
    while giving a genuinely DIFFERENT group element (conjugated generator).
    So U_B := V^dag U_A V has EXACTLY |Tr U_B| = |Tr U_A| and a different
    adjoint rotation. That is the clean magnitude-matched pair.

THE MAXIM UNDER TEST
    "Fix the comparison's domain, not the comparison's threshold."

    FORM A (current live domain): compare candidate against the STATE it started
        from.   cos = (1/K) sum_k R_k^T A_c R_k  ->  Tr(A_c)/8 for isotropic R
                => Delta = (9 - |Tr U_c|^2)/16 : option MAGNITUDE only.
    FORM B (blueprint spec section 3.2, exteroceptive): compare candidate against
        the OBSERVED TRANSITION.  cos ~ Tr(A_c^T A_t)/8 = (|Tr(U_c^dag U_t)|^2-1)/8
                => reads the RELATIVE group element: does the candidate PREDICT
                   what the environment actually did.

PREDICTIONS (pre-registered before reading numbers)
    P1  FORM A, isotropic reference, EXACT-equal-trace pair: NO separation.
    P2  FORM A, structured reference, same pair: separation (content survives
        only when the reference is content-bearing).
    P3  FORM B, same pair: separation under BOTH references.
    KILL: if P1 AND P2 both show no separation, the channel is magnitude-only
        wherever it is pointed -> UHR-01 live value FALSIFIED.
"""
from __future__ import annotations
import math, pathlib, sys
import torch

_H = pathlib.Path(__file__).resolve()
for _c in (_H.parents[1], _H.parents[2], _H.parents[3]):
    if (_c / "uhr_rfss.py").exists():
        sys.path.insert(0, str(_c)); break
from uhr_rfss import adjoint_matrix  # noqa: E402

K = 8192
_s3 = math.sqrt(3.0)
BASIS = torch.tensor([
    [[0,1,0],[1,0,0],[0,0,0]], [[0,-1j,0],[1j,0,0],[0,0,0]],
    [[1,0,0],[0,-1,0],[0,0,0]], [[0,0,1],[0,0,0],[1,0,0]],
    [[0,0,-1j],[0,0,0],[1j,0,0]], [[0,0,0],[0,0,1],[0,1,0]],
    [[0,0,0],[0,0,-1j],[0,1j,0]], [[1/_s3,0,0],[0,1/_s3,0],[0,0,-2/_s3]],
], dtype=torch.complex64)
BAND = 0.5 * math.sqrt(2.0/10.0) / math.sqrt(K)      # 1-sigma, measured in uhr02_domain_probe


def roles_iso(seed, k=K):
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(k, 8, generator=g); return r / r.norm(dim=-1, keepdim=True)


def roles_struct(seed, coh=6.0, k=K):
    g = torch.Generator().manual_seed(seed)
    r = coh * torch.randn(8, generator=g).unsqueeze(0) + torch.randn(k, 8, generator=g)
    return r / r.norm(dim=-1, keepdim=True)


def gens(seed, sc, k=4):
    g = torch.Generator().manual_seed(seed)
    th = torch.randn(8, generator=g) * sc
    return [1j * torch.einsum("a,aij->ij", th.to(BASIS.dtype), BASIS) for _ in range(k)]


def U_of(gs):
    U = torch.eye(3, dtype=torch.complex64)
    for g in gs: U = U @ torch.matrix_exp(g)
    return U


def A_of(U):
    """Ad(U) via uhr_rfss.adjoint_matrix: A[c,a] = Tr(U la U^dag lc)/2."""
    return adjoint_matrix(torch.linalg.matrix_log if False else _log(U), BASIS)


def _log(U):
    """su(3) element log(U): a generator matrix compatible with adjoint_matrix."""
    # adjoint_matrix accepts a generator; use the principal branch via eig.
    w, Vm = torch.linalg.eig(U)
    L = Vm @ torch.diag(torch.log(w.to(torch.complex64))) @ torch.linalg.inv(Vm)
    # project onto the anti-Hermitian part (su(3) directions)
    return (L - L.conj().transpose(-2, -1)) / 2.0


def wave(R, A):
    y = torch.einsum("ij,kj->ki", A.to(R.dtype), R)
    return y / y.norm(dim=-1, keepdim=True)


def delta(a, b):
    x = a.flatten().to(torch.float32); y = b.flatten().to(torch.float32)
    return 0.5 * (1.0 - float((x*y).sum()) / (float(x.norm())*float(y.norm())))


def conj_pair(gs, seed=555):
    """U_B = V^dag U_A V : EXACTLY equal |Tr|, different rotation."""
    Ua = U_of(gs)
    g = torch.Generator().manual_seed(seed)
    H = torch.randn(3, 3, generator=g, dtype=torch.complex64)
    H = H + H.conj().transpose(-2, -1)                     # Hermitian
    V = torch.matrix_exp(1j * H)
    Ub = V.conj().transpose(-2, -1) @ Ua @ V
    return Ua, Ub, V


def main():
    print("torch", torch.__version__, "| K =", K, "| band = %.3e" % BAND)
    gs = gens(4242, 0.30)
    U_A, U_B, _ = conj_pair(gs)
    trA = float(torch.abs(U_A.trace().reshape(())))
    trB = float(torch.abs(U_B.trace().reshape(())))
    A_A, A_B = A_of(U_A), A_of(U_B)
    print("EXACT-TRACE CONTROL")
    print("  |Tr U_A| = %.10f" % trA)
    print("  |Tr U_B| = %.10f   |diff| = %.3e  <- must be ~machine precision"
          % (trB, abs(trA - trB)))
    print("  ||Ad(A) - Ad(B)||_F = %.4f  <- genuinely different rotation"
          % float((A_A - A_B).norm()))
    print("  orth_err(A)=%.2e orth_err(B)=%.2e" %
          (float((A_A@A_A.T - torch.eye(8)).abs().max()),
           float((A_B@A_B.T - torch.eye(8)).abs().max())))
    print()
    res = {}
    for nm, ref in (("ISOTROPIC", roles_iso(1)), ("STRUCTURED c=6", roles_struct(1, 6.0))):
        gn = torch.Generator().manual_seed(77)
        R_obs = wave(ref, A_A) + 0.05 * torch.randn(K, 8, generator=gn)
        R_obs = R_obs / R_obs.norm(dim=-1, keepdim=True)
        dAa, dAb = delta(wave(ref, A_A), ref), delta(wave(ref, A_B), ref)
        dBa, dBb = delta(wave(ref, A_A), R_obs), delta(wave(ref, A_B), R_obs)
        d_none = delta(ref, R_obs)
        print("=== %s ===" % nm)
        print("  FORM A (vs current state)  true=%.9f  conj=%.9f  sep=%.3e  band=%.3e  %s"
              % (dAa, dAb, abs(dAa-dAb), BAND,
                 "NO-SEP(magnitude-only)" if abs(dAa-dAb) < 4*BAND else "SEP"))
        print("  FORM B (vs observed trans) true=%.9f  conj=%.9f  sep=%.3e  band=%.3e  %s"
              % (dBa, dBb, abs(dBa-dBb), BAND,
                 "NO-SEP" if abs(dBa-dBb) < 4*BAND else "SEP"))
        print("  do-nothing candidate delta  = %.9f" % d_none)
        res[nm] = (abs(dAa-dAb), abs(dBa-dBb))
    (sA_iso, sB_iso) = res["ISOTROPIC"]
    (sA_str, sB_str) = res["STRUCTURED c=6"]
    print()
    print("=== TAU CALIBRATION (FORM B, population at the store's natural scale) ===")
    comp, inval = [], []
    for sd in (101,102,103,104,105,106):
        gt = gens(sd, 0.25); At = A_of(U_of(gt))
        gan = torch.Generator().manual_seed(sd+50)
        Robs = wave(roles_iso(1), At) + 0.05*torch.randn(K,8,generator=gan)
        Robs = Robs / Robs.norm(dim=-1, keepdim=True)
        comp.append(delta(wave(roles_iso(1), At), Robs))          # predicts what happened
        gx = gens(sd+7000, 0.25)                                   # unrelated option
        inval.append(delta(wave(roles_iso(1), A_of(U_of(gx))), Robs))
    print("  compliant (predicts transition) n=%d min=%.6f max=%.6f mean=%.6f"
          % (len(comp), min(comp), max(comp), sum(comp)/len(comp)))
    print("  invalid   (unrelated option)    n=%d min=%.6f max=%.6f mean=%.6f"
          % (len(inval), min(inval), max(inval), sum(inval)/len(inval)))
    gap_lo, gap_hi = max(comp), min(inval)
    print("  measured band for tau = (%.6f, %.6f)   blueprint 0.35 %s"
          % (gap_lo, gap_hi,
             "LIES IN BAND" if gap_lo < 0.35 < gap_hi else "IS OUTSIDE THE MEASURED BAND"))
    print()
    print("VERDICT")
    print("  P1 FORM A isotropic  no-sep : %s  (sep=%.3e)" % (sA_iso < 4*BAND, sA_iso))
    print("  P2 FORM A structured sep    : %s  (sep=%.3e)" % (sA_str > 4*BAND, sA_str))
    print("  P3 FORM B separates, iso    : %s  (sep=%.3e)" % (sB_iso > 4*BAND, sB_iso))
    print("  P3 FORM B separates, struct : %s  (sep=%.3e)" % (sB_str > 4*BAND, sB_str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
