"""henri_readout -- a NULL-TESTED linear-readability readout. Shared infrastructure.

WHY THIS MODULE EXISTS
======================
Every probe this session scored representations with a ridge readout whose penalty
was a FIXED constant scaled by n:

    k = X_std @ X_std.T ;  a = solve(k + lam_rel * n * I, y)

MEASURED FAILURE (vlm_control_artifact_receipt.json, RTX 5090): the PERMUTATION
NULL -- feature row i deliberately paired with state row j != i, which MUST score
~0 -- returned

    r2_scalar = -1.4811      r2_perdim = -1.4822

A strongly negative held-out R^2 is the signature of catastrophic overfitting, not
of a real effect. The arithmetic is simple: after standardisation the Gram diagonal
is ~n_train (192 here), and the penalty is 1e-3 * 256 = 0.256 -- effectively ZERO
shrinkage. With d = 65536 >> n_train the Gram matrix is full rank but has many
near-zero eigenvalues in noise directions, so the solve explodes.

CONSEQUENCE: absolute r2 values at d = 65536 from that readout are NOT trustworthy.
This includes the numbers behind the 'bridge discards state' conclusion.

THE FIX
-------
Regularise RELATIVE TO THE GRAM SCALE, not to n:

    lam = lam_frac * trace(k) / k.shape[0]        (lam_frac * mean eigenvalue)

`trace(k)/n` is the mean eigenvalue, so `lam_frac` is a dimensionless shrinkage
fraction and the same `lam_frac` means the same thing regardless of d or n.

WHAT THIS MODULE GUARANTEES, AND HOW
------------------------------------
`self_test()` runs four cases that a trustworthy readout must pass:
  1. permutation null      -> r2 ~ 0     (catches overfitting)
  2. pure noise            -> r2 ~ 0     (catches a readout that finds anything)
  3. informative low-d      -> r2 high    (catches over-shrinkage)
  4. informative high-d     -> r2 comparable to low-d at equal intrinsic rank
A readout that cannot pass its own nulls is not evidence for anything.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["r2_ridge", "offdiag_cos", "participation_ratio", "self_test", "unit"]

DEFAULT_LAM_FRAC = 1e-2


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def _standardize(xtr: torch.Tensor, xte: torch.Tensor, per_dim: bool = True):
    xm = xtr.mean(0, keepdim=True)
    xtr, xte = xtr - xm, xte - xm
    if per_dim:
        s = xtr.std(0, keepdim=True)
        floor = float(s.median()) * 1e-2
        s = s.clamp(min=max(floor, 1e-30))
    else:
        s = xtr.std().clamp(min=1e-12)
    return xtr / s, xte / s


def r2_ridge(x: torch.Tensor, y: torch.Tensor, frac: float = 0.25,
             lam_frac: float = DEFAULT_LAM_FRAC, seed: int = 0,
             per_dim: bool = True) -> float:
    """Held-out R^2 from a ridge readout with TRACE-SCALED regularisation.

    x : [n, d] features.  y : [n, m] targets.
    lam_frac : shrinkage as a fraction of the mean Gram eigenvalue. This is
               dimensionless, so the value is comparable across dimensions.
    """
    n = int(x.shape[0])
    if n < 8:
        return float("nan")
    nte = max(1, int(n * frac))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    te, tr = perm[:nte], perm[nte:]
    xtr, xte = _standardize(x[tr].double(), x[te].double(), per_dim)
    ytr, yte = y[tr].double(), y[te].double()
    ym = ytr.mean(0, keepdim=True)
    ytr, yte = ytr - ym, yte - ym
    k = xtr @ xtr.T
    lam = float(lam_frac) * float(torch.diagonal(k).mean())
    a = torch.linalg.solve(k + lam * torch.eye(k.shape[0], dtype=k.dtype), ytr)
    pred = (xte @ xtr.T) @ a
    res = ((yte - pred) ** 2).sum(0)
    tot = (yte ** 2).sum(0).clamp(min=1e-12)
    return float((1.0 - res / tot).mean())


def _r2_legacy(x: torch.Tensor, y: torch.Tensor, frac: float = 0.25,
               lam_rel: float = 1e-3, seed: int = 0) -> float:
    """The SUPERSEDED readout, kept ONLY to demonstrate its defect in self_test().

    DO NOT USE for new measurements. It regularises by `lam_rel * n`, an absolute
    penalty that is effectively zero once features are standardized (Gram diagonal
    ~ n_train), so at d >> n the solve overfits and the PERMUTATION NULL returns a
    strongly NEGATIVE R^2 (~-1.48 at d=65536, measured 2026-09-29). Any absolute r2
    produced by this function at high dimension is untrustworthy.
    """
    n = int(x.shape[0])
    nte = max(1, int(n * frac))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    te, tr = perm[:nte], perm[nte:]
    xtr, xte = x[tr].double(), x[te].double()
    ytr, yte = y[tr].double(), y[te].double()
    xm, ym = xtr.mean(0, keepdim=True), ytr.mean(0, keepdim=True)
    s = xtr.std(0, keepdim=True).mean().clamp(min=1e-12)
    xtr, xte = (xtr - xm) / s, (xte - xm) / s
    ytr, yte = ytr - ym, yte - ym
    k = xtr @ xtr.T
    a = torch.linalg.solve(k + lam_rel * n * torch.eye(k.shape[0], dtype=k.dtype), ytr)
    pred = (xte @ xtr.T) @ a
    res = ((yte - pred) ** 2).sum(0)
    tot = (yte ** 2).sum(0).clamp(min=1e-12)
    return float((1.0 - res / tot).mean())


def offdiag_cos(f: torch.Tensor) -> float:
    """Mean pairwise cosine over rows -- the collapse/collinearity monitor."""
    z = F.normalize(f.reshape(f.shape[0], -1).double(), dim=-1)
    c = z @ z.T
    n = c.shape[0]
    if n < 2:
        return float("nan")
    return float((c.sum() - c.diag().sum()) / (n * (n - 1)))


def participation_ratio(f: torch.Tensor) -> float:
    """Effective number of active directions (1.0 == rank-1 collapse)."""
    g = (f.reshape(f.shape[0], -1).double() @ f.reshape(f.shape[0], -1).double().T)
    lam = torch.linalg.eigvalsh(g.cpu()).clamp(min=0.0)
    s = lam.sum()
    return float((s * s) / (lam * lam).sum().clamp(min=1e-30))


def self_test(verbose: bool = True) -> dict:
    """Nulls a trustworthy readout must pass. Returns a dict of passed flags."""
    torch.manual_seed(0)
    n, sdim = 256, 4
    out: dict = {"schema": "henri.readout_selftest/1", "n": n, "state_dim": sdim,
                 "lam_frac": DEFAULT_LAM_FRAC, "cases": {}}

    S = torch.rand(n, sdim, generator=torch.Generator().manual_seed(1)) * 0.5 + 0.25

    def informative(d: int, rank: int, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        # DEFECT FIXED 2026-09-29: W was built as [rank, d] and multiplied by the
        # [n, sdim] state, raising "mat1 and mat2 shapes cannot be multiplied
        # (256x4 and 512x65536)". The map must be COMPOSED -- state -> rank -> d --
        # which also makes the intended intrinsic rank explicit.
        A = torch.randn(sdim, rank, generator=g)
        B = torch.randn(rank, d, generator=g)
        return (S - S.mean(0, keepdim=True)) @ A @ B + 1e-3 * torch.randn(
            n, d, generator=g)

    # 1. permutation null -- MUST be ~0. The FIXED readout must pass where the
    #    superseded one fails; both are reported so the defect is visible here.
    x = informative(65536, 512, 2)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(3))
    out["cases"]["permutation_null_d65536"] = round(r2_ridge(x, S[idx]), 4)
    out["cases"]["permutation_null_d65536_LEGACY_READOUT"] = round(
        _r2_legacy(x, S[idx]), 4)
    # 2. pure noise -- MUST be ~0
    out["cases"]["pure_noise_d65536"] = round(
        r2_ridge(torch.randn(n, 65536, generator=torch.Generator().manual_seed(4)), S), 4)
    # 3. informative low-d -- MUST be high
    out["cases"]["informative_d512"] = round(r2_ridge(informative(512, 512, 5), S), 4)
    # 4. informative high-d -- MUST stay high (the case the old readout broke)
    out["cases"]["informative_d65536"] = round(
        r2_ridge(informative(65536, 512, 6), S), 4)

    c = out["cases"]
    out["verdicts"] = {
        "permutation_null_holds": abs(c["permutation_null_d65536"]) < 0.10,
        "pure_noise_unreadable": abs(c["pure_noise_d65536"]) < 0.10,
        "informative_low_d_readable": c["informative_d512"] > 0.80,
        "informative_high_d_readable": c["informative_d65536"] > 0.80,
    }
    out["READOUT_TRUSTWORTHY"] = bool(all(out["verdicts"].values()))
    if verbose:
        import json
        print(json.dumps(out, indent=2))
        print("\nRESULT:", "PASS" if out["READOUT_TRUSTWORTHY"] else "FAIL")
    return out


if __name__ == "__main__":
    raise SystemExit(0 if self_test()["READOUT_TRUSTWORTHY"] else 1)
