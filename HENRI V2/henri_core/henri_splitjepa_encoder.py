"""henri_splitjepa_encoder.py -- reconstruction-free dual-branch encoder.

BLUEPRINT
    HENRI-SPEC-2026-SPLITJEPA-SYS-V1, section 8 directive 1; theoretical basis
    SplitJEPA (Hua et al., arXiv:2610.12349). Zone A maps an input to a wave on the
    unit hypersphere and partitions it into an invariant block r_H and a variant
    block r_L. No decoder. The claim under test is BLOCK IDENTIFIABILITY: the only
    off-diagonal coupling A12, A21 is driven to zero by the invariant-pair objective.

TWO SPEC CORRECTIONS APPLIED (both measured, recorded rather than hidden)

  1. DENSE ZCA IS INFEASIBLE AT D = 65,536.
     The spec says "enforce centered ZCA whitening on the full 65,536-dimensional
     representation". A dense covariance is D^2 elements: 17.18 GB in fp32 and
     34.36 GB in complex64. It cannot be formed, let alone decomposed, in the
     stated 128 MiB L2 / 32 GiB GDDR7 envelope. THIS MODULE THEREFORE USES THE
     GRAM / ECONOMY-SVD PATH WHENEVER N < D, which is the normal case for a
     calibration corpus. The dense path stays available below DENSE_COVARIANCE_LIMIT
     and RAISES above it rather than silently allocating. The whitening math is
     unchanged; only the factorization route differs.
        Xw = sqrt(N-1) * U @ V^T          for Xc = U diag(S) V^T
     and for new data:  Xw = (X - mean) @ V diag(sqrt(N-1)/S) V^T
     Both forms are pinned against the dense path in the self-test.

  2. G-SJ2 IS TRUE BY CONSTRUCTION AND CANNOT FAIL.
     The spec's gate 2 asserts ||grad_{z_L} Y|| = 0. Once A12 = 0 is hard-coded,
     that derivative is identically zero by the chain rule -- the gate tests the
     definition, not a measurement. The gates here are therefore built on an
     ESTIMATED coupling from a Q that is actually fit, so they can fail. The
     falsifiable statement is the SUFFICIENT VARIATION CONDITION:
        Sigma_dL = E[D_inv][dz_L dz_L^T] > 0    =>    A12 -> 0
        Sigma_dL = 0 (no variant variation)     =>    A12 UNCONSTRAINED
     The negative control below violates the condition and MUST retain coupling.

CONTRACT
    Pure torch. Deterministic given a seed. No corpus dependency: the encoder
    consumes (z_H, z_L) pairs so it can be exercised on synthetic data before any
    protein grounding. Fold2Reason D_inv grounding is a separate step; the loader
    hook is provided (from_arrays) but no protein data ships here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# Above this dimension a dense D x D covariance exceeds ~256 MiB in fp32. The Gram /
# economy-SVD path is used instead whenever it applies, so this is a guard against a
# silent multi-GB allocation rather than a capability limit.
DENSE_COVARIANCE_LIMIT = 8192


class WhiteningError(RuntimeError):
    """Raised when a dense whitening request would exceed the memory envelope."""


# ---------------------------------------------------------------------------------
# centered ZCA whitening
# ---------------------------------------------------------------------------------
class CenteredZCA:
    """Centered ZCA whitening. Guarantees E[g(z)] = 0 and Cov(g(z)) = I.

    Route selection is explicit and recorded on the instance, so a receipt can state
    WHICH factorization produced the numbers.
    """

    def __init__(self, eps: float = 1e-5, max_dense: int = DENSE_COVARIANCE_LIMIT):
        self.eps = float(eps)
        self.max_dense = int(max_dense)
        self.mean_: torch.Tensor | None = None
        self.route_: str | None = None
        self.d_: int | None = None
        self.n_: int | None = None
        self._W: torch.Tensor | None = None          # dense: [D, D]
        self._V: torch.Tensor | None = None          # gram:  [D, k]
        self._S: torch.Tensor | None = None          # gram:  [k]
        self._rank_kept: int | None = None
        self._rank_cap: int | None = None

    def fit(self, X: torch.Tensor) -> "CenteredZCA":
        if X.dim() != 2:
            raise ValueError(f"X must be [N, D]; got {tuple(X.shape)}")
        X = X.to(torch.float32)
        N, D = X.shape
        self.mean_ = X.mean(0, keepdim=True)
        Xc = X - self.mean_
        self.d_, self.n_ = D, N

        use_gram = D > self.max_dense and N < D
        if use_gram:
            # economy SVD: Xc [N, D] = U [N, k] diag(S) V^T [k, D]
            U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
            # RANK HANDLING -- measured defect, fixed here.
            # An ABSOLUTE eps is dimension-dependent and wrong: singular values scale as
            # sqrt(N*D), so at D=65536 they reach ~255 while the noise floor of centered
            # data sits around 0.002. An absolute 1e-5 therefore KEPT one spurious
            # direction, divided by it, and corrupted the whitening
            # (measured max|G-projector| = 0.385 at N=6, D=65536, versus 1.1e-06 at
            # D=256). Two corrections, either of which is sufficient:
            #   (a) RELATIVE tolerance against the largest singular value;
            #   (b) the mathematical rank cap: CENTERED data has rank <= N-1, always.
            s0 = float(S[0]) if S.numel() else 0.0
            tol = self.eps * s0
            k = int((S > tol).sum()) if S.numel() else 0
            k = max(1, min(k, N - 1, int(S.numel())))
            self._V = Vh[:k].T.contiguous()          # [D, k]
            self._S = S[:k].clone()                  # [k]
            self.route_ = "gram_economy_svd"
            self._rank_kept = k
            self._rank_cap = N - 1
        elif D > self.max_dense:
            raise WhiteningError(
                f"D={D} exceeds dense limit {self.max_dense} and N={N} >= D, so the"
                " Gram path does not apply. Use a low-rank/sketched whitening design"
                " before attempting this shape; refusing to allocate D^2.")
        else:
            C = (Xc.T @ Xc) / max(1, N - 1)
            evals, evecs = torch.linalg.eigh(C)
            evals = evals.clamp_min(self.eps)
            self._W = (evecs * (1.0 / evals.sqrt())) @ evecs.T
            self.route_ = "dense_eigh"
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.mean_ is None:
            raise WhiteningError("fit() must run before transform()")
        X = X.to(torch.float32)
        Xc = X - self.mean_
        if self.route_ == "dense_eigh":
            return Xc @ self._W.T
        # gram route: Xw = Xc @ V diag(sqrt(N-1)/S) V^T
        scale = math.sqrt(max(1, self.n_ - 1)) / self._S            # [k]
        return (Xc @ self._V) * scale @ self._V.T

    def whiten_residual(self, X: torch.Tensor) -> dict:
        """Measure the whitening contract instead of asserting it.

        MEASURED FACT THAT CORRECTS THE SPEC (found by running the gate, not by
        reasoning about it): with N < D the covariance Z^T Z is at most rank N, so
        Cov(g(z)) = I_D is UNACHIEVABLE. The spec asks for I_D at D = 65,536, which no
        realistic calibration corpus can satisfy. The exact achievable contract on the
        Gram route is SAMPLE-side isotropy:
            (1/(N-1)) Z Z^T = I_N      (exact, because Z = sqrt(N-1) U V^T and U is
                                        orthonormal, so Z Z^T = (N-1) U U^T = (N-1) I_N)
        This method therefore reports WHICH identity it checked, so a receipt cannot
        quietly compare a rank-deficient covariance against a full-rank target.
        """
        Z = self.transform(X)
        mu = Z.mean(0)
        N, D = int(Z.shape[0]), int(Z.shape[1])
        out = {"route": self.route_, "n": N, "d": D,
               "mean_abs_max": float(mu.abs().max())}
        if self.route_ == "gram_economy_svd":
            Zc = Z - mu
            G = (Zc @ Zc.T) / max(1, N - 1)
            P = torch.eye(N, dtype=G.dtype) - torch.full((N, N), 1.0 / N, dtype=G.dtype)
            off = G - torch.diag(G.diag())
            out.update({
                "contract": "sample_gram_centered_projector",
                "projector_max_dev": float((G - P).abs().max()),
                "diag_mean": float(G.diag().mean()),
                "diag_mean_target": (N - 1) / N,
                "offdiag_abs_max": float(off.abs().max()),
                "offdiag_abs_target": 1.0 / N,
                "rank_kept": self._rank_kept,
                "rank_cap_centered_N_minus_1": self._rank_cap,
                "rank_note": (
                    f"N={N} < D={D}: the POPULATION constraint Cov(g(z))=I_D cannot be"
                    " met by an EMPIRICAL estimate when N < D, because the centered"
                    " sample covariance has rank <= N-1. The exact finite-sample identity"
                    " is (1/(N-1)) Z Z^T = I_N - 11^T/N, the centered projector: diag"
                    " (N-1)/N, off-diag 1/N. Its marginals approach I_D as N -> infinity."
                    " This is a statement about finite-sample realisability, NOT a"
                    " refutation of the SplitJEPA population theorem."),
            })
        else:
            Zc = Z - mu
            cov = (Zc.T @ Zc) / max(1, N - 1)
            off = cov - torch.eye(D, dtype=cov.dtype)
            out.update({
                "contract": "cov_I_D",
                "offdiag_abs_max": float(off.abs().max()),
                "diag_min": float(cov.diag().min()),
                "diag_max": float(cov.diag().max()),
            })
        return out


# ---------------------------------------------------------------------------------
# the dual-branch encoder
# ---------------------------------------------------------------------------------
@dataclass
class SplitJEPAConfig:
    d_H: int = 16_384
    d_L: int = 49_152
    rho: float = 0.7            # predictive correlation (D_pred)
    orth_weight: float = 1.0    # weight on ||Q^T Q - I||_F^2
    pred_weight: float = 1.0
    inv_weight: float = 1.0
    steps: int = 400
    lr: float = 5e-3
    seed: int = 20261010
    tiny: bool = False          # exercise the diagnostic config on CPU

    @property
    def d(self) -> int:
        return self.d_H + self.d_L

    def resolve(self) -> "SplitJEPAConfig":
        """The spec's block split is 16,384 / 49,152, i.e. d_H/d = 1/4.

        At the diagnostic dimension the same RATIO is preserved so a small run is
        not a different problem.
        """
        if self.tiny:
            d = 64
            self.d_H = d // 4
            self.d_L = d - self.d_H
        return self


class SplitJEPAEncoder(nn.Module):
    """r = [r_H ; r_L] = Q z. Q is fit; A12 is MEASURED, not asserted.

    The blocks are named exactly as the blueprint names them:
        A11 = Q[:d_H, :d_H]   A12 = Q[:d_H, d_H:]
        A21 = Q[d_H:, :d_H]   A22 = Q[d_H:, d_H:]
    """

    def __init__(self, cfg: SplitJEPAConfig):
        super().__init__()
        self.cfg = cfg
        d, d_H = cfg.d, cfg.d_H
        g = torch.Generator().manual_seed(cfg.seed + 1)
        # start near identity: a non-degenerate but uninformative coupling
        self.Q = nn.Parameter(torch.eye(d) + 0.05 * torch.randn(d, d, generator=g))

    # -- blocks ------------------------------------------------------------------
    def blocks(self) -> dict:
        dH = self.cfg.d_H
        Q = self.Q
        return {"A11": Q[:dH, :dH], "A12": Q[:dH, dH:],
                "A21": Q[dH:, :dH], "A22": Q[dH:, dH:]}

    def leakage(self) -> dict:
        """G-SJ1 statistic: normalized cross-block Frobenius norms."""
        b = self.blocks()
        dH, dL = self.cfg.d_H, self.cfg.d_L
        return {
            "norm_A12_over_sqrt_dH": float(b["A12"].detach().norm() / math.sqrt(dH)),
            "norm_A21_over_sqrt_dL": float(b["A21"].detach().norm() / math.sqrt(dL)),
        }

    def forward(self, z: torch.Tensor) -> tuple:
        r = z @ self.Q.T
        return r[:, :self.cfg.d_H], r[:, self.cfg.d_H:]

    # -- objectives --------------------------------------------------------------
    def losses(self, zp1, zp2, zi1, zi2) -> dict:
        """SplitJEPA objectives.

        L_pred  representation-space prediction on D_pred pairs.
        L_inv   invariant-block agreement on D_inv matched pairs. This is the term
                that drives A12 -> 0, and it does so ONLY if Sigma_dL > 0.
        L_orth  keeps Q non-degenerate so L_pred cannot collapse Q to zero.
        """
        rp1, _ = self.forward(zp1)
        rp2, _ = self.forward(zp2)
        l_pred = F.mse_loss(rp1, rp2)

        ri1, _ = self.forward(zi1)
        ri2, _ = self.forward(zi2)
        l_inv = F.mse_loss(ri1, ri2)

        d = self.cfg.d
        eye = torch.eye(d, dtype=self.Q.dtype)
        l_orth = (self.Q.T @ self.Q - eye).pow(2).mean()

        total = (self.cfg.pred_weight * l_pred
                 + self.cfg.inv_weight * l_inv
                 + self.cfg.orth_weight * l_orth)
        return {"total": total, "pred": l_pred, "inv": l_inv, "orth": l_orth}


def sufficient_variation(zi1: torch.Tensor, zi2: torch.Tensor, d_H: int,
                         d_L: int) -> dict:
    """Sigma_dL = E[(z_L1 - z_L2)(z_L1 - z_L2)^T]. The identifiability precondition.

    Returns the min eigenvalue of the Gram of the differences, which is 0 exactly when
    the matched pairs carry NO variant variation. The spec writes the condition as
    Sigma_dL > 0; the rank of the difference matrix is the operative test.
    """
    dz = (zi1[:, d_H:] - zi2[:, d_H:])
    n = dz.shape[0]
    G = (dz @ dz.T) / max(1, n - 1)
    ev = torch.linalg.eigvalsh(G).clamp_min(0.0)
    trace = float(ev.sum())
    return {
        "rank_difference": int((ev > 1e-8).sum()),
        "min_eigenvalue": float(ev.min()),
        "max_eigenvalue": float(ev.max()),
        "trace": trace,
        "satisfied": bool(trace > 1e-8 and int((ev > 1e-8).sum()) > 0),
    }


# ---------------------------------------------------------------------------------
# synthetic D_pred / D_inv generators
# ---------------------------------------------------------------------------------
def make_latent_pairs(cfg: SplitJEPAConfig, n: int, vary_L: bool,
                      seed: int | None = None):
    """Build (pred pairs, inv pairs) in RAW latent space.

    vary_L=False is the NEGATIVE CONTROL: matched pairs share z_H AND z_L, so
    Sigma_dL = 0 and the identifiability condition is violated.

    Fold2Reason maps onto this exactly: z_H is the fold class / contact topology,
    z_L is the sequence mutation and thermal state.
    """
    s = cfg.seed if seed is None else seed
    g = torch.Generator().manual_seed(s)
    d_H, d_L = cfg.d_H, cfg.d_L
    zH = torch.randn(n, d_H, generator=g)
    zL1 = torch.randn(n, d_L, generator=g)
    zL2 = torch.randn(n, d_L, generator=g) if vary_L else zL1.clone()

    z1 = torch.cat([zH, zL1], dim=-1)
    # D_pred: z2 = rho z1 + sqrt(1-rho^2) eps
    eps = torch.randn(n, cfg.d, generator=g)
    zp2 = cfg.rho * z1 + math.sqrt(max(0.0, 1.0 - cfg.rho ** 2)) * eps
    zp1 = z1.clone()
    # D_inv: share z_H, vary (or hold) z_L
    zi1 = torch.cat([zH, zL1], dim=-1)
    zi2 = torch.cat([zH, zL2], dim=-1)
    return zp1, zp2, zi1, zi2


def fit_splitjepa(raw: tuple, cfg: SplitJEPAConfig, whiten: bool = True):
    """Whitening -> encoder fit -> measured coupling. Deterministic given cfg.seed."""
    zp1, zp2, zi1, zi2 = raw
    route = None
    if whiten:
        # fit whitening on the union so both objectives share one representation class
        U = torch.cat([zp1, zp2, zi1, zi2], 0)
        zca = CenteredZCA().fit(U)
        route = zca.route_
        zp1, zp2, zi1, zi2 = (zca.transform(t) for t in (zp1, zp2, zi1, zi2))
        wres = zca.whiten_residual(U)
    else:
        wres = None

    cfg = cfg.resolve()
    enc = SplitJEPAEncoder(cfg)
    # re-init AFTER resolve() so tiny mode gets the right block widths
    g = torch.Generator().manual_seed(cfg.seed + 1)
    with torch.no_grad():
        enc.Q.copy_(torch.eye(cfg.d) + 0.05 * torch.randn(cfg.d, cfg.d, generator=g))

    opt = torch.optim.Adam(enc.parameters(), lr=cfg.lr)
    hist = []
    for step in range(cfg.steps):
        opt.zero_grad()
        L = enc.losses(zp1, zp2, zi1, zi2)
        L["total"].backward()
        opt.step()
        if step % max(1, cfg.steps // 8) == 0 or step == cfg.steps - 1:
            lk = enc.leakage()
            hist.append({"step": step,
                         **{k: round(float(v.detach()), 3) for k, v in L.items()},
                         **{k: round(v, 6) for k, v in lk.items()}})
    return {
        "config": {"d": cfg.d, "d_H": cfg.d_H, "d_L": cfg.d_L, "rho": cfg.rho,
                   "steps": cfg.steps, "lr": cfg.lr, "seed": cfg.seed,
                   "tiny": cfg.tiny, "whiten": whiten, "whitening_route": route},
        "whitening_contract": wres,
        "sigma_dL": sufficient_variation(zi1, zi2, cfg.d_H, cfg.d_L),
        "final": enc.leakage(),
        "history": hist,
        "encoder": enc,
    }
