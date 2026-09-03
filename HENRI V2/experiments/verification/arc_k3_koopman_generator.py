"""Carrier K3 — Empirical (block-local) Koopman transition generator (default-OFF).

Prereg: docs/spec/carrier_k3_empirical_koopman_preregistration.md (sealed 2026-09-03).
Supplied kernel staged byte-identical: carrier_k3_supplied_kernel.py
(SHA-256 bff0174955e5eea7d22be222c4a8056f2e04d02ad077de272776a7bf8ce66e4e).
Base: P1GoalSteeringEngine lineage @ 2f9bc57 (post-C1 closeout; C1 sealed
FALSIFIED — K3 is a NEW mechanism with its own gates, no tuning through C1).

Mechanism (spec eq. 1-4): per-action per-block Ridge Koopman operators from
LIVE causal transition pairs (Psi_t, a_t, Psi_{t+1}) captured at the G4
run-loop post-step hook:

    K_a = argmin_K ||Y_a - K X_a||_F^2 + alpha ||K||_F^2
factorized block-diagonally over M=8192 Clifford blocks of d=8:

    K_m = B_m (A_m + alpha I_8)^-1,  A_m = sum_i x_i x_i^T, B_m = sum_i y_i x_i^T

then the contractive projection: blocks whose spectral norm sigma_max > 1.0
are scaled by 1/sigma_max (exact sigma via screening + batched svdvals on the
suspicious set), so rho(K_m) <= 1.0 is enforced (KG4). The projection carries
engagement counters (fired-block count, raw sigma_max); a never-fired
projection is reported, never claimed as a pass.

Geometry: live planner waves are [num_blocks, 8] per-block unit rows; K3
operators act block-wise and are fit from per-action rings of fp16 rows
(cap 256) cleared at every environment boundary (dynamics are env-specific;
a cross-environment operator would be a causal leak).

Disclosed engineering adaptations (sealed prereg amendments):
1. The supplied Triton kernel accumulates only A/B covariances; the solve is
   a batched torch cholesky_solve in the reference class. The engine refits
   from ring tensors with equivalent einsum accumulations; the supplied
   kernel + class are exercised by the CUDA equivalence test and the
   tau_solve measurement (triton is imported lazily; this module never
   requires triton at import).
2. The supplied class's spectral scaling is unconditional. Here the scaling
   is exact for fired blocks (two-stage screening); fired counts and raw
   sigma_max are always reported.
3. Cond guard: the prereg's cond(XaXa^T + alpha I) > 1e5 -> alpha <- 2alpha
   guard is implemented as cholesky-failure escalation (alpha doubled up to
   4 times), then a counted pinv fallback. alpha starts at 1e-4.
4. Held-out KG1 rows (the newest W = clamp(N/4, 2, 8) rows of the ring) are
   never in the fit sums. Held-out error is evaluated once per (env, action)
   at the environment boundary, over the trailing rows that were excluded
   from every fit (causal: an operator never sees the rows it is scored
   against at evaluation time).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

K3_FLAG = "HENRI_K3_KOOPMAN"
K3_SEED = 20260930
K3_M = 8192            # Clifford blocks
K3_D = 8               # block dimension
K3_ALPHA = 1e-4
K3_RING_CAP = 256
KFIT_MIN_N = 8         # minimum fit rows before an operator is fitted
K3_EVAL_WINDOW_MAX = 8
K3_ALPHA_MAX_DOUBLINGS = 4
K3_POWER_ITERS = 16
K3_EXACT_TOP = 64
K3_HEADROOM = 1.05
K3_SCREEN_FLOOR = 0.95   # est below this (with headroom) is never scaled
K3_SCREEN_SEED = 20260903
K3_ABORT_DIR_ENV = "HENRI_K3_ABORT_DIR"


def require_k3_flag() -> None:
    """Fail closed unless the carrier flag is exactly '1'."""
    if os.environ.get(K3_FLAG, "0") != "1":
        raise RuntimeError(f"{K3_FLAG} is not set to '1'; Carrier K3 is default-OFF.")


class K3NumericalAbort(RuntimeError):
    """Fail-closed NaN/Inf abort (prereg §3.4). Ring state is serialized first."""


class K3RingAccumulator:
    """Causal per-action ring of (psi_t, psi_next) rows (fp16 storage)."""

    def __init__(self, cap: int, m_blocks: int, d_block: int,
                 device: torch.device, dtype=torch.float16):
        self.cap = int(cap)
        self.M = int(m_blocks)
        self.d = int(d_block)
        self.device = device
        self.dtype = dtype
        self.x = torch.zeros(cap, m_blocks, d_block, device=device, dtype=dtype)
        self.y = torch.zeros(cap, m_blocks, d_block, device=device, dtype=dtype)
        self.n = 0  # total rows pushed (causal order)

    def reset(self) -> None:
        self.n = 0

    def push(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Push one causal pair (pre-state, post-state). x/y: [M, d] or flat [M*d]."""
        xv = x.detach().reshape(self.M, self.d).to(self.device)
        yv = y.detach().reshape(self.M, self.d).to(self.device)
        if not torch.isfinite(xv).all() or not torch.isfinite(yv).all():
            raise K3NumericalAbort(
                "K3 ring push received non-finite wave (fail-closed).")
        pos = int(self.n % self.cap)
        self.x[pos].copy_(xv, non_blocking=True)
        self.y[pos].copy_(yv, non_blocking=True)
        self.n += 1

    @property
    def count(self) -> int:
        return min(self.n, self.cap)

    def ordered(self):
        """(X, Y) fp32 tensors [count, M, d] in arrival order."""
        c = self.count
        if c == 0:
            return (self.x.new_zeros(0, self.M, self.d, dtype=torch.float32),
                    self.y.new_zeros(0, self.M, self.d, dtype=torch.float32))
        start = int(self.n % self.cap) if self.n >= self.cap else 0
        idx = (torch.arange(start, start + c, device=self.device) % self.cap)
        return self.x[idx].float(), self.y[idx].float()

    def fit_eval_split(self):
        """(n_fit, w): oldest n_fit rows are fit rows; newest w are held out."""
        c = self.count
        w = int(min(K3_EVAL_WINDOW_MAX, max(2, c // 4)))
        n_fit = max(0, c - w)
        return n_fit, w


def _screen_sigma_max(K: torch.Tensor):
    """Return (sigma_est, exact_mask) for the [M, 8, 8] operator batch.

    sigma_est: per-block spectral-norm estimate. The top K3_EXACT_TOP
    estimated blocks (always including the argmax) carry EXACT sigma via
    batched svdvals (exact_mask True); the rest carry a deterministic
    power-iteration estimate (16 steps, two starts: ones and a seeded random
    vector) on K^T K.

    KG4 enforcement guarantee (sealed prereg amendment, disclosed): the
    estimate is asserted in the contract suite to satisfy
    est >= 0.95 * sigma_max on random and adversarially clustered batches
    (16 power steps on 8x8 real blocks). Under that assumption the caller's
    conservative scale (max(1, est/0.95) for estimated blocks, max(1, exact)
    for exact blocks) enforces post-scale sigma <= 1.0. Residual risk is
    documented, not silently enforced: a block whose true sigma > 1.0 while
    its estimate stays below 0.95*sigma would not be scaled.
    """
    M, d = K.shape[0], K.shape[1]
    if M == 0:
        return K.new_zeros(0), K.new_zeros(0, dtype=torch.bool)
    # K^T K via einsum over the row index: (K^T K)[i,k] = sum_j K[j,i] K[j,k].
    KtK = torch.einsum("mji,mjk->mik", K, K)   # [M, d, d]
    est = []
    for start_idx, base in enumerate((torch.ones(M, d, device=K.device, dtype=K.dtype),
                                      None)):
        if base is None:
            g = torch.Generator(device=K.device).manual_seed(K3_SCREEN_SEED + start_idx)
            base = torch.randn(M, d, device=K.device, dtype=K.dtype, generator=g)
        v = F.normalize(base, p=2, dim=-1)
        for _ in range(K3_POWER_ITERS):
            v = F.normalize(torch.einsum("mij,mj->mi", KtK, v), p=2, dim=-1)
        s2 = torch.einsum("mij,mj->mi", KtK, v)
        s = (v * s2).sum(-1).clamp(min=0.0).sqrt()  # Rayleigh sqrt
        est.append(s)
    est = torch.stack(est).max(dim=0).values
    exact_mask = torch.zeros(M, dtype=torch.bool, device=K.device)
    if M > K3_EXACT_TOP:
        top_idx = torch.argsort(est, descending=True)[:K3_EXACT_TOP]
    else:
        top_idx = torch.arange(M, device=K.device, dtype=torch.long)
    exact_mask[top_idx] = True
    exact = torch.linalg.svdvals(K[top_idx]).max(dim=-1).values
    est = est.clone()
    est[top_idx] = exact
    return est, exact_mask


class BlockRidgeKoopmanFit:
    """Block-local ridge Koopman solve + contractive projection."""

    def __init__(self, alpha: float = K3_ALPHA,
                 max_alpha_doublings: int = K3_ALPHA_MAX_DOUBLINGS):
        self.alpha = float(alpha)
        self.max_doublings = int(max_alpha_doublings)
        self.alpha_doublings = 0
        self.pinv_fallbacks = 0

    def fit(self, X: torch.Tensor, Y: torch.Tensor, fit_n: int) -> dict:
        """X, Y: [count, M, d] fp32 arrival-ordered rows; use the first fit_n.

        Returns {K [M,d,d], fired_blocks, sigma_max, alpha_doublings,
        pinv_fallback, n_fit}.
        """
        d = X.shape[-1]
        Xf = X[:fit_n]
        Yf = Y[:fit_n]
        A = torch.einsum("nmi,nmj->mij", Xf, Xf)   # [M, d, d]
        B = torch.einsum("nmi,nmj->mij", Yf, Xf)
        eye = torch.eye(d, device=A.device, dtype=A.dtype)
        alpha = self.alpha
        K = None
        self.alpha_doublings = 0
        for _ in range(1 + self.max_doublings):
            Aa = A + alpha * eye
            try:
                L = torch.linalg.cholesky(Aa)
                K = torch.cholesky_solve(B.transpose(-1, -2), L).transpose(-1, -2)
                break
            except torch.linalg.LinAlgError:
                alpha *= 2.0
                self.alpha_doublings += 1
        pinv = False
        if K is None:
            pinv = True
            self.pinv_fallbacks += 1
            K = torch.matmul(B, torch.linalg.pinv(A + alpha * eye))
        if not torch.isfinite(K).all():
            raise K3NumericalAbort("K3 operator fit produced non-finite K "
                                   "(fail-closed; ring state preserved).")
        # Contractive projection (conservative under the disclosed estimate
        # assumption: exact blocks scale by max(1, exact); estimated blocks
        # scale by max(1, est/0.95) so post sigma <= 1.0 when the 16-step
        # estimate is within 5% of the true sigma — asserted empirically in
        # the contract suite).
        sig, exact_mask = _screen_sigma_max(K)
        scale_est = (sig / K3_SCREEN_FLOOR).clamp(min=1.0)
        scale_exact = sig.clamp(min=1.0)
        scale = torch.where(exact_mask, scale_exact, scale_est)
        fired = int((scale > 1.0).sum().item())
        Kc = K / scale.unsqueeze(-1).unsqueeze(-1)
        # Exact post-scale spectral max over the FIRED set (the only set the
        # KG4 gate constrains). Unfired estimated blocks satisfy true sigma
        # <= 1.0 under the disclosed est >= 0.95*sigma assumption.
        fired_mask = scale > 1.0
        if bool(fired_mask.any().item()):
            post_max = float(torch.linalg.svdvals(
                Kc[fired_mask]).max(dim=-1).values.max().item())
        else:
            post_max = float(sig.max().item()) if sig.numel() else 0.0
        return {
            "K": Kc,
            "fired_blocks": fired,
            "sigma_max": float(sig.max().item()) if sig.numel() else 0.0,
            "sigma_post_max": post_max,
            "alpha_doublings": self.alpha_doublings,
            "pinv_fallback": pinv,
            "n_fit": int(fit_n),
        }

    @staticmethod
    def heldout_error(Xe: torch.Tensor, Ye: torch.Tensor,
                      K: torch.Tensor) -> float:
        """Relative one-step error on eval rows: ||Ye - K Xe||_F / ||Ye||_F."""
        pred = torch.einsum("mij,nmj->nmi", K, Xe)
        num = (Ye - pred).norm(p=2, dim=(-2, -1))
        den = Ye.norm(p=2, dim=(-2, -1)).clamp(min=1e-12)
        return float((num / den).mean().item())

    @staticmethod
    def apply(K: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """Block-wise K psi for psi [.., M, d]; returns same shape."""
        return torch.einsum("mij,...mj->...mi", K, psi)
