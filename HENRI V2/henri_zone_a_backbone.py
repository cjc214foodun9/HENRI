"""henri_zone_a_backbone.py -- novel trainable Zone A backbone (default-OFF, additive).

Design law (sealed measurement, commit 5b9b5cd):

    K = diag(m) + A S B^H        m = exp(i*theta)  in C^D     (full-rank elementwise)
                                 A, B in C^{D x q}, S = diag(s), s in R^q

    Low-PARAMETER is NOT low-RANK.  diag(m) is full rank but costs O(D) parameters.
    A rank-r-only kernel CANNOT express elementwise transforms: measured at D=2048,
    FACTORIZED r=64 held-out corr 0.0583 ~ shuffled 0.0515 (margin +0.0067), while
    DIAGONAL full-rank reached 0.7521 vs shuffled 0.1438 (margin +0.6083), confirmed
    across seeds (+0.7103 / +0.6119 / +0.6864).  A rank-r-only backbone reproduces
    the memorisation failure.  The diagonal term is therefore mandatory.

Components
    ZoneATransitionOperator   trainable kernel; complex; full-rank diagonal core.
    PCALMInferenceState       layer-local dual-state credit (arXiv:2605.31022).
    PreSnapCovarianceProbe    pre-snap spectrum as an OPTIONAL confidence
                              channel.  It never materialises a [D, D] tensor:
                              at D=65536 that is 32 GiB and OOMs a 32 GB GPU.
                              H3/H3b FALSIFIED the claim that snapped tokens
                              are causally blind, so this is not a required
                              readout (see the class docstring).
    ZoneABackbone            composes operator + probe behind one default-OFF gate.

Hard invariants
    - Default-OFF: unless HENRI_ZONE_A_BACKBONE=1, constructors raise.  No silent
      fallback path.
    - Additive: this module never imports or replaces the CLASS51 adapter path.
    - Determinism: all PRNG derives from an immutable RunManifest seed.
    - No dense [D, D] allocation; the operator is never materialised.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

ENV_ENABLE_FLAG = "HENRI_ZONE_A_BACKBONE"

try:  # pragma: no cover - import surface
    from henri.determinism import derive_seed as _derive_seed
except Exception:  # pragma: no cover
    _derive_seed = None


def zone_a_backbone_enabled() -> bool:
    """True only when the explicit opt-in flag is set."""
    return os.environ.get(ENV_ENABLE_FLAG, "").strip() in {"1", "true", "True", "yes"}


class ZoneABackboneError(RuntimeError):
    """Base class for Zone A backbone failures."""


class ZoneABackboneDisabledError(ZoneABackboneError):
    """Raised when the backbone is used with HENRI_ZONE_A_BACKBONE unset or 0."""


def _to_complex(t: torch.Tensor) -> torch.Tensor:
    """Promote a real tensor to complex64 without changing its value."""
    if t.is_complex():
        return t
    return torch.complex(t, torch.zeros_like(t))


def _gram_inv_sqrt(m: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Gram^{-1/2} for complex m [D, r].

    torch.linalg.qr ignores mode='reduced' on complex [D, r] inputs and returns a
    [D, D] factor; Gram^{-1/2} is the projection that is actually correct here.
    """
    gram = m.conj().transpose(-2, -1) @ m
    gram = 0.5 * (gram + gram.conj().transpose(-2, -1))
    evals, evecs = torch.linalg.eigh(gram)
    evals = torch.clamp(evals.real, min=eps)
    inv_sqrt = evecs @ torch.diag(evals.pow(-0.5).to(evecs.dtype)) @ evecs.conj().transpose(-2, -1)
    return inv_sqrt


def stiefel_project(m: torch.Tensor) -> torch.Tensor:
    """Project complex m [D, r] onto the complex Stiefel manifold (orthonormal cols)."""
    return m @ _gram_inv_sqrt(m)


class ZoneATransitionOperator(nn.Module):
    """Trainable transition operator K = diag(m) + A S B^H, m = exp(i*theta).

    Full-rank elementwise core plus rank-q cross-channel mixing.  Parameter count
    is O(D q), never O(D^2).
    """

    def __init__(
        self,
        dim: int,
        mixing_rank: int = 1,
        *,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        init_mixing_scale: float = 0.02,
        learn_phase: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ZoneABackboneError(f"dim must be positive; got {dim}")
        if mixing_rank < 0:
            raise ZoneABackboneError(f"mixing_rank must be >= 0; got {mixing_rank}")
        self.dim = int(dim)
        self.mixing_rank = int(mixing_rank)
        gen = generator
        if gen is None:
            gen = torch.Generator(device="cpu")
            if seed is None:
                if _derive_seed is not None:
                    seed = _derive_seed(20261001, "zone_a_backbone/operator")
                else:
                    seed = 20261001
            gen.manual_seed(int(seed) % (2**31))
        # Elementwise phase: m = exp(i * theta).  |m| = 1 => diag(m) is invertible,
        # hence full rank, hence able to express elementwise transforms.
        theta0 = torch.zeros(self.dim, dtype=torch.float32)
        self.theta = nn.Parameter(theta0) if learn_phase else theta0
        q = self.mixing_rank
        if q > 0:
            a = torch.randn(self.dim, q, generator=gen, dtype=torch.float32) * init_mixing_scale
            b = torch.randn(self.dim, q, generator=gen, dtype=torch.float32) * init_mixing_scale
            self.A = nn.Parameter(torch.complex(a, torch.zeros_like(a)))
            self.B = nn.Parameter(torch.complex(b, torch.zeros_like(b)))
            self.s = nn.Parameter(torch.ones(q, dtype=torch.float32))
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)
            self.register_parameter("s", None)

    # -- geometry ---------------------------------------------------------
    def diagonal(self) -> torch.Tensor:
        """m = exp(i*theta) in C^D; |m| = 1 exactly."""
        theta = self.theta if isinstance(self.theta, torch.Tensor) else self.theta
        return torch.exp(torch.complex(torch.zeros_like(theta), theta))

    def param_coordinates(self) -> int:
        """Number of real trainable coordinates (the leanness metric)."""
        n = self.dim  # one real coordinate per phase
        if self.mixing_rank > 0:
            n += 4 * self.dim * self.mixing_rank  # A, B complex => 2 reals each
            n += self.mixing_rank
        return int(n)

    def footprint_report(self) -> Dict[str, Any]:
        coords = self.param_coordinates()
        return {
            "dim": self.dim,
            "mixing_rank": self.mixing_rank,
            "trainable_float_coordinates": coords,
            "dense_equivalent_bytes_fp32": 2 * self.dim * self.dim * 4,
            "footprint_bytes_fp32": coords * 4,
            "footprint_bytes_fp64": coords * 8,
        }

    # -- forward ----------------------------------------------------------
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply K to x (any leading batch dims) without materialising [D, D]."""
        if x.shape[-1] != self.dim:
            raise ZoneABackboneError(f"expected last dim {self.dim}; got {tuple(x.shape)}")
        xc = _to_complex(x)
        m = self.diagonal().to(xc.dtype)
        y = m * xc
        if self.mixing_rank > 0:
            a = self.A.to(xc.dtype)
            b = self.B.to(xc.dtype)
            s = self.s.to(xc.real.dtype)
            proj = xc @ b.conj()            # [..., q] == B^H x
            y = y + (proj * s) @ a.transpose(-2, -1)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply(x)

    @torch.no_grad()
    def dense_equivalent(self, max_dim: int = 512) -> torch.Tensor:
        """Materialise [D, D] for small D only (verification, never production)."""
        if self.dim > max_dim:
            raise ZoneABackboneError(
                f"refusing to materialise {self.dim}x{self.dim}; cap {max_dim}"
            )
        eye = torch.eye(self.dim, dtype=torch.complex64)
        return self.apply(eye).transpose(0, 1)

    @torch.no_grad()
    def numerical_rank(self, max_dim: int = 512, tol: float = 1e-6) -> int:
        k = self.dense_equivalent(max_dim=max_dim)
        sv = torch.linalg.svdvals(k)
        return int((sv > tol * sv[0]).sum().item())


@dataclass
class PCALMReport:
    layer: int
    residual_norm: float
    dual_norm: float
    credit_norm: float


class PCALMInferenceState:
    """Layer-local dual-state inference (arXiv:2605.31022, PC-ALM).

    Constrained problem:  h_i = W_i h_{i-1}.  Augmented Lagrangian

        F = 0.5||y - W_L h_{L-1}||^2 + sum_i [ lambda_i^T r_i + (rho/2)||r_i||^2 ]
        r_i = h_i - W_i h_{i-1},   composite credit  e_i = lambda_i + rho * r_i.

    Inference is layer-local: grad_{h_j} = e_j - W_{j+1}^T e_{j+1}, and the output
    layer adds -W_L^T (y - W_L h_{L-1}).  Dual ascent  lambda_i += rho * r_i.
    At the KKT point r_i -> 0, e_i -> lambda_i, and lambda_i equals the BP adjoint.

    Dual state is PER-SAMPLE INFERENCE STATE, never a persisted parameter.
    """

    def __init__(
        self,
        weights: Sequence[torch.Tensor],
        *,
        rho: float = 1.0,
        eta_h: float = 0.25,
        steps: Optional[int] = None,
        alpha: float = 1.0,
        use_dual: bool = True,
    ) -> None:
        if len(weights) < 1:
            raise ZoneABackboneError("need at least one weight matrix")
        self.W = [w for w in weights]
        self.L = len(self.W)
        self.rho = float(rho)
        self.eta_h = float(eta_h)
        self.alpha = float(alpha)
        self.steps = int(steps) if steps is not None else 2 * self.L
        self.use_dual = bool(use_dual)

    def _forward_init(self, x: torch.Tensor) -> List[torch.Tensor]:
        h = [x]
        for w in self.W[:-1]:
            h.append(h[-1] @ w.transpose(0, 1))
        return h

    def _residuals(self, h: List[torch.Tensor]) -> List[torch.Tensor]:
        """r_i = h_{i-1} W_{i-1}^T - h_i, for i = 1..L-1.  ZERO at the forward pass.

        Sign convention: prediction minus state.  With this choice the KKT
        multipliers converge to +dL/dh_i (the BP adjoint), matching the source
        claim in arXiv:2605.31022.  Using h_i - W_i h_{i-1} instead flips the
        sign of lambda; the sign is a convention, the magnitude is the adjoint.
        """
        r = []
        for i in range(len(h) - 1):
            pred = h[i] @ self.W[i].transpose(0, 1)
            r.append(pred - h[i + 1])
        return r

    def _energy(self, h: List[torch.Tensor], y: torch.Tensor, lam: List[torch.Tensor]) -> torch.Tensor:
        out = h[-1] @ self.W[-1].transpose(0, 1)
        F = 0.5 * ((y - out) ** 2).sum()
        r = self._residuals(h)
        for i, ri in enumerate(r):
            F = F + (lam[i] * ri).sum() + 0.5 * self.rho * (ri ** 2).sum()
        return F

    def _local_grads(
        self, h: List[torch.Tensor], y: torch.Tensor, lam: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Analytic layer-local gradients + composite credits e_i = lambda_i + rho*r_i.

            dF/dh_j       = -e_{j-1} + e_j @ W_j          (1 <= j <= L-2)
            dF/dh_{L-1}   = -e_{L-2} - (y - out) @ W_{L-1}

        The supervised loss depends explicitly only on h_{L-1}; the chain
        dependence on earlier layers enters through the constraint terms.  This
        is why inference propagates a credit wavefront instead of a global
        backward sweep.  Verified against autograd of `_energy` in the tests.
        """
        r = self._residuals(h)
        e = [lam[i] + self.rho * r[i] for i in range(len(r))]
        grads = [None] * len(h)
        for j in range(1, len(h) - 1):
            grads[j] = -e[j - 1] + e[j] @ self.W[j]
        out = h[-1] @ self.W[-1].transpose(0, 1)
        grads[len(h) - 1] = -e[-1] - (y - out) @ self.W[-1]
        return grads, e

    def run(
        self, x: torch.Tensor, y: torch.Tensor, *, record: bool = False
    ) -> Dict[str, Any]:
        """Run T inner inference steps; return final state and diagnostics."""
        h = self._forward_init(x)
        lam = [torch.zeros_like(h[i]) for i in range(1, self.L)]
        history: List[PCALMReport] = []
        for t in range(self.steps):
            grads, e = self._local_grads(h, y, lam)
            for j in range(1, self.L):
                h[j] = h[j] - self.eta_h * grads[j]
            if self.use_dual:
                r = self._residuals(h)
                for i in range(len(lam)):
                    lam[i] = lam[i] + self.rho * r[i]
            if record and (t % max(1, self.steps // 8) == 0 or t == self.steps - 1):
                r = self._residuals(h)
                history.append(
                    PCALMReport(
                        layer=t,
                        residual_norm=float(torch.sqrt(sum((ri ** 2).sum() for ri in r))),
                        dual_norm=float(torch.sqrt(sum((li ** 2).sum() for li in lam))),
                        credit_norm=float(torch.sqrt(sum((ei ** 2).sum() for ei in e))),
                    )
                )
        r = self._residuals(h)
        _, e_final = self._local_grads(h, y, lam)
        out = h[-1] @ self.W[-1].transpose(0, 1)
        return {
            "h": h,
            "lambda": lam,
            "credits": e_final,
            "output": out,
            "residual": r,
            "history": history,
        }

    def bp_adjoint(self, x: torch.Tensor, y: torch.Tensor) -> List[torch.Tensor]:
        """Exact dL/dh_i on the forward pass (the reference signal for lambda).

        h_{j+1} = h_j W_j^T  =>  dL/dh_j = (dL/dh_{j+1}) W_j.
        """
        h = self._forward_init(x)
        out = h[-1] @ self.W[-1].transpose(0, 1)
        adj: List[Optional[torch.Tensor]] = [None] * len(h)
        adj[-1] = -(y - out) @ self.W[-1]          # dL/dh_{L-1}
        for j in range(len(h) - 2, 0, -1):
            adj[j] = adj[j + 1] @ self.W[j]        # noqa: E113 - indexed list
        return adj


class PreSnapCovarianceProbe:
    """Measure the residual stream BEFORE the snap.

    This probe tracks the pre-snap residual covariance spectrum so the consumer
    can read a confidence channel.  It does NOT claim the snap is blind: H3 and
    H3b FALSIFIED that claim at D=2048 (snap window 5 steps, covariance 6; no
    within-cell blind zone at eps <= 0.3).  Treat this as an optional extra
    channel, not a required readout.

    MEMORY CONTRACT (D=65536 blocker).  The naive form `cov = rc^H rc` is a
    [D, D] tensor: at D=65536 that is 32 GiB complex64, before a second [D, D]
    EMA accumulator and the eigensolver workspace.  It cannot run on a 32 GB
    GPU.  This probe never materialises a D x D tensor.
    """

    # Largest stacked residual budget, in rows.  Bounds the retained state at
    # max_rows * dim * 8 bytes (512 x 65536 x 8 B = 256 MiB at D=65536).
    MAX_ROWS = 512
    # Fail-closed ceiling on the stacked working set.
    MAX_BYTES = 1 << 30

    def __init__(
        self,
        dim: int,
        k: int = 4,
        ema: float = 0.5,
        max_rows: Optional[int] = None,
    ) -> None:
        if dim <= 0:
            raise ZoneABackboneError("dim must be positive")
        if not 0.0 < ema < 1.0:
            raise ZoneABackboneError("ema must lie strictly inside (0, 1)")
        self.dim = int(dim)
        self.k = int(k)
        self.ema = float(ema)
        self.max_rows = int(max_rows if max_rows is not None else self.MAX_ROWS)
        if self.max_rows < 2:
            raise ZoneABackboneError("max_rows must be >= 2")
        self._blocks: List[Tuple[torch.Tensor, int]] = []
        self._history: List[torch.Tensor] = []

    @torch.no_grad()
    def observe(self, residual: torch.Tensor) -> torch.Tensor:
        """Update the running covariance and return the top-k eigenvalue spectrum.

        Why this is exact and cheap.  The EMA covariance is a WEIGHTED SUM of
        per-call covariances `C_i = R_i^H R_i / (n_i - 1)`:

            Sigma_T = w_0 C_0 + sum_{i>=1} w_i C_i
            w_0 = ema**T ,  w_i = (1 - ema) * ema**(T - i)

        A weighted sum of Gram matrices is itself the Gram of a stacked,
        weighted residual matrix:

            Sigma_T = Rs^H Rs ,  Rs = concat_i [ sqrt(w_i / (n_i - 1)) * R_i ]

        The non-zero eigenvalues of `Rs^H Rs` equal the squared singular values
        of `Rs`.  So the spectrum is obtained EXACTLY from `svdvals` on
        `[Ntot, D]`, with memory O(Ntot * D) instead of O(D^2).

        Truncation is declared: only the last `max_rows` residual rows are kept.
        While every block is retained the result is exact, and it matches the
        old dense path (see test_probe_spectrum_matches_dense_path).
        """
        r = _to_complex(residual).reshape(-1, self.dim)
        if r.shape[0] < 2:
            raise ZoneABackboneError("need >= 2 samples to form a covariance")
        rc = r - r.mean(dim=0, keepdim=True)
        self._blocks.append((rc, int(rc.shape[0])))
        self._evict()
        spec = self._spectrum()
        self._history.append(spec.clone())
        return spec

    def _evict(self) -> None:
        """Drop the oldest blocks until the retained row count is bounded."""
        total = sum(n for _b, n in self._blocks)
        while len(self._blocks) > 1 and total > self.max_rows:
            _b, n = self._blocks.pop(0)
            total -= n

    def _weights(self) -> List[float]:
        """EMA weights for the retained blocks, oldest first.

        Exact reproduction of `Sigma <- ema*Sigma + (1-ema)*C` when no block has
        been evicted: the weights sum to 1 for every T.
        """
        t = len(self._blocks) - 1
        w = [self.ema ** t]
        for i in range(1, t + 1):
            w.append((1.0 - self.ema) * (self.ema ** (t - i)))
        return w

    def _spectrum(self) -> torch.Tensor:
        """Top-k eigenvalues of the EMA covariance, never touching D x D."""
        ntot = sum(n for _b, n in self._blocks)
        need = ntot * self.dim * 8
        if need > self.MAX_BYTES:
            raise ZoneABackboneError(
                f"stacked residual working set {need} B exceeds cap "
                f"{self.MAX_BYTES} B; lower max_rows or dim"
            )
        parts = []
        for (b, n), wi in zip(self._blocks, self._weights()):
            scale = (wi / float(n - 1)) ** 0.5
            if scale > 0.0:
                parts.append(b * scale)
        rs = torch.cat(parts, dim=0)                # [Ntot, D]
        sv = torch.linalg.svdvals(rs)               # descending, exact
        return (sv[: self.k] ** 2).real.contiguous()

    def state_shapes(self) -> List[Tuple[int, ...]]:
        """Shapes of every retained tensor.  Proves no [D, D] state exists."""
        return [tuple(b.shape) for b, _n in self._blocks]

    def subspace_shift(self) -> Optional[float]:
        """Spectral shift between the last two observations, or None if <2 seen."""
        if len(self._history) < 2:
            return None
        a, b = self._history[-2], self._history[-1]
        denom = float(a.norm()) + 1e-12
        return float((b - a).norm() / denom)


@dataclass
class ZoneABackbone:
    """Composes operator + pre-snap probe behind one default-OFF gate."""

    dim: int
    mixing_rank: int = 1
    seed: Optional[int] = None
    operator: Optional[ZoneATransitionOperator] = None
    probe: Optional[PreSnapCovarianceProbe] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not zone_a_backbone_enabled():
            raise ZoneABackboneDisabledError(
                f"{ENV_ENABLE_FLAG} is not set; Zone A backbone is disabled"
            )
        if self.operator is None:
            self.operator = ZoneATransitionOperator(
                dim=self.dim, mixing_rank=self.mixing_rank, seed=self.seed
            )
        if self.probe is None:
            self.probe = PreSnapCovarianceProbe(dim=self.dim)
        self.telemetry = {
            "dim": self.dim,
            "mixing_rank": self.mixing_rank,
            "enabled_flag": ENV_ENABLE_FLAG,
            **self.operator.footprint_report(),
        }

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """One transition through the trainable operator."""
        return self.operator.apply(x)
