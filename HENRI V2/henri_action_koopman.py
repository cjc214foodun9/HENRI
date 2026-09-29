"""HENRI Action-Conditioned World Model — latent rollout (Gap 4).

WHY THIS MODULE EXISTS
======================
HENRI processes state snapshots as a STATIC mapping X -> Y. It holds no
action-conditioned transition operator, so it cannot answer "what happens if I do
a?" without executing. Measured consequence: no rollout means no tree search, no
counterfactual scoring, and no way to reject a candidate action before spending a
real environment step.

The missing object is a family of Koopman transition operators, one per action
primitive:

    Psi(t + dt) = K_a Psi(t)

WHAT IS AND IS NOT CLAIMED
==========================
* `K_a` is fitted by LEAST SQUARES on observed (state_t, a, state_{t+1}) triples in
  a GIVEN feature map. It is a linear operator in that feature space. This is not a
  claim that the environment is linear: a linear operator on an adequate feature
  map can represent nonlinear dynamics, and whether THIS feature map is adequate
  is an empirical question this module reports (rollout error), not asserts.
* An UNSEEN action ABSTAINS. Substituting an identity operator for an unknown
  action would make a rollout look successful while predicting nothing; the
  module raises instead.
* Default-OFF sidecar: nothing live imports it. It emits rollouts and scores; it
  does not select a live action.

CONTROLS (the four that caught real defects this session)
========================================================
  1. DEFAULT/OFF + IDENTITY: with `horizon=0` a rollout returns the input state.
  2. DIFFERENTIAL EFFECT: two different actions must produce different rollouts
     (guards a degenerate operator pool where every K_a fitted to the same thing).
  3. ACTION-SENSITIVITY: swapping the action sequence must change the trajectory.
  4. FAIL-CLOSED: unknown action -> raise; shape mismatch -> raise; a rollout
     whose norm is non-finite -> raise.

THE COUNTERFACTUAL SCORE
========================
`score_actions(state, candidates, horizon)` returns, per candidate, the predicted
landing state and its distance to a goal. That is the value a planner ranks by. The
module ALSO reports `action_spread` -- the maximum pairwise distance between the
candidate landings. A pool whose candidates all land in the same place cannot
discriminate, and reporting the spread makes that visible instead of silent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

LAM = 1e-4            # ridge regularisation for the least-squares fit

# CONTRACT A ENFORCEMENT (Directive 3). A dense operator costs dim^2 * 4 bytes:
#   dim=65,536 -> 17.18 GB per action, 137 GB for 8 -> instant OOM on 32 GB.
# Below this dimension a dense operator is affordable and useful (toy/test scale);
# above it, constructing one requires an EXPLICIT `allow_dense=True`, so a future
# caller cannot reach production scale by inertia. Fail-closed, not advisory.
DENSE_DIM_LIMIT = 16384


class WorldModelError(RuntimeError):
    """Fail-closed contract violation."""


@dataclass
class RolloutResult:
    states: List[torch.Tensor] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    horizon: int = 0


class ActionConditionedKoopman:
    """One linear transition operator per action, fitted in a feature map.

    fit(triples) where a triple is (state_t, action, state_t1), each a flat float
    tensor of width `dim`. The operator for action a is
        K_a = argmin || K S_a - S'_a ||_F^2 + lam ||K||^2
    solved in closed form from the normal equations. Production sizing uses the
    Gram matrix on the DIMENSION, never a [d, d] solve, so memory stays O(d * n)
    for n samples (the same discipline as the dual/thin-SVD EDMD contract).
    """

    def __init__(self, dim: int, n_actions: int, lam: float = LAM,
                 enforce_unit_norm: bool = True, rank: Optional[int] = None,
                 allow_dense: bool = False):
        """`rank=None` keeps the LEGACY dense operator (default, byte-identical).

        `rank=r` selects the LOW-RANK form K_a = U_a Vs_a^T with r <= min(r, n, dim),
        which never allocates a [dim, dim] matrix. Contract A, Directive 3.

        `allow_dense` is the ESCAPE HATCH for a deliberate, small dense operator.
        Above DENSE_DIM_LIMIT a dense construction FAILS CLOSED unless the caller
        asks for it explicitly and records why: at dim=65,536 the dense operator is
        17.18 GB per action, so a silent default would OOM a 32 GB device.
        """
        if dim < 2:
            raise WorldModelError("dim must be >= 2")
        if n_actions < 1:
            raise WorldModelError("n_actions must be >= 1")
        self.dim = int(dim)
        self.n_actions = int(n_actions)
        self.lam = float(lam)
        self.enforce_unit_norm = bool(enforce_unit_norm)
        if rank is not None:
            rank = int(rank)
            if rank < 1:
                raise WorldModelError("rank must be >= 1 when set")
        self.rank_requested = rank        # as asked; may exceed dim
        if rank is None and self.dim > DENSE_DIM_LIMIT and not allow_dense:
            raise WorldModelError(
                f"refusing to construct a dense [{self.dim},{self.dim}] operator: "
                f"that is {self.dim * self.dim * 4 / (1024 ** 3):.2f} GB per action "
                f"(Contract A: never form D^2). Pass rank=r with r <= 64 for the "
                f"low-rank form, or allow_dense=True to override deliberately.")
        if rank is not None and rank > self.dim:
            # Architecture contract: enforce EFFECTIVE rank min(r, d) before
            # allocation, rather than raising. The clamp is not silent: it is
            # reported via `effective_rank` / `operator_bytes()["rank_requested"]`,
            # so a caller can assert on it. (Raising here contradicted the
            # catalog rule and made a toy-scale rank A/B impossible.)
            rank = self.dim
        self.rank = rank                  # None => legacy dense path
        self.K: Dict[int, torch.Tensor] = {}   # dense path only
        self.U: Dict[int, torch.Tensor] = {}   # low-rank LEFT factor  [dim, r]
        self.Vs: Dict[int, torch.Tensor] = {}  # low-rank RIGHT factor [dim, r]
        self.effective_rank: Dict[int, int] = {}
        self.singular_values: Dict[int, torch.Tensor] = {}
        self.counts: Dict[int, int] = {}
        self.residual: Dict[int, float] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, triples: Sequence[Tuple[torch.Tensor, int, torch.Tensor]]):
        """Fit one operator per action. Actions with no samples are left UNFITTED
        so `roll` can abstain instead of inventing an identity."""
        if not triples:
            raise WorldModelError("fit needs >= 1 triple")
        per: Dict[int, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
        for s, a, s1 in triples:
            s, s1 = self._as_flat(s), self._as_flat(s1)
            if s.shape != s1.shape:
                raise WorldModelError("state and successor shapes differ")
            if int(a) < 0 or int(a) >= self.n_actions:
                raise WorldModelError(f"action {a} outside [0,{self.n_actions})")
            src, dst = per.setdefault(int(a), ([], []))
            src.append(s)
            dst.append(s1)

        self.K.clear()
        self.U.clear()
        self.Vs.clear()
        self.counts.clear()
        self.residual.clear()
        for a, (src, dst) in per.items():
            X = torch.stack(src).to(torch.float64)      # [n, d]
            Y = torch.stack(dst).to(torch.float64)      # [n, d]
            self.counts[a] = int(X.shape[0])
            if self.rank is None:
                # ---- LEGACY DENSE PATH (default, byte-identical) ----
                # Solve in dimension: K = Y^T X (X^T X + lam I)^-1. For tiny
                # sample counts the dual form is cheaper, so pick the smaller
                # system.
                n, d = X.shape
                if n >= d:
                    A = X.t() @ X + self.lam * torch.eye(d, dtype=torch.float64)
                    B = Y.t() @ X
                    K = torch.linalg.solve(A, B.t()).t()
                else:
                    A = X @ X.t() + self.lam * torch.eye(n, dtype=torch.float64)
                    alpha = torch.linalg.solve(A, Y)          # [n, d]
                    K = X.t() @ alpha                         # [d, d]
                if not torch.isfinite(K).all():
                    raise WorldModelError(f"action {a}: non-finite operator")
                pred = X @ K.t()
                self.residual[a] = float((pred - Y).norm() / (Y.norm() + 1e-12))
                self.K[a] = K.to(torch.float32)
                continue
            # ---- LOW-RANK PATH (Contract A, Directive 3) ----
            # K = alpha^T X,  alpha = (X X^T + lam I)^-1 Y   [n, d].
            # Truncated SVD of K WITHOUT ever forming [d, d]:
            #     G2 = X X^T = Q2 diag(w) Q2^T   [n, n]  (eigh, ascending)
            #     X  = Q2 S V^T                  => S V^T = Q2^T X
            #     K  = alpha^T Q2 (Q2^T X)
            # so with Q_top = Q2[:, top-r]:   U = alpha^T Q_top,  Vs = X^T Q_top
            # and  K_r = U Vs^T = alpha^T Q_top Q_top^T X  (rank-r projection).
            # NOTE: NO division by w or sqrt(w). An earlier form of this code
            # divided the factors by sqrt(w) / w, which is NOT the truncated
            # operator; it was caught by re-deriving S V^T = Q2^T X before running.
            # Rank is bounded by min(r, n, d) -- reported, never assumed.
            n, d = X.shape
            A = X @ X.t() + self.lam * torch.eye(n, dtype=torch.float64)
            alpha = torch.linalg.solve(A, Y)                 # [n, d]
            G2 = X @ X.t()                                   # [n, n]
            w, Q2 = torch.linalg.eigh(G2)                    # ascending
            r_eff = int(min(self.rank, n, d))
            sl = slice(n - r_eff, n)                         # top r_eff
            Q_top = Q2[:, sl]                                # [n, r_eff]
            U = alpha.t() @ Q_top                            # [d, r_eff]
            Vs = X.t() @ Q_top                               # [d, r_eff]
            if not (torch.isfinite(U).all() and torch.isfinite(Vs).all()):
                raise WorldModelError(f"action {a}: non-finite low-rank factors")
            pred = (X @ Vs) @ U.t()
            self.residual[a] = float((pred - Y).norm() / (Y.norm() + 1e-12))
            self.U[a] = U.to(torch.float32)
            self.Vs[a] = Vs.to(torch.float32)
            self.effective_rank[a] = r_eff
            self.singular_values[a] = w[sl].flip(0).sqrt().to(torch.float64)
        return self

    # --------------------------------------------------------------- rollout
    def _apply_dense(self, s: torch.Tensor, a: int) -> torch.Tensor:
        return s.to(torch.float32) @ self.K[a].t()

    def _apply_lowrank(self, s: torch.Tensor, a: int) -> torch.Tensor:
        """s K_a^T with K_a = U Vs^T, i.e. (s Vs) U^T.

        CONTRACT WITH THE DENSE PATH: `_apply_dense` computes `s @ K.T`. Since
        K = U Vs^T => K^T = Vs U^T, the low-rank path MUST be `(s @ Vs) @ U.T`.

        DEFECT FIXED 2026-09-28: this method first read `(s @ U) @ Vs.T`, which
        is `s @ K`, NOT `s @ K.T`. The two agree only when K is SYMMETRIC. The
        rotation-based synthetic truth is not symmetric, so the equivalence test
        measured 1.4641 (low-rank) vs 0.0164 (dense) at rank == dim, where the
        operator itself matched to 0.0. The bug was in the APPLY, not the fit.
        Memory stays O(d * r); the [d, d] operator is never formed.
        """
        return (s.to(torch.float32) @ self.Vs[a]) @ self.U[a].t()

    def step(self, state: torch.Tensor, action: int) -> torch.Tensor:
        a = int(action)
        fitted = (a in self.K) if self.rank is None else (a in self.U)
        if not fitted:
            raise WorldModelError(
                f"ABSTAIN: no operator fitted for action {a}; refusing to substitute "
                f"an identity (that would predict nothing while looking successful)")
        s = self._as_flat(state)
        out = self._apply_dense(s, a) if self.rank is None else self._apply_lowrank(s, a)
        if not torch.isfinite(out).all():
            raise WorldModelError("rollout produced non-finite state")
        if self.enforce_unit_norm:
            out = out / (torch.linalg.vector_norm(out) + 1e-8)
        return out

    def roll(self, state: torch.Tensor, actions: Sequence[int]) -> RolloutResult:
        r = RolloutResult(horizon=len(actions))
        if len(actions) == 0:
            r.states = [self._as_flat(state).to(torch.float32)]
            return r
        s = state
        for a in actions:
            s = self.step(s, a)
            r.states.append(s)
            r.actions.append(int(a))
        return r

    # ------------------------------------------------------- counterfactual
    def score_actions(self, state: torch.Tensor, candidates: Sequence[int],
                      goal: Optional[torch.Tensor] = None,
                      horizon: int = 1) -> dict:
        """Rank candidate action SEQUENCES by predicted goal distance.

        Returns landings, distances, the ranking, and `action_spread` -- the max
        pairwise distance between candidate landings. A pool whose candidates land
        in the same place cannot discriminate; the spread makes that explicit.
        """
        if not candidates:
            raise WorldModelError("score_actions needs >= 1 candidate")
        if horizon < 1:
            raise WorldModelError("horizon must be >= 1")
        g = None if goal is None else self._as_flat(goal).to(torch.float32)
        landings: Dict[int, torch.Tensor] = {}
        dists: Dict[int, float] = {}
        for a in candidates:
            seq = [int(a)] * horizon
            land = self.roll(state, seq).states[-1]
            landings[int(a)] = land
            dists[int(a)] = (float(torch.linalg.vector_norm(land - g)) if g is not None
                             else 0.0)
        ranked = sorted(candidates, key=lambda a: dists[int(a)])
        vals = list(landings.values())
        spread = 0.0
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                spread = max(spread, float(torch.linalg.vector_norm(vals[i] - vals[j])))
        return {
            "landings": {int(k): v for k, v in landings.items()},
            "distances": {int(k): v for k, v in dists.items()},
            "ranking": [int(a) for a in ranked],
            "action_spread": spread,
            "discriminating": bool(spread > 1e-6),
            "fitted_actions": sorted(self.K),
            "residual": dict(self.residual),
        }

    # -------------------------------------------------------------- helpers
    def _fitted(self):
        """Sorted fitted action ids for whichever representation is active."""
        return sorted(self.K if self.rank is None else self.U)

    def operator_bytes(self) -> dict:
        """Memory accounting for the ACTIVE representation.

        Cheap insurance for a logit tensor: separates the two cost models so a
        low-rank run cannot silently be a dense run. The dense model costs
        O(dim^2) per action; the low-rank model costs O(dim * r).
        """
        if self.rank is None:
            per_action = self.dim * self.dim * 4      # float32
            return {
                "representation": "dense_legacy",
                "bytes_per_fitted_action": per_action,
                "total_bytes": per_action * len(self.K),
                "dense_equivalent_bytes": per_action * len(self.K),
                "dense_matrix_formed": True,
            }
        n = 0
        dense_eq = 0
        for a in self.U:
            r = self.U[a].shape[1]
            n += self.dim * r * 4 * 2                 # U and Vs
            dense_eq += self.dim * self.dim * 4
        return {
            "representation": "low_rank",
            "rank_requested": self.rank_requested,
            "rank_used": self.rank,
            "rank_was_clamped": (self.rank_requested is not None
                                 and self.rank_requested != self.rank),
            "bytes_per_fitted_action": (self.dim * self.rank * 4 * 2
                                        if self.U else 0),
            "total_bytes": n,
            "dense_equivalent_bytes": dense_eq,
            "dense_matrix_formed": False,
            "savings_factor": (dense_eq / n) if n else None,
            "effective_rank": dict(self.effective_rank),
        }

    def _as_flat(self, s) -> torch.Tensor:
        if isinstance(s, torch.Tensor):
            t = s
        else:
            t = torch.as_tensor(s, dtype=torch.float32)
        if t.is_complex():
            t = t.real
        t = t.reshape(-1).to(torch.float32)
        if t.numel() != self.dim:
            raise WorldModelError(f"state width {t.numel()} != dim {self.dim}")
        if not torch.isfinite(t).all():
            raise WorldModelError("non-finite state")
        return t

    def report(self) -> dict:
        fitted = self._fitted()
        return {
            "schema": "henri.world-model.action-koopman.report.v1",
            "dim": self.dim, "n_actions": self.n_actions, "lam": self.lam,
            "fitted_actions": fitted,
            "abstaining_actions": sorted(set(range(self.n_actions)) - set(fitted)),
            "samples_per_action": dict(self.counts),
            "relative_residual": dict(self.residual),
            "enforce_unit_norm": self.enforce_unit_norm,
            "memory": self.operator_bytes(),
            "honest_limit": ("Linear operator in a GIVEN feature map. Not a claim that "
                             "the environment is linear; adequacy is reported as "
                             "rollout residual, not asserted. Unseen actions ABSTAIN."),
        }


def make_synthetic_triples(n_actions: int, dim: int, n_per_action: int, seed: int,
                           rng_scale: float = 0.05):
    """Synthetic triples from a KNOWN ground-truth operator per action.

    Each action gets its own invertible rotation-like operator, so a correct fit
    must recover distinct K_a and a rollout must match the truth closely. This is
    the scaffold that validates the machinery before any real trajectory data.
    """
    g = torch.Generator().manual_seed(seed)
    truth: Dict[int, torch.Tensor] = {}
    for a in range(n_actions):
        th = (a + 1) * 0.37
        R = torch.eye(dim)
        # block-diagonal 2x2 rotations -> orthogonal and deterministic
        for i in range(0, dim - 1, 2):
            R[i, i] = math.cos(th)
            R[i, i + 1] = -math.sin(th)
            R[i + 1, i] = math.sin(th)
            R[i + 1, i + 1] = math.cos(th)
        truth[a] = R
    triples = []
    for a in range(n_actions):
        for _ in range(n_per_action):
            s = torch.randn(dim, generator=g)
            s = s / s.norm()
            # RELATIVE noise. DEFECT FIXED 2026-09-27: the first form added
            # rng_scale*randn(dim), whose norm is rng_scale*sqrt(dim) -- for
            # rng_scale=0.05 and dim=32 that is 0.28 against a UNIT signal, i.e.
            # 28% noise, so a correct least-squares fit still showed 0.25 rollout
            # error and the scaffold looked broken. Scaling by 1/sqrt(dim) makes
            # `rng_scale` the intended signal-relative noise level.
            noise = rng_scale * torch.randn(dim, generator=g) / math.sqrt(dim)
            s1 = truth[a] @ s + noise
            triples.append((s, a, s1))
    return triples, truth


def rollout_error(model: ActionConditionedKoopman, truth: Dict[int, torch.Tensor],
                  dim: int, horizon: int, seed: int = 0) -> float:
    """Mean relative error of an h-step rollout against the ground-truth operator."""
    g = torch.Generator().manual_seed(seed)
    errs = []
    for a in sorted(truth):
        s = torch.randn(dim, generator=g)
        s = s / s.norm()
        ref = s
        for _ in range(horizon):
            ref = truth[a] @ ref
            ref = ref / (ref.norm() + 1e-12)
        got = model.roll(s, [a] * horizon).states[-1]
        errs.append(float(torch.linalg.vector_norm(got - ref)))
    return sum(errs) / len(errs)
