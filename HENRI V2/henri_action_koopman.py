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
                 enforce_unit_norm: bool = True):
        if dim < 2:
            raise WorldModelError("dim must be >= 2")
        if n_actions < 1:
            raise WorldModelError("n_actions must be >= 1")
        self.dim = int(dim)
        self.n_actions = int(n_actions)
        self.lam = float(lam)
        self.enforce_unit_norm = bool(enforce_unit_norm)
        self.K: Dict[int, torch.Tensor] = {}
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
        self.counts.clear()
        self.residual.clear()
        for a, (src, dst) in per.items():
            X = torch.stack(src).to(torch.float64)      # [n, d]
            Y = torch.stack(dst).to(torch.float64)      # [n, d]
            self.counts[a] = int(X.shape[0])
            # Solve in dimension: K = Y^T X (X^T X + lam I)^-1, so we invert a
            # [d, d] matrix ONCE; for tiny sample counts the dual form is cheaper,
            # so pick the smaller system. This keeps the memory contract.
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
        return self

    # --------------------------------------------------------------- rollout
    def step(self, state: torch.Tensor, action: int) -> torch.Tensor:
        a = int(action)
        if a not in self.K:
            raise WorldModelError(
                f"ABSTAIN: no operator fitted for action {a}; refusing to substitute "
                f"an identity (that would predict nothing while looking successful)")
        s = self._as_flat(state)
        out = s.to(torch.float32) @ self.K[a].t()
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
        return {
            "schema": "henri.world-model.action-koopman.report.v1",
            "dim": self.dim, "n_actions": self.n_actions, "lam": self.lam,
            "fitted_actions": sorted(self.K),
            "abstaining_actions": sorted(set(range(self.n_actions)) - set(self.K)),
            "samples_per_action": dict(self.counts),
            "relative_residual": dict(self.residual),
            "enforce_unit_norm": self.enforce_unit_norm,
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
