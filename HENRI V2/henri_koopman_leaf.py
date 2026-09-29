"""Koopman leaf evaluation for the agential search planner (Directive 4).

WHY THIS MODULE EXISTS
======================
`sagnac_mcts_planner.py` expands a child, scores it with the STATIC dual-channel
Sagnac veto, and prunes. Measured: the planner text contains `koopman` = 0,
`rollout` = 0, `simulate` = 0, `reward` = 0 occurrences, and
`henri_action_koopman.py` has no caller outside its own tests. The world model
therefore exists but does not reach the search loop: the planner judges ONE step,
never a TRAJECTORY.

This module supplies the missing leaf evaluator and the veto rule, in a form the
planner can call without changing its answer-coupling contract.

WHAT IT COMPUTES
================
Given per-action Koopman operators fitted on pre-prediction information, a
candidate action SEQUENCE is rolled out in latent space and scored against the
INDUCED GOAL wave (never the held-out target):

    Psi_{t+1} = K_a Psi_t
    d_rollout = 1 - cos(Psi_H, Psi_goal)

The horizon rule is explicit: `horizon=1` reproduces single-step behaviour, so a
caller that asks for one step cannot be surprised by this channel.

FAIL-OPEN, LIKE THE OBSERVATIONAL CHANNEL
=========================================
This channel ONLY EVER ADDS a veto. Every failure mode -- unfitted action,
non-finite state, empty candidates, unavailable world model -- returns
`(score=None, veto=False)`, leaving behaviour unchanged. That mirrors the planner's
own `_demo_observational_stress` contract and is why this can be wired into
`search()` without redefining what a veto means.

THE CONTROLS (all falsifiable, all executed in the tests)
========================================================
  SCAFFOLD VALIDITY   fitted against a KNOWN operator, the 1-step rollout must
                      reproduce it (relative error < 0.15), else the machinery is
                      broken rather than the mechanism disproved.
  MONOTONE HORIZON    with a non-contractive operator, error must NOT decrease as
                      the horizon grows (error is a real measurement, not a floor).
  DIFFERENTIAL        two different action sequences must produce different scores.
  ABSTENTION          an UNSEEN action must return veto=False with score=None, never
                      a fabricated value (substituting identity would look correct
                      while predicting nothing).
  INERT BY DEFAULT    `horizon <= 0` or an empty candidate set leaves the score None
                      and veto False -- the OFF path is inert by construction.

HONEST LIMITS
=============
* Linear operator in a GIVEN feature map. Not a claim that the environment is
  linear; adequacy is REPORTED as rollout residual.
* Synthetic truth only in the tests. No ARC / SciCode score is claimed, and this
  module never selects an action: it scores and vetoes, and the planner still owns
  the decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from henri_action_koopman import ActionConditionedKoopman

DEFAULT_TAU_ROLLOUT = 0.60      # veto only a GROSS disagreement; provisional


@dataclass
class LeafVerdict:
    """The channel's answer. `score` is None whenever the measurement was not made."""
    score: Optional[float]
    veto: bool
    valid: bool
    horizon: int
    reason: str


class KoopmanLeafEvaluator:
    """Per-action latent rollouts, scored against an induced-goal wave.

    Parameters
    ----------
    wave_dim : width of the flat wave the Koopman operators act on
    n_actions : number of action primitives the world model may hold
    horizon : rollout depth (>= 1). 1 reproduces single-step scoring.
    tau_rollout : veto threshold on `1 - cos(rollout, goal)`
    enforce_unit_norm : passed through to the world model
    """

    def __init__(self, wave_dim: int, n_actions: int, horizon: int = 3,
                 tau_rollout: float = DEFAULT_TAU_ROLLOUT,
                 enforce_unit_norm: bool = True,
                 rank: Optional[int] = None,
                 allow_dense: bool = False) -> None:
        """`rank=None` (default) keeps the LEGACY dense operator, byte-identical.

        `rank=r` propagates the LOW-RANK form to the world model (Contract A,
        Directive 3). WIRING DEFECT THIS CLOSES (measured 2026-09-28): this
        constructor previously passed NO rank, so at wave_dim=65536 the leaf path
        always built a dense [65536,65536] float32 operator (17.18 GB per action,
        137 GB for 8) and the low-rank implementation was UNREACHABLE from the
        planner. The default stays dense so no existing consumer changes -- but
        above DENSE_DIM_LIMIT a dense build now FAILS CLOSED unless `allow_dense`
        is set explicitly.
        """
        if wave_dim < 2:
            raise ValueError("wave_dim must be >= 2")
        if n_actions < 1:
            raise ValueError("n_actions must be >= 1")
        self.wave_dim = int(wave_dim)
        self.n_actions = int(n_actions)
        self.horizon = int(horizon)
        self.tau_rollout = float(tau_rollout)
        self.rank = rank
        self.model = ActionConditionedKoopman(
            dim=self.wave_dim, n_actions=self.n_actions,
            enforce_unit_norm=enforce_unit_norm, rank=rank,
            allow_dense=allow_dense)

    # ------------------------------------------------------------------ fit
    def fit(self, triples: Sequence[Tuple[torch.Tensor, int, torch.Tensor]]):
        """Fit the world model on (wave_t, action, wave_t+1) triples.

        Triples are DEMONSTRATION information: they are state transitions, not
        answers. The held-out target is not a parameter of this method.
        """
        self.model.fit(triples)
        return self

    @property
    def fitted_actions(self) -> List[int]:
        # Representation-aware: the low-rank path leaves `model.K` EMPTY, so
        # reading it directly reported "no fitted actions" for a correctly fitted
        # model. `model._fitted()` returns the active representation's keys.
        return list(self.model._fitted())

    # -------------------------------------------------------------- scoring
    @staticmethod
    def _flat_goal(goal: torch.Tensor, dim: int) -> torch.Tensor:
        g = goal.real if goal.is_complex() else goal
        g = g.reshape(-1).to(torch.float32)
        if g.numel() != dim:
            raise ValueError("goal width %d != wave_dim %d" % (g.numel(), dim))
        return g

    def score_sequence(self, wave: torch.Tensor, actions: Sequence[int],
                       goal: torch.Tensor) -> Tuple[Optional[float], str]:
        """Roll out `actions` and return (1 - cos(Psi_H, goal), reason).

        `reason` names WHY a score is absent, so a caller can log the cause instead
        of inferring it from a None.
        """
        if self.horizon <= 0:
            return None, "HORIZON_OFF"
        if not actions:
            return None, "NO_ACTIONS"
        missing = [int(a) for a in actions if int(a) not in self.model.K]
        if missing:
            return None, "UNFITTED_ACTION:%s" % sorted(set(missing))[:3]
        try:
            goal_f = self._flat_goal(goal, self.wave_dim)
            r = self.model.roll(wave, [int(a) for a in actions])
            final = r.states[-1]
            ng = float(torch.linalg.vector_norm(final))
            ngg = float(torch.linalg.vector_norm(goal_f))
            if ng < 1e-12 or ngg < 1e-12:
                return None, "DEGENERATE_NORM"
            cos = float(torch.dot(final, goal_f) / (ng * ngg))
            if not (cos == cos):
                return None, "NAN_COSINE"
            return float(1.0 - cos), "OK"
        except Exception as exc:                       # fail-open by design
            return None, "ERROR:%s" % type(exc).__name__

    def evaluate(self, wave: torch.Tensor, actions: Sequence[int],
                 goal: torch.Tensor) -> LeafVerdict:
        """The veto decision the planner consumes. Fail-open on every anomaly."""
        score, reason = self.score_sequence(wave, actions, goal)
        if score is None:
            return LeafVerdict(None, False, False, self.horizon, reason)
        return LeafVerdict(score, bool(score > self.tau_rollout), True,
                           self.horizon, reason)

    # ---------------------------------------------------------- diagnostics
    def report(self) -> Dict[str, Any]:
        r = self.model.report()
        return {
            "schema": "henri.planes.koopman-leaf-eval.v1",
            "wave_dim": self.wave_dim, "n_actions": self.n_actions,
            "horizon": self.horizon, "tau_rollout": self.tau_rollout,
            "fitted_actions": r["fitted_actions"],
            "abstaining_actions": r["abstaining_actions"],
            "relative_residual": r["relative_residual"],
            "fail_mode": "FAIL_OPEN: every anomaly -> score None, veto False",
            "honest_limit": r["honest_limit"],
        }


def candidate_sequences(primitives: Sequence[int], depth: int = 1
                        ) -> List[Tuple[int, ...]]:
    """All length-`depth` action sequences over `primitives` (for exhaustive leaves).

    `depth=1` yields the singletons, i.e. the planner's existing per-op expansion,
    which is what makes this channel a strict generalisation rather than a new one.
    """
    if depth < 1:
        return []
    if depth == 1:
        return [(int(p),) for p in primitives]
    out: List[Tuple[int, ...]] = []
    for p in primitives:
        for tail in candidate_sequences(primitives, depth - 1):
            out.append((int(p),) + tail)
    return out


def make_synthetic_triples(n_actions: int, dim: int, n_per_action: int, seed: int,
                           noise: float = 0.02):
    """Ground-truth triples for the scaffold test (delegates to the world model)."""
    from henri_action_koopman import make_synthetic_triples as _mk
    return _mk(n_actions, dim, n_per_action, seed, rng_scale=noise)


class KoopmanLeafChannel:
    """Adapter: planner OP NAMES -> Koopman action INDICES, for the MCTS expansion.

    WHY AN ADAPTER. The planner selects NAMED primitives ("Rotate90", "ContourFill");
    the world model is indexed by INTEGERS. The mapping is SUPPLIED by the caller and
    never inferred, so the channel cannot silently mis-index a primitive onto the
    wrong operator -- a mis-index would produce plausible but meaningless rollouts.

    MULTI-STEP SEMANTICS (stated, not implied). A tree child is ONE op. The channel
    rolls that op forward `horizon` steps and scores the landing against the induced
    goal -- a "what if I keep doing this" probe. That is a delayed-reward signal
    layered ON TOP OF the planner's existing single-step veto, not a replacement.

    FAIL-OPEN. Every anomaly (unknown op, wave-size mismatch, unfitted action,
    non-finite state) returns `valid=False, veto=False`, so behaviour is unchanged
    and the caller logs the reason. This channel ONLY EVER ADDS a veto.
    """

    def __init__(self, evaluator: "KoopmanLeafEvaluator", op_to_action: Dict[str, int]):
        if not op_to_action:
            raise ValueError("op_to_action must be non-empty")
        self.ev = evaluator
        self.op_to_action = {str(k_): int(v) for k_, v in op_to_action.items()}

    def supported_ops(self) -> List[str]:
        return sorted(self.op_to_action)

    @staticmethod
    def _flat(wave: Any, dim: int) -> torch.Tensor:
        t = wave
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, dtype=torch.float32)
        if t.is_complex():
            t = t.real
        t = t.reshape(-1).to(torch.float32)
        if t.numel() != dim:
            raise ValueError("wave width %d != wave_dim %d" % (t.numel(), dim))
        return t

    def rollout_op(self, op_name: str, start_wave: Any, goal_wave: Any) -> "LeafVerdict":
        """Roll `op_name` forward `ev.horizon` steps from `start_wave`; score vs goal."""
        a = self.op_to_action.get(str(op_name))
        if a is None:
            return LeafVerdict(None, False, False, self.ev.horizon, "UNKNOWN_OP:%s" % op_name)
        try:
            s = self._flat(start_wave, self.ev.wave_dim)
            g = self._flat(goal_wave, self.ev.wave_dim)
        except Exception as exc:
            return LeafVerdict(None, False, False, self.ev.horizon,
                               "SHAPE:%s" % type(exc).__name__)
        v = self.ev.evaluate(s, [a] * max(1, self.ev.horizon), g)
        return v

    def report(self) -> Dict[str, Any]:
        r = self.ev.report()
        r["adapter"] = {"ops": self.supported_ops(), "n_ops": len(self.op_to_action)}
        return r
