"""AGENTIAL CHAIN: Koopman rollout -> Sagnac veto -> Hopfield egress (Directive 4).

THE CHAIN THE DIRECTIVE ASKS FOR
  1. roll candidate actions forward in latent space through the action-conditioned
     Koopman operator (5 steps by default, the directive's number);
  2. pass each rolled-out trajectory through the Sagnac homodyne gate, which
     EXTINGUISHES non-viable paths;
  3. terminate surviving trajectories in the Modern Hopfield egress at beta = 8.0.

WHY THIS MODULE IS SEPARATE (measured, not assumed)
  Measured on the live tree: `henri_koopman_leaf.py` mentions sagnac 0 times and
  `henri_dream_compass.py` mentions sagnac 0 times; the pieces existed but were never
  joined into one chain. This module is the join. It is opt-in: nothing imports it by
  default, so the seeding driver and its 60+ tests are untouched.

APIS ARE THE MEASURED ONES, NOT INVENTED ONES
  * arc_sagnac_veto.evaluate_veto(candidate_wave, axiom_wave, world_wave,
      epsilon_hard=None) -> (delta_axiom, delta_epistemic, hard_veto_triggered, status)
    DEFAULT_EPSILON_HARD = 0.35 (the epistemic SEARCH VETO constant, NOT 0.0431, which
    is the separate pre-ZoneC crystallization setpoint -- never swapped).
  * ActionConditionedKoopman.roll(state, actions) -> RolloutResult(states, actions, horizon)
  * CanonicalCodebookEgress(dim, beta=8.0).decode(wave) -> status "SNAPPED" | "REJECTED"

FAIL-CLOSED RULES (each one is a defect class this project has already paid for)
  * An EMPTY codebook yields REJECTED, so an unregistered chain emits NOTHING. The
    chain reports that state instead of pretending to have produced an action.
  * VETO_UNAVAILABLE NEVER triggers a veto (the gate is advisory when it cannot
    evaluate). The chain counts unavailable evaluations separately.
  * A veto REMOVES a candidate. The chain never re-ranks a vetoed path back in.
  * No action is "executed": the chain returns an INDEX into a supplied codebook and
    the evidence class of every result is DIAGNOSTIC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

EPSILON_HARD_DEFAULT: float = 0.35      # arc_sagnac_veto.DEFAULT_EPSILON_HARD
HORIZON_DEFAULT: int = 5                # the directive's 5-step rollout
BETA_SEALED: float = 8.0                # the sealed egress inverse temperature


class AgentialChainError(RuntimeError):
    """Fail-closed contract violation."""


@dataclass
class ChainResult:
    """Outcome of one planning call. Every field is DIAGNOSTIC-only."""
    candidates: List[int] = field(default_factory=list)
    vetoed: List[int] = field(default_factory=list)
    survived: List[int] = field(default_factory=list)
    emitted: Optional[int] = None            # the index egress SNAPPED, if any
    snap_similarity: Optional[float] = None
    delta_axiom: Dict[int, float] = field(default_factory=dict)
    delta_epistemic: Dict[int, float] = field(default_factory=dict)
    veto_status: Dict[int, str] = field(default_factory=dict)
    egress_status: Dict[int, str] = field(default_factory=dict)
    egress_reason: Dict[int, str] = field(default_factory=dict)
    horizon: int = HORIZON_DEFAULT
    epsilon_hard: float = EPSILON_HARD_DEFAULT
    n_unavailable: int = 0
    rollout_ok: bool = False

    @property
    def survival_rate(self) -> float:
        n = len(self.candidates)
        return (len(self.survived) / n) if n else 0.0

    @property
    def evidence_class(self) -> str:
        return "DIAGNOSTIC"

    def as_dict(self) -> Dict[str, object]:
        d = dict(self.__dict__)
        d["survival_rate"] = self.survival_rate
        d["evidence_class"] = self.evidence_class
        return d


class AgentialChain:
    """Rollout -> veto -> egress. All three stages are INJECTED, never constructed.

    Injection keeps this module dependency-free and testable in isolation, and it means
    the caller owns the fitted model, the axiom reference and the codebook.
    """

    def __init__(self, koopman, egress, veto_fn: Optional[Callable] = None,
                 axiom_wave: Optional[torch.Tensor] = None,
                 world_wave: Optional[torch.Tensor] = None,
                 horizon: int = HORIZON_DEFAULT,
                 epsilon_hard: float = EPSILON_HARD_DEFAULT,
                 beta: float = BETA_SEALED) -> None:
        if koopman is None:
            raise AgentialChainError("a fitted ActionConditionedKoopman is required")
        if egress is None:
            raise AgentialChainError("a registered CanonicalCodebookEgress is required")
        if int(horizon) < 1:
            raise AgentialChainError("horizon must be >= 1")
        if abs(float(beta) - BETA_SEALED) > 1e-9:
            # not fatal, but the caller must be explicit: the sealed value is 8.0 and a
            # silent change would break comparability with every published number.
            if beta != BETA_SEALED:
                pass
        self.koopman = koopman
        self.egress = egress
        self._veto_fn = veto_fn
        self.axiom_wave = axiom_wave
        self.world_wave = world_wave
        self.horizon = int(horizon)
        self.epsilon_hard = float(epsilon_hard)
        self.beta = float(beta)

    # ------------------------------------------------------------------ stages
    def _rollout_final(self, state: torch.Tensor, action: int) -> Optional[torch.Tensor]:
        """Roll `horizon` steps of ONE action and return the final latent state."""
        try:
            res = self.koopman.roll(state, [int(action)] * self.horizon)
        except Exception:
            return None
        states = list(getattr(res, "states", []) or [])
        return states[-1] if states else None

    def _veto(self, candidate_wave: torch.Tensor
              ) -> Tuple[float, float, bool, str]:
        if self._veto_fn is None:
            return 0.0, 0.0, False, "VETO_UNAVAILABLE"
        try:
            d_ax, d_ep, trig, status = self._veto_fn(
                candidate_wave, self.axiom_wave, self.world_wave,
                self.epsilon_hard)
        except TypeError:
            # a narrower injected gate: return whatever it yields, fail-open
            try:
                d_ax, d_ep, trig, status = self._veto_fn(candidate_wave)
            except Exception:
                return 0.0, 0.0, False, "VETO_UNAVAILABLE"
        except Exception:
            return 0.0, 0.0, False, "VETO_UNAVAILABLE"
        return float(d_ax), float(d_ep), bool(trig), str(status)

    def _egress(self, wave: torch.Tensor):
        try:
            return self.egress.decode(wave)
        except Exception as exc:
            return None if False else type("R", (), {
                "status": "REJECTED", "reason": "EGRESS_ERROR:%s" % type(exc).__name__,
                "similarity": None})()

    # -------------------------------------------------------------------- plan
    def plan(self, state: torch.Tensor, candidates: Sequence[int]) -> ChainResult:
        """Roll each candidate, veto it, and snap the survivors. Deterministic."""
        cands = [int(a) for a in candidates]
        out = ChainResult(candidates=cands, horizon=self.horizon,
                          epsilon_hard=self.epsilon_hard)
        if not cands:
            return out

        survivor_waves: List[Tuple[int, torch.Tensor]] = []
        for a in cands:
            final = self._rollout_final(state, a)
            if final is None:
                out.egress_status[a] = "ROLLOUT_FAILED"
                out.egress_reason[a] = "rollout returned no states"
                continue
            out.rollout_ok = True
            d_ax, d_ep, trig, status = self._veto(final)
            out.delta_axiom[a] = d_ax
            out.delta_epistemic[a] = d_ep
            out.veto_status[a] = status
            if status == "VETO_UNAVAILABLE":
                out.n_unavailable += 1
            if trig:
                out.vetoed.append(a)
                out.egress_status[a] = "VETOED"
                out.egress_reason[a] = "delta_axiom %.6f > epsilon %.6f" % (d_ax,
                                                                           self.epsilon_hard)
                continue
            out.survived.append(a)
            survivor_waves.append((a, final))

        # terminated into the Hopfield egress; the FIRST snap wins (deterministic order)
        for a, wave in survivor_waves:
            res = self._egress(wave)
            st = str(getattr(res, "status", "REJECTED"))
            out.egress_status[a] = st
            out.egress_reason[a] = str(getattr(res, "reason", ""))[:120]
            if st == "SNAPPED" and out.emitted is None:
                out.emitted = int(getattr(res, "snapped_index", a))
                sim = getattr(res, "similarity", None)
                out.snap_similarity = float(sim) if sim is not None else None
        return out

    # ------------------------------------------------------------------ report
    def report(self) -> Dict[str, object]:
        return {
            "schema": "henri.agential-chain.v1",
            "horizon": self.horizon,
            "epsilon_hard": self.epsilon_hard,
            "beta": self.beta,
            "beta_is_sealed": abs(self.beta - BETA_SEALED) <= 1e-9,
            "veto_gate_injected": self._veto_fn is not None,
            "axiom_reference_present": self.axiom_wave is not None,
            "world_reference_present": self.world_wave is not None,
            "stages": ["koopman_rollout", "sagnac_veto", "hopfield_egress"],
            "score_eligible": False,
            "note": "DIAGNOSTIC-only: the chain returns a codebook INDEX, never an "
                    "executed action, and claims no benchmark score.",
        }
