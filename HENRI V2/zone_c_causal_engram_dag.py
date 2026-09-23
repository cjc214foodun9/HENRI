"""Zone C Causal Engram DAG — blueprint spec HENRI-SPEC-2026-CAUSAL-REALITY-V1, section 3.

WHAT THIS IMPLEMENTS (and what it refuses to implement)
    Section 3.1 -- the empirical causal node (engram):
        wave (real [K, 8] block state), symbolic invariant contract, empirical
        utility counter, last-verification timestamp.
    Section 3.2 -- directed causal edges under action a_k, forged IF AND ONLY IF
        three conditions hold:
          1. temporal priority      t_i < t_j
          2. exteroceptive verify   the environment moved: ext_delta != 0
          3. statistical conjunction  the residual passes tau
        `do(a)` is Pearl's intervention operator: an edge records an INTERVENTION
        HENRI performed, never a passive correlation. Condition 2 is therefore
        the anti-solipsism gate -- ext_delta == 0 is SOLIPSISM_VETO and writes
        NOTHING.
    Section 3.3 -- Landauer metabolic apoptosis: an edge's retention energy
        decays exponentially with idle time and the edge is pruned below a floor.
        Un-reinforced hypotheses evaporate; invariant laws persist.

WHY THIS IS THE DOMAIN FIX, NOT A THRESHOLD FIX
    The recorded defect was a gate that could not discriminate: with an empty
    outcome store every candidate scored identically (delta_axiom == 0.0 exactly,
    8/8). Raising a threshold cannot repair a comparison whose two operands are
    the same object. This store supplies the SECOND, INDEPENDENT operand -- the
    transition the environment actually made -- so a candidate is scored against
    something that is not itself. That is the whole point of section 3.2.

CONTRACTS (each has a control that can FAIL it)
    * family:     every wave is a real [K, 8] unit-block tensor. A complex
                  operand RAISES; that is the recorded cross-family defect.
    * causality:  a future observation must never forge or score a present
                  action. `forge_edge` takes the ALREADY-observed successor.
    * solipsism:  ext_delta == 0 forges no edge, in either direction.
    * pruning:    a pruned edge is LOGGED with its signature and never silently
                  dropped; a dead-input (never-reinforced) hypothesis must be
                  pruned, and a reinforced one must survive the same clock.
    * energy:     the retention decay is information-theoretic. The kT ln 2 ->
                  joules mapping is labelled DERIVED and is NOT claimed as a
                  physical heat measurement.

FLAG
    Production wiring behind `HENRI_ZONEC_CAUSAL_DAG` (default OFF). The classes
    here are pure and flag-free so they stay directly testable.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional, Sequence, Tuple

import torch

try:  # one reader for the gate math: never a second copy
    from uhr02_exteroceptive_gate import (
        AXIOM_BLOCK_NORM_TOL,
        DOMAIN_VIOLATION,
        TAU_BLUEPRINT,
        ad_of,
        assert_single_family,
        delta,
        predict_next,
    )
except Exception:  # pragma: no cover
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from uhr02_exteroceptive_gate import (  # type: ignore
        AXIOM_BLOCK_NORM_TOL,
        DOMAIN_VIOLATION,
        TAU_BLUEPRINT,
        ad_of,
        assert_single_family,
        delta,
        predict_next,
    )

FLAG_ENV = "HENRI_ZONEC_CAUSAL_DAG"
SOLIPSISM_VETO = "SOLIPSISM_VETO"
ATTRIBUTION_VIOLATION = "ATTRIBUTION_VIOLATION"
# 1 information unit = k_B * T * ln 2 at T = 300 K  ->  2.871e-21 J.  DERIVED,
# and reported as a dimensionless unit count; never as measured heat.
KT_LN2_JOULES_300K = 2.871e-21
DEFAULT_TAU_RETAIN_TICKS = 64.0
DEFAULT_PRUNE_FLOOR = 0.05


def flag_enabled() -> bool:
    """Default OFF: the default production path stays byte-identical."""
    return os.environ.get(FLAG_ENV, "0") == "1"


def retention_energy(
    idle_ticks: float,
    e0: float = 1.0,
    tau_retain: float = DEFAULT_TAU_RETAIN_TICKS,
) -> float:
    """E(t) = E0 * 2^(-idle/tau_retain): exponential decay, half-life form.

    The blueprint writes the decay as an equation image that did not survive
    extraction, so the HALF-LIFE form is used and its parameter is explicit.
    Units are information units (see KT_LN2_JOULES_300K); the joules conversion
    is DERIVED.
    """
    if tau_retain <= 0.0:
        raise ValueError("retention_energy: tau_retain must be positive")
    if idle_ticks < 0.0:
        raise ValueError("retention_energy: idle_ticks must be non-negative")
    return float(e0) * (2.0 ** (-float(idle_ticks) / float(tau_retain)))


@dataclass
class CausalEngramNode:
    """Spec 3.1: an invariant state cluster or observational milestone."""

    node_id: str
    wave: torch.Tensor                     # real [K, 8], unit blocks
    contract: str                          # symbolic invariant contract
    utility: int = 0                       # empirical utility counter
    last_verified_tick: int = 0
    grounding_count: int = 0               # exteroceptively verified arrivals

    def as_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "contract": self.contract,
            "utility": self.utility,
            "last_verified_tick": self.last_verified_tick,
            "grounding_count": self.grounding_count,
            "n_blocks": int(self.wave.shape[0]),
        }


@dataclass
class CausalEdge:
    """Spec 3.2: a directed transition invariant (s_i, a_k) -> s_j."""

    src: str
    action: int
    dst: str
    ext_delta: float                       # exteroceptive movement of the env
    delta_pred: float                      # residual of the observed transition
    t_src: int
    t_dst: int
    weight: float = 1.0
    reinforcements: int = 0
    last_reinforced_tick: int = 0

    @property
    def key(self) -> Tuple[str, int, str]:
        return (self.src, int(self.action), self.dst)

    def as_dict(self) -> dict:
        return {
            "src": self.src,
            "action": int(self.action),
            "dst": self.dst,
            "ext_delta": round(float(self.ext_delta), 9),
            "delta_pred": round(float(self.delta_pred), 9),
            "weight": round(float(self.weight), 6),
            "reinforcements": int(self.reinforcements),
            "temporal_priority": bool(self.t_src < self.t_dst),
        }


@dataclass
class PrunedEdge:
    """A rejected candidate is LOGGED, never retried blind (spec 3.3)."""

    key: Tuple[str, int, str]
    reason: str
    energy: float
    idle_ticks: float
    reinforcements: int

    def as_dict(self) -> dict:
        return {
            "src": self.key[0],
            "action": int(self.key[1]),
            "dst": self.key[2],
            "reason": self.reason,
            "energy": float("%.6g" % self.energy),
            "idle_ticks": round(float(self.idle_ticks), 3),
            "reinforcements": int(self.reinforcements),
        }


@dataclass
class ForgeOutcome:
    """Receipt for one intervention attempt."""

    forged: bool
    reason: Optional[str]
    edge: Optional[CausalEdge] = None
    ext_delta: float = 0.0

    def as_dict(self) -> dict:
        d = {"forged": bool(self.forged), "reason": self.reason,
             "ext_delta": round(float(self.ext_delta), 9)}
        d["edge"] = self.edge.as_dict() if self.edge else None
        return d


class ZoneCCausalEngramDAG:
    """Directed causal engram graph with exteroceptive gating and Landauer pruning.

    Both `preference_store_size`-style counters and the DAG are reported so a
    telemetry reader can see the store population directly.
    """

    def __init__(
        self,
        tau: float = TAU_BLUEPRINT,
        tau_retain: float = DEFAULT_TAU_RETAIN_TICKS,
        prune_floor: float = DEFAULT_PRUNE_FLOOR,
        max_nodes: int = 4096,
    ) -> None:
        if not (0.0 < tau < 2.0):
            raise ValueError("tau must be in (0, 2); got %r" % (tau,))
        if prune_floor <= 0.0:
            raise ValueError("prune_floor must be positive")
        self.tau = float(tau)
        self.tau_retain = float(tau_retain)
        self.prune_floor = float(prune_floor)
        self.max_nodes = int(max_nodes)
        self.tick = 0
        self._nodes: Dict[str, CausalEngramNode] = {}
        self._edges: Dict[Tuple[str, int, str], CausalEdge] = {}
        self._pruned: list[PrunedEdge] = []

    # ---------------- node API ----------------
    def add_node(self, node_id: str, wave: torch.Tensor, contract: str) -> CausalEngramNode:
        assert_single_family(wave, "wave")
        if node_id in self._nodes:
            raise ValueError("add_node: duplicate node_id %r" % (node_id,))
        node = node_id
        if len(self._nodes) >= self.max_nodes:
            self._prune_weakest_node()
        self._nodes[node] = CausalEngramNode(
            node_id=node, wave=wave.detach().clone(), contract=str(contract),
            last_verified_tick=self.tick,
        )
        return self._nodes[node]

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def node(self, node_id: str) -> CausalEngramNode:
        if node_id not in self._nodes:
            raise KeyError("unknown node %r" % (node_id,))
        return self._nodes[node_id]

    # ---------------- write path (spec 3.2) ----------------
    def forge_edge(
        self,
        src: str,
        observed_next: torch.Tensor,
        action: int,
        ext_delta: float,
        truth_generators: Sequence[torch.Tensor],
        gell_mann_basis: torch.Tensor,
        dst: Optional[str] = None,
        t_dst: Optional[int] = None,
    ) -> ForgeOutcome:
        """Forge a directed edge from an ALREADY-OBSERVED transition.

        Conditions are checked in the spec's own order so the refusal reason is
        unambiguous:
          1. temporal priority   -- denied if the successor is not later
          2. exteroceptive gate  -- ext_delta == 0 => SOLIPSISM_VETO, nothing written
          3. statistical conjunction -- residual must pass tau
        """
        if src not in self._nodes:
            raise KeyError("forge_edge: unknown src node %r" % (src,))
        assert_single_family(observed_next, "observed_next")
        if not math.isfinite(ext_delta):
            raise ValueError("forge_edge: ext_delta must be finite; got %r" % (ext_delta,))
        t_dst_v = self.tick if t_dst is None else int(t_dst)

        # --- refusal checks -------------------------------------------------
        # ORDER MATTERS FOR DIAGNOSABILITY, and this order is the blueprint's
        # own: section 3.2's comparator diagram branches on "Delta_S_ext == 0:
        # SOLIPSISM_VETO" FIRST. The numbered list above it is a conjunction
        # ("forged if and only if" all three hold), which is order-free; a
        # conjunction cannot say which conjunct failed, so the check order is
        # chosen to name the most fundamental refusal.
        #
        # MEASURED (this module's own smoke test, first run): checking temporal
        # priority first made every forge return NO_TEMPORAL_PRIORITY and MASKED
        # the solipsism and conjunction refusals -- the test could not tell a
        # dead-input control from a timing control. Checking solipsism first
        # fixes that.
        #
        # (1) exteroceptive verification: the anti-solipsism gate. If the
        #     environment did not move, no transition occurred at all.
        if float(ext_delta) == 0.0:
            return ForgeOutcome(False, SOLIPSISM_VETO, ext_delta=ext_delta)

        # (2) temporal priority: a successor at or before the source time is not
        #     a transition. `last_verified_tick` is stamped when the node is
        #     observed, so a caller that has NOT advanced the clock gets this
        #     refusal -- advance() first, then forge.
        if t_dst_v <= self._nodes[src].last_verified_tick:
            return ForgeOutcome(False, "NO_TEMPORAL_PRIORITY", ext_delta=ext_delta)

        # (3) statistical conjunction, measured in the shared domain
        A_t = ad_of(truth_generators, gell_mann_basis)
        pred = predict_next(self._nodes[src].wave, A_t)
        dp = delta(pred, observed_next)
        if dp > self.tau:
            return ForgeOutcome(False, "CONJUNCTION_FAILED", ext_delta=ext_delta)

        dst_id = dst or ("n%d" % (len(self._nodes),))
        if dst_id not in self._nodes:
            self.add_node(dst_id, observed_next, contract="unspecified")

        edge = CausalEdge(
            src=src, action=int(action), dst=dst_id, ext_delta=float(ext_delta),
            delta_pred=dp, t_src=self._nodes[src].last_verified_tick, t_dst=t_dst_v,
            last_reinforced_tick=self.tick,
        )
        self._edges[edge.key] = edge
        self._nodes[src].utility += 1
        self._nodes[dst_id].grounding_count += 1
        self._nodes[dst_id].last_verified_tick = t_dst_v
        return ForgeOutcome(True, None, edge=edge, ext_delta=float(ext_delta))

    def reinforce(self, key: Tuple[str, int, str]) -> CausalEdge:
        """Re-observation of the same conjunction: reinforce, do not duplicate."""
        if key not in self._edges:
            raise KeyError("reinforce: unknown edge %r" % (key,))
        e = self._edges[key]
        e.reinforcements += 1
        e.weight = 1.0 - math.exp(-(e.reinforcements + 1) / 4.0)
        e.last_reinforced_tick = self.tick
        return e

    # ---------------- tick + metabolic apoptosis (spec 3.3) ----------------
    def advance(self, ticks: int = 1) -> list[PrunedEdge]:
        """Advance the clock and return edges pruned on this step."""
        if ticks < 0:
            raise ValueError("advance: ticks must be non-negative")
        self.tick += int(ticks)
        return self.prune()

    def prune(self, observed_keys: Optional[set] = None) -> list[PrunedEdge]:
        """Prune edges whose retention energy has fallen below the floor.

        An edge touched this tick is never pruned: pruning is a function of IDLE
        time, so the strongest surviving edge is the one just reinforced. Pruned
        edges are logged with their signature.
        """
        newly: list[PrunedEdge] = []
        for key in list(self._edges.keys()):
            e = self._edges[key]
            idle = float(self.tick - e.last_reinforced_tick)
            energy = retention_energy(idle, e0=1.0, tau_retain=self.tau_retain)
            if energy < self.prune_floor and idle > 0.0:
                self._pruned.append(PrunedEdge(
                    key=key, reason="LANDAUER_APOPTOSIS", energy=energy,
                    idle_ticks=idle, reinforcements=e.reinforcements,
                ))
                del self._edges[key]
                newly.append(self._pruned[-1])
        if observed_keys:
            for k in observed_keys:
                if k in self._edges:
                    self.reinforce(k)
        return newly

    def _prune_weakest_node(self) -> None:
        weakest = min(self._nodes.values(), key=lambda n: (n.utility, n.grounding_count))
        for key in [k for k in self._edges if weakest.node_id in (k[0], k[2])]:
            e = self._edges.pop(key)
            self._pruned.append(PrunedEdge(
                key=key, reason="CAPACITY_EVICTION", energy=0.0,
                idle_ticks=float(self.tick - e.last_reinforced_tick),
                reinforcements=e.reinforcements,
            ))
        del self._nodes[weakest.node_id]

    # ---------------- read path ----------------
    def gate_candidate(
        self,
        src: str,
        candidate_generators: Sequence[torch.Tensor],
        gell_mann_basis: torch.Tensor,
        recorded_successor: Optional[torch.Tensor] = None,
    ) -> dict:
        """Score a candidate against the LAST RECORDED successor of `src`.

        Past constant conjunction only: the successor is a stored observation,
        never a future one. Raises when `src` has no grounded edge yet, because
        scoring against nothing is exactly the self-comparison degeneracy.
        """
        if src not in self._nodes:
            raise KeyError("gate_candidate: unknown src node %r" % (src,))
        succ = recorded_successor
        if succ is None:
            edges = [e for e in self._edges.values() if e.src == src]
            if not edges:
                raise ValueError(
                    "%s: node %r has no grounded edge; a candidate cannot be "
                    "scored against an unrecorded transition" % (ATTRIBUTION_VIOLATION, src)
                )
            succ = self._nodes[edges[-1].dst].wave
        assert_single_family(succ, "recorded_successor")
        A_c = ad_of(candidate_generators, gell_mann_basis)
        pred = predict_next(self._nodes[src].wave, A_c)
        return {
            "delta_pred": round(delta(pred, succ), 9),
            "delta_state": round(delta(pred, self._nodes[src].wave), 9),
            "hard_vetoed": bool(delta(pred, succ) > self.tau),
            "tau": self.tau,
        }

    # ---------------- telemetry ----------------
    def store_size(self) -> int:
        """The population counter a telemetry reader watches (UHR-01 blocker)."""
        return len(self._edges)

    def count_populated(self) -> dict:
        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "pruned": len(self._pruned),
            "tick": self.tick,
            "tau": self.tau,
        }

    def pruned_log(self) -> list[dict]:
        return [p.as_dict() for p in self._pruned]

    def edges(self) -> Iterator[CausalEdge]:
        return iter(self._edges.values())

    def weakest_energy(self) -> float:
        if not self._edges:
            return 0.0
        return min(
            retention_energy(float(self.tick - e.last_reinforced_tick), tau_retain=self.tau_retain)
            for e in self._edges.values()
        )


__all__ = [
    "ATTRIBUTION_VIOLATION",
    "CausalEdge",
    "CausalEngramNode",
    "DEFAULT_PRUNE_FLOOR",
    "DEFAULT_TAU_RETAIN_TICKS",
    "FLAG_ENV",
    "ForgeOutcome",
    "KT_LN2_JOULES_300K",
    "PrunedEdge",
    "SOLIPSISM_VETO",
    "ZoneCCausalEngramDAG",
    "flag_enabled",
    "retention_energy",
]
