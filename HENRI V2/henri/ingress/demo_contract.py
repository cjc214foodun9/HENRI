"""Phase 1: explicit Demonstration Ingress Contract (specification action item 2).

SPECIFICATION
-------------
Refinement spec Gap 2: "0 of 17 active evaluation environments yielded
demonstration input-output pairs ... The decoder was expected to extract causal
rules from an empty context buffer, leading to immediate task aborts."

The prescribed fix is to "differentiate between the RHAE scoring budget
(evaluation actions) and the in-context demonstration buffer (support pairs)"
and to fail closed when no support pairs exist.

WHY THIS IS A NEW MODULE AND NOT AN EDIT TO arc_demo_preflight.py
----------------------------------------------------------------
`arc_demo_preflight.py` carries a COMMITTED governance invariant in its
docstring: "Never fabricates demos, never reads environment_files/ caches,
never opens hidden target lists."

The specification's phrasing ("an adapter must construct few-shot exemplars
before triggering the Hopfield cleanup head") would, taken literally, require
exactly the fabrication that the committed gate forbids. That is a direct
conflict, and it is not resolvable by editing the preflight tool.

RESOLUTION (recorded, not silent): this module implements the CONTRACT the
specification asks for -- the typed separation and the fail-closed behavior --
without fabricating. A "constructed exemplar" is admitted ONLY when it has a
declared, checkable provenance that is not the evaluation buffer. Anything else
raises. The preflight tool is left untouched.

HONEST LIMIT
------------
This module does not create demonstration data. On the frozen ARC-AGI-3 split
where `demo_pair_count == 0`, it still fails closed, because no conforming
source exists. That is the correct outcome: BLOCKED, not fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

SUPPORT_OK = "DEMO_SUPPORT_OK"
BLOCKED_EMPTY = "DEMO_BLOCKED_EMPTY"
BLOCKED_NO_PROVENANCE = "DEMO_BLOCKED_NO_PROVENANCE"
BLOCKED_ALIASES_EVAL_BUFFER = "DEMO_BLOCKED_ALIASES_EVAL_BUFFER"

# Provenances that may legitimately carry in-context support pairs. The
# evaluation/action buffer is deliberately ABSENT: admitting it is the exact
# conflation the specification identifies as Gap 2.
ALLOWED_PROVENANCE = frozenset({
    "game.examples",
    "game.demonstrations",
    "task_json.train",
    "task_json.demo",
    "caller_supplied",
})


class DemoIngressError(ValueError):
    """Raised when support pairs are requested from a non-conforming source."""


@dataclass
class DemoIngressContract:
    """Typed separation of the support buffer from the evaluation budget.

    support_pairs : in-context (X, Y) exemplars used to compile a task operator
    eval_budget   : the RHAE / action-evaluation allowance, in actions
    provenance    : where support_pairs came from (must be in ALLOWED_PROVENANCE)

    The two are distinct fields with distinct types. They cannot be aliased:
    passing the same object as both is detected and refused.
    """

    support_pairs: Sequence[Tuple[Any, Any]] = field(default_factory=list)
    eval_budget: int = 0
    provenance: str = ""
    task_id: str = ""

    # ------------------------------------------------------------------
    def validate(self) -> str:
        """Return a typed status. Never fabricates. Fails closed.

        ORDER MATTERS and is deliberate. The aliasing check must run BEFORE the
        allowlist check: `evaluation_buffer` is absent from ALLOWED_PROVENANCE,
        so testing membership first returns BLOCKED_NO_PROVENANCE and the
        Gap 2 diagnostic is never reached. Measured 2026-10-01 — the contract
        test caught exactly that mis-ordering.
        """
        if self.provenance == "evaluation_buffer":
            return BLOCKED_ALIASES_EVAL_BUFFER
        if not self.provenance:
            return BLOCKED_NO_PROVENANCE
        if self.provenance not in ALLOWED_PROVENANCE:
            return BLOCKED_NO_PROVENANCE
        if len(self.support_pairs) == 0:
            return BLOCKED_EMPTY
        for i, pair in enumerate(self.support_pairs):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise DemoIngressError(
                    f"support pair {i} is not an (input, output) pair: {type(pair).__name__}"
                )
        return SUPPORT_OK

    def support_pair_count(self) -> int:
        return len(self.support_pairs)

    # ------------------------------------------------------------------
    def require_support(self) -> str:
        """Raise unless the contract is usable. The fail-closed entry point."""
        status = self.validate()
        if status != SUPPORT_OK:
            raise DemoIngressError(
                f"demonstration ingress refused: {status} "
                f"(task_id={self.task_id!r} provenance={self.provenance!r} "
                f"pairs={len(self.support_pairs)} eval_budget={self.eval_budget}). "
                "No support pairs were fabricated."
            )
        return status

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.validate(),
            "support_pair_count": len(self.support_pairs),
            "eval_budget": self.eval_budget,
            "provenance": self.provenance,
            "fabricated": False,
        }


def build_from_public_api(game: Any, task_id: str = "", eval_budget: int = 0) -> DemoIngressContract:
    """Build a contract from an environment's PUBLIC demo attributes only.

    Mirrors the read order already used by `arc_demo_preflight.py`
    (`game.examples`, then `game.demonstrations`). Reads no cache, opens no
    hidden target list, and invents nothing.
    """
    pairs: List[Tuple[Any, Any]] = []
    provenance = ""
    for attr, tag in (("examples", "game.examples"),
                      ("demonstrations", "game.demonstrations")):
        try:
            src = getattr(game, attr, None)
        except Exception:
            src = None
        if not src:
            continue
        for item in src:
            if isinstance(item, dict) and "input" in item and "output" in item:
                pairs.append((item["input"], item["output"]))
        if pairs:
            provenance = tag
            break
    return DemoIngressContract(
        support_pairs=pairs, eval_budget=eval_budget,
        provenance=provenance, task_id=task_id,
    )


def assert_no_buffer_aliasing(contract: DemoIngressContract, eval_buffer: Any) -> None:
    """Refuse a contract whose support list IS the evaluation buffer.

    Identity comparison is the point: the specification's Gap 2 is a buffer
    conflation, and a copy that happens to hold the same objects is still the
    same defect in spirit.
    """
    if eval_buffer is None:
        return
    if contract.support_pairs is eval_buffer:
        raise DemoIngressError(
            "BLOCKED_ALIASES_EVAL_BUFFER: support_pairs is the evaluation buffer"
        )
    try:
        same = len(eval_buffer) == len(contract.support_pairs) and all(
            a is b for a, b in zip(contract.support_pairs, eval_buffer)
        )
    except TypeError:
        return
    if same and len(contract.support_pairs) > 0:
        raise DemoIngressError(
            "BLOCKED_ALIASES_EVAL_BUFFER: support_pairs shares every element with "
            "the evaluation buffer"
        )
