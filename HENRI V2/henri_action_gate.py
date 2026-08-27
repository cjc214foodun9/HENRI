"""Phase 10.2 typed action gate (default-OFF, fail-closed).

Four-Phase Reconciliation Report (HENRI-ARCH-2026-TEMPORAL-GROUNDING-
RECONCILIATION, inbox sha 7313359b) Phase 10.2 structural gate:

    Deep Egress Logits
      -> Hopfield Energy Minimization / hard snap to nearest action engram
      -> Filter Against Legal Action Subset Mask (A_legal)
      -> Extract Typed (GameAction, coordinate_payload)
      -> If Malformed -> Fail-Closed Rejection (No-Op)
      -> External Environment Execution

This module implements the gate: a typed (GameAction, data) tuple is emitted
ONLY when the Hopfield-decoded action is in the legal subset AND a valid
payload exists. Every other path returns a TypedActionRejection with an
explicit reason (ACTION_NOT_LEGAL, PAYLOAD_MALFORMED, DECODE_FAILED) — the
caller must treat a rejection as a No-Op, never as a fallback action.

Zero trainable parameters. Default-OFF: HENRI_ACTION_GATE=1 must be set or
get_action_gate() returns None (factory contract; the production runner
never imports this module without the flag).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

from arc_action_payloads import (
    DEFAULT_COMPLEX_ACTION_NAMES,
    build_payload_candidates,
    select_payload,
)

FLAG = "HENRI_ACTION_GATE"

REASON_ACTION_NOT_LEGAL = "ACTION_NOT_LEGAL"
REASON_PAYLOAD_MALFORMED = "PAYLOAD_MALFORMED"
REASON_DECODE_FAILED = "DECODE_FAILED"


@dataclass(frozen=True)
class TypedActionRejection:
    """Fail-closed No-Op result. The caller must NOT execute any action."""

    reason: str
    decoded_action_name: Optional[str] = None
    legal_actions: Tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class TypedAction:
    """Typed egress action: complete (GameAction, data) tuple."""

    action: Any
    data: Optional[dict]
    source: str
    x: Optional[int] = None
    y: Optional[int] = None
    coordinate_space: str = "grid"
    confidence: float = 0.0

    @property
    def payload_complete(self) -> bool:
        return self.data is not None


class TypedActionGate:
    """Hopfield snap -> legal-subset mask -> typed (GameAction, data) -> fail-closed."""

    def __init__(
        self,
        decoder: Any,
        complex_action_names: Tuple[str, ...] = DEFAULT_COMPLEX_ACTION_NAMES,
        seed: int = 0,
        confidence_threshold: float = 0.0,
    ):
        self.decoder = decoder
        self.complex_action_names = tuple(complex_action_names)
        self.seed = seed
        self.confidence_threshold = float(confidence_threshold)

    def _legal_names(self, allowed_actions: Sequence[Any]) -> Tuple[str, ...]:
        return tuple(getattr(a, "name", str(a)) for a in allowed_actions)

    def gate(
        self,
        policy_wave: Any,
        grid: Sequence[Sequence[int]],
        allowed_actions: Sequence[Any],
    ):
        """Return a TypedAction on success or a TypedActionRejection (No-Op).

        Never raises on a malformed action; it returns a rejection with an
        explicit reason. The caller executes an action ONLY on TypedAction.
        """
        legal_names = self._legal_names(allowed_actions)

        # 1. Hopfield hard snap to the nearest action engram.
        try:
            decoded_action, confidence = self.decoder.decode_wave_to_action(policy_wave)
        except Exception as exc:
            return TypedActionRejection(
                reason=REASON_DECODE_FAILED, legal_actions=legal_names,
                detail=f"{type(exc).__name__}: {exc}")
        decoded_name = getattr(decoded_action, "name", str(decoded_action))

        # 2. Legal-subset mask (A_legal). A decoded action outside the legal
        #    set is a fail-closed No-Op — never silently remapped.
        if decoded_name not in legal_names:
            return TypedActionRejection(
                reason=REASON_ACTION_NOT_LEGAL,
                decoded_action_name=decoded_name,
                legal_actions=legal_names,
            )

        # 3. Typed payload extraction: coordinate actions need (GameAction,
        #    data); simple actions carry data=None (no invented payloads).
        data: Optional[dict] = None
        source = "none"
        x = y = None
        coord_space = "grid"
        if decoded_name in self.complex_action_names:
            try:
                candidates = build_payload_candidates(
                    grid, allowed_actions,
                    complex_action_names=self.complex_action_names,
                    max_candidates=8, seed=self.seed,
                )
                chosen = select_payload(candidates, decoded_action)
            except Exception as exc:
                return TypedActionRejection(
                    reason=REASON_PAYLOAD_MALFORMED,
                    decoded_action_name=decoded_name,
                    legal_actions=legal_names,
                    detail=f"payload build failed: {type(exc).__name__}: {exc}")
            if chosen is None or chosen.data is None:
                return TypedActionRejection(
                    reason=REASON_PAYLOAD_MALFORMED,
                    decoded_action_name=decoded_name,
                    legal_actions=legal_names,
                    detail="no payload candidate for coordinate action")
            data = chosen.data
            source = chosen.source
            x, y = chosen.x, chosen.y
            coord_space = chosen.coordinate_space
        elif decoded_action is None:
            return TypedActionRejection(
                reason=REASON_DECODE_FAILED, legal_actions=legal_names,
                detail="decoded action is None")

        # 4. Confidence gate (optional; threshold default 0.0 disables it).
        if confidence < self.confidence_threshold:
            return TypedActionRejection(
                reason=REASON_DECODE_FAILED,
                decoded_action_name=decoded_name,
                legal_actions=legal_names,
                detail=f"confidence {confidence:.4f} < {self.confidence_threshold}")

        return TypedAction(
            action=decoded_action, data=data, source=source,
            x=x, y=y, coordinate_space=coord_space, confidence=float(confidence),
        )


def get_action_gate(decoder: Any, **kwargs) -> Optional[TypedActionGate]:
    """Flag-gated factory: returns None unless HENRI_ACTION_GATE=1."""
    if os.environ.get(FLAG, "0") != "1":
        return None
    return TypedActionGate(decoder, **kwargs)
