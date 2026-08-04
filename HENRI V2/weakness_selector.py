"""Bounded extension-mass selection for HENRI candidate ties.

This module implements a finite, pre-decision diagnostic inspired by Bennett's
weakness formalism.  It does not infer a hypothesis extension from text length,
model entropy, similarity, or a post-hoc outcome.

For a candidate with an explicit finite continuation set E(c), the measured
quantity is extension mass W(c) = |E(c)|.  The selector may use W(c) only to
break a declared tie in an existing baseline ranking.  It never changes a
non-tied ranking and never admits a rejected candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Any, Sequence


class WeaknessSelectorError(ValueError):
    """Raised when finite extension data violates the selector contract."""


@dataclass(frozen=True)
class WeaknessSelection:
    """Decision produced by the deterministic tie-breaker."""

    selected_position: int
    status: str
    tie_positions: tuple[int, ...]
    selected_extension_mass: int | None


def _as_sequence(value: Any, *, name: str) -> list[Any]:
    """Materialize only bounded sequence-like inputs.

    Tensor inputs are accepted only through their explicit ``tolist`` method;
    arbitrary iterators are rejected because their resource use is unknown.
    """
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        value = value.detach().cpu().tolist()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise WeaknessSelectorError(f"{name} must be a finite sequence")
    return list(value)


def _count_mask(mask: Any, *, candidate: int, max_extensions: int) -> int:
    values = _as_sequence(mask, name=f"extension mask {candidate}")
    if len(values) > max_extensions:
        raise WeaknessSelectorError(
            f"extension mask {candidate} exceeds max_extensions={max_extensions}"
        )
    if not all(isinstance(item, bool) for item in values):
        raise WeaknessSelectorError(
            f"extension mask {candidate} must contain only boolean values"
        )
    return sum(values)


def normalize_extension_masses(
    extensions: Any,
    candidate_count: int,
    *,
    max_extensions: int = 1_000_000,
) -> tuple[int, ...]:
    """Return exact finite extension cardinalities for each candidate.

    ``extensions`` is either a sequence of non-negative integer cardinalities
    or a sequence of finite boolean masks.  The outer sequence must align with
    ``candidate_count``.  The bound is a safety limit, not an estimate.
    """
    if not isinstance(candidate_count, int) or candidate_count < 1:
        raise WeaknessSelectorError("candidate_count must be a positive integer")
    if not isinstance(max_extensions, int) or max_extensions < 0:
        raise WeaknessSelectorError("max_extensions must be a non-negative integer")

    outer = _as_sequence(extensions, name="extensions")
    if len(outer) != candidate_count:
        raise WeaknessSelectorError(
            f"extensions length {len(outer)} != candidate_count {candidate_count}"
        )

    masses: list[int] = []
    for index, item in enumerate(outer):
        if isinstance(item, (str, bytes, bytearray)):
            raise WeaknessSelectorError(f"extension {index} has invalid type")
        if isinstance(item, Integral) and not isinstance(item, bool):
            mass = int(item)
            if mass < 0 or mass > max_extensions:
                raise WeaknessSelectorError(
                    f"extension mass {index} must be in [0, {max_extensions}]"
                )
            masses.append(mass)
            continue
        if isinstance(item, Real) and not isinstance(item, bool):
            if not math.isfinite(float(item)) or float(item) != int(item):
                raise WeaknessSelectorError(
                    f"extension mass {index} must be a finite integer"
                )
            mass = int(item)
            if mass < 0 or mass > max_extensions:
                raise WeaknessSelectorError(
                    f"extension mass {index} must be in [0, {max_extensions}]"
                )
            masses.append(mass)
            continue
        masses.append(_count_mask(item, candidate=index, max_extensions=max_extensions))
    return tuple(masses)


def select_weakest_tie(
    ranked_candidates: Sequence[dict[str, Any]],
    *,
    tie_tolerance: float = 0.0,
) -> WeaknessSelection:
    """Select the largest explicit extension mass within the baseline tie.

    ``ranked_candidates`` must already be sorted by the baseline score.  Stable
    order is preserved for equal extension masses.  Rejected candidates are
    invalid input to this function; the planner filters them before calling it.
    """
    if not isinstance(ranked_candidates, Sequence) or not ranked_candidates:
        raise WeaknessSelectorError("ranked_candidates must be non-empty")
    if not isinstance(tie_tolerance, Real) or not math.isfinite(float(tie_tolerance)):
        raise WeaknessSelectorError("tie_tolerance must be finite")
    if float(tie_tolerance) < 0.0:
        raise WeaknessSelectorError("tie_tolerance must be non-negative")

    for index, candidate in enumerate(ranked_candidates):
        if not isinstance(candidate, dict):
            raise WeaknessSelectorError(f"candidate {index} must be a mapping")
        if candidate.get("rejected", False):
            raise WeaknessSelectorError("rejected candidates cannot enter weakness selection")
        score = candidate.get("efe")
        mass = candidate.get("extension_mass")
        if not isinstance(score, Real) or not math.isfinite(float(score)):
            raise WeaknessSelectorError(f"candidate {index} has a non-finite efe")
        if not isinstance(mass, Integral) or isinstance(mass, bool) or int(mass) < 0:
            raise WeaknessSelectorError(f"candidate {index} has invalid extension_mass")

    winner_score = float(ranked_candidates[0]["efe"])
    tie_positions = tuple(
        index
        for index, candidate in enumerate(ranked_candidates)
        if abs(float(candidate["efe"]) - winner_score) <= float(tie_tolerance)
    )
    if len(tie_positions) == 1:
        return WeaknessSelection(0, "no_tie", tie_positions, int(ranked_candidates[0]["extension_mass"]))

    selected = max(
        tie_positions,
        key=lambda index: int(ranked_candidates[index]["extension_mass"]),
    )
    masses = [int(ranked_candidates[index]["extension_mass"]) for index in tie_positions]
    status = "tie_equal_mass" if len(set(masses)) == 1 else "selected"
    return WeaknessSelection(
        selected_position=selected,
        status=status,
        tie_positions=tie_positions,
        selected_extension_mass=int(ranked_candidates[selected]["extension_mass"]),
    )


__all__ = [
    "WeaknessSelectorError",
    "WeaknessSelection",
    "normalize_extension_masses",
    "select_weakest_tie",
]
