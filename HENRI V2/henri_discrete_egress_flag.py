"""HENRI discrete-egress strip flag -- Decision 2, Carrier E6.

WHAT THIS IS
  A single, default-OFF flag that disables the discrete-token egress surfaces of
  Zone A. Decision 2 (approved 2026-09-11) orders the discrete BPE projection
  matrices, top-k logit sorts, and token unbinding heads removed from the live
  path, while preserving byte-identical behaviour when the flag is not set.

FLAG SEMANTICS (explicit, so the default cannot be misread)
  HENRI_STRIP_DISCRETE_EGRESS
      unset / "0"  -> STRIP INERT. Every discrete egress surface constructs
                      exactly as it does today. This is the default and the
                      shipping behaviour. Byte-identity is asserted by test.
      "1" / "true" -> STRIP ACTIVE. Constructing any discrete egress surface
                      raises EgressDiscreteStripEnabledError (fail closed).

  Rationale for fail-closed: a silent None or a zero-length vocabulary would
  propagate a corrupt object into production_arc_run.py:788 and produce a
  fabricated benchmark result. An explicit exception cannot be mistaken for
  output.

WHY THE GUARD SITS IN THE CONSTRUCTORS
  The discrete egress surfaces are reached through 16 live construction sites in
  14 files (AST-measured, receipt e6_d2_ast_sites.json sha 6e4356bfda9bc12c;
  supersedes the earlier and incorrect "8 sites" claim). Guarding the four class
  constructors covers all 16 sites now and every future site, with four bounded
  edits instead of sixteen. The guard is imported lazily inside each constructor
  so no import-time behaviour changes.

NO DELETION. The classes and weights stay in the tree; only live construction is
  gated. Deleting them would break importers that are not on the live path.
"""
from __future__ import annotations

import os

FLAG_ENV = "HENRI_STRIP_DISCRETE_EGRESS"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


class EgressDiscreteStripEnabledError(RuntimeError):
    """Raised when the discrete-egress strip is enabled and a gated surface is built."""


def strip_enabled(env: dict | None = None) -> bool:
    """Return True only when the strip flag is explicitly enabled."""
    src = os.environ if env is None else env
    return str(src.get(FLAG_ENV, "0")).strip().lower() in _TRUTHY


def guard_discrete_egress(surface: str) -> None:
    """Fail closed when the strip is enabled.

    When the flag is unset this function returns None and has no side effect, so
    the enclosing constructor is byte-identical to its pre-patch behaviour.
    """
    if strip_enabled():
        raise EgressDiscreteStripEnabledError(
            f"discrete egress surface {surface!r} is stripped "
            f"({FLAG_ENV}=1); Zone A must emit action vectors or wave egress, "
            f"not discrete token distributions"
        )
