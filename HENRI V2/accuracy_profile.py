"""HENRI execution-profile contract — accuracy-first fidelity remediation.

Class 4 synthesis (HENRI_Class_4_Semantic_Representation_Synthesis.md,
2026-08-20): abandon latency-first validity. A benchmark run is valid and
score-promotable ONLY under an explicit fidelity profile; latency metrics
are diagnostic telemetry, never validity gates.

Profiles:
- FIDELITY_SCORE_BEARING: full representation, full candidate coverage,
  official evaluator, honest eligibility telemetry. Score promotion allowed.
- DIAGNOSTIC_BALANCED: representation intact, experimental selection
  (e.g. sealed ranking levers enabled). Telemetry only; not promotable.
- PERFORMANCE_ONLY: latency-optimized execution. NEVER score-promotable;
  may reduce coverage, precision, or depth.

Named migration flag (default OFF): HENRI_ACCURACY_FIRST_CLASS4.
When ON, score-bearing runners fail closed unless they run the fidelity
profile. When OFF, legacy behavior is preserved byte-for-byte; this module
only adds telemetry.

Invariants:
- No latency budget may reduce candidate coverage or representation
  fidelity on a score-bearing path.
- Sealed ranking levers (2026-08-20 closure) fail closed under the flag.
- The FALSIFIED ring-mod-256 functor algebra (phase5 p3 KILL c0e3128)
  fails closed under the flag.
- Performance mode is diagnostic-only and can never promote scores.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

FIDELITY_SCORE_BEARING = "FIDELITY_SCORE_BEARING"
DIAGNOSTIC_BALANCED = "DIAGNOSTIC_BALANCED"
PERFORMANCE_ONLY = "PERFORMANCE_ONLY"

SCORE_PROMOTABLE = frozenset({FIDELITY_SCORE_BEARING})

# Named fidelity-migration flag. Default OFF: legacy behavior preserved.
FIDELITY_MIGRATION_FLAG = "HENRI_ACCURACY_FIRST_CLASS4"

# Sealed ranking-lever class (2026-08-20): wave-ranking levers on the
# HumanEval path are CLOSED. Gate A' did not transfer; Gate B stayed at
# baseline 2/50; the bottleneck is grammar expressiveness, not candidate
# order. Reopening requires a new semantic representation and a new
# pre-registered kill gate — never the flag itself.
SEALED_RANKING_LEVERS: Dict[str, str] = {
    "reward_rank": "test-time positive-exemplar re-ranking",
    "decoder_rank": "trained-decoder token-predictability ranking",
    "spec_rank": "docstring-spec target ranking",
    "trained_rank": "execution-grounded correctness-head ranking",
    "ast_idf_only": "IDF-weighted MBPP-codebook ranking",
}


class ExecutionProfileError(RuntimeError):
    """Base error for execution-profile contract violations."""


class FidelityGuardError(ExecutionProfileError):
    """Fail-closed guard fired under the fidelity migration flag."""


class FalsifiedOperatorError(FidelityGuardError):
    """A formally FALSIFIED operator was used on a fidelity-guarded path."""


class MockShortcutRejectedError(FidelityGuardError):
    """A mock/short-circuit output was reached on a fidelity-guarded path."""


def fidelity_migration_enabled() -> bool:
    """True when HENRI_ACCURACY_FIRST_CLASS4=1 (default OFF)."""
    return os.environ.get(FIDELITY_MIGRATION_FLAG, "0") == "1"


def is_score_promotable(profile: Optional[str]) -> bool:
    """Only the fidelity profile may promote scores."""
    return profile in SCORE_PROMOTABLE


def enabled_sealed_levers(levers: Dict[str, bool]) -> List[str]:
    """Return the sealed lever names that are enabled (in declaration order)."""
    return [name for name in SEALED_RANKING_LEVERS if levers.get(name, False)]


def runner_execution_profile(*, sealed_lever_enabled: bool) -> str:
    """Profile for grammar/sandbox runners: fidelity unless a sealed lever
    is active (then diagnostic only)."""
    return DIAGNOSTIC_BALANCED if sealed_lever_enabled else FIDELITY_SCORE_BEARING


def profile_record(
    profile: Optional[str],
    *,
    d_model: Optional[int] = None,
    precision: Optional[str] = None,
    candidate_coverage: Optional[int] = None,
    attempts: Optional[int] = None,
    checkpoint_policy: Optional[str] = None,
    checkpoint_load_status: Optional[str] = None,
    evaluator: Optional[str] = None,
    dataset_sha256: Optional[str] = None,
) -> dict:
    """Compact, decision-relevant profile telemetry for scorecards."""
    return {
        "execution_profile": profile,
        "score_promotable": is_score_promotable(profile),
        "fidelity_migration_flag": fidelity_migration_enabled(),
        "d_model": d_model,
        "precision": precision,
        "candidate_coverage": candidate_coverage,
        "attempts": attempts,
        "checkpoint_policy": checkpoint_policy,
        "checkpoint_load_status": checkpoint_load_status,
        "evaluator": evaluator,
        "dataset_sha256": dataset_sha256,
    }
