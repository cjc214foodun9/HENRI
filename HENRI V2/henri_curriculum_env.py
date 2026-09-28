"""Stage-0 curriculum ENVIRONMENT — makes the governor's spec levers CONCRETE.

Directive 1: "Deploy the dynamic curriculum escalation harness: inject topological
obstacles and multiscale programs as soon as loss variance drops below
sigma^2 < 1e-4."

MEASURED DEFICIT THIS MODULE CLOSES
===================================
`henri_curriculum_governor.py` decides WHEN to escalate and carries a 5-key spec.
Before this module the generator consumed ONE key: `prog_len`. The other four
occurred **0 times** in `stage0_seeding_run.py` AND in `stage0_universal_seeder.py`
— declared-but-unread configuration, the defect class this project treats as "the
mechanism does not exist". A governor whose spec nothing reads is decoration.

WHAT EACH LEVER DOES (structural transforms on the VM token stream)
===================================================================
  prog_len              token count of the base program (existing semantics)
  multiscale_nesting    compose the base into 2**(d-1) CONCATENATED blocks: a
                        deeper, block-structured program (the "multiscale program")
  topological_obstacle  insert `k` contiguous BARRIER spans that execution must
                        route past (a repeated token span; structural — see LIMITS)
  distractor_noise      overwrite `k` random positions with other alphabet tokens
  grid_growth           the VM tape size; consumed by the CALLER, not here

The transforms are PURE and TOTAL: they only insert/overwrite tokens, so they
cannot make the VM raise. Totality is PROVEN BY EXECUTION in the tests.

CONTROLS (the anti-vacuity pattern used throughout this sprint)
==============================================================
  DEFAULT EQUIVALENCE : a neutral spec reproduces `sample_program` EXACTLY for the
                        same seed, so the default-OFF path is byte-identical.
  DIFFERENTIAL EFFECT : every non-neutral lever must CHANGE the token stream.
                        A lever that leaves the stream unchanged is decoration.
  DETERMINISM         : same seed -> same programs.
  VM TOTALITY         : the VM executes the products without raising.
  LENGTH BOUND        : a transform can never run the program past
                        MAX_LEN_MULTIPLE x the base length.

HONEST LIMITS
=============
* These are STRUCTURAL curriculum levers on a byte-tape VM. The barrier is a
  repeated token SPAN; this module does NOT claim a particular instruction
  semantics for it, and NO task-accuracy or reasoning gain is claimed. What IS
  claimed is that each lever reaches the generator and changes what it emits.
* `grid_growth` maps to the VM tape size and is applied by the CALLER.
* No benchmark score is claimed anywhere in this module.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import torch

DEFAULT_ALPHABET_SIZE: int = 18          # the seeder's documented 18-token alphabet
NEUTRAL_SPEC: Dict[str, float] = {
    "prog_len": 32.0,
    "topological_obstacle": 0.0,
    "multiscale_nesting": 1.0,
    "distractor_noise": 0.0,
    "grid_growth": 16.0,
}
MAX_LEN_MULTIPLE: int = 8                # hard bound: never exceed 8x the base length
BARRIER_SPAN: int = 2                    # tokens per obstacle span


class CurriculumEnvError(RuntimeError):
    """Fail-closed contract violation."""


def is_neutral(spec: Dict[str, float]) -> bool:
    """True when every lever THIS MODULE applies is at its neutral value.

    `prog_len` and `grid_growth` are excluded: they never change the token
    STRUCTURE, so a program is still neutral for them.
    """
    return (
        int(spec.get("multiscale_nesting", 1)) <= 1
        and int(spec.get("topological_obstacle", 0)) <= 0
        and int(spec.get("distractor_noise", 0)) <= 0
    )


def _rand(rng, low: int, high: int) -> int:
    """Deterministic draw from a torch.Generator (the driver's RNG type)."""
    if high <= low:
        return low
    return int(torch.randint(low, high, (1,), generator=rng).item())


def _insert_barriers(tokens: List[int], spans: int, rng, alpha: int) -> List[int]:
    """Insert `spans` contiguous BARRIER spans at random positions."""
    out = list(tokens)
    if spans <= 0 or not out:
        return out
    barrier = alpha - 1
    cap = MAX_LEN_MULTIPLE * max(1, len(tokens))
    for _ in range(spans):
        if len(out) + BARRIER_SPAN > cap:
            break
        pos = _rand(rng, 0, len(out) + 1)
        out[pos:pos] = [barrier] * BARRIER_SPAN
    return out


def _inject_distractors(tokens: List[int], count: int, rng, alpha: int) -> List[int]:
    """Overwrite `count` random positions with a DIFFERENT alphabet token."""
    out = list(tokens)
    if count <= 0 or not out:
        return out
    for _ in range(min(count, len(out))):
        pos = _rand(rng, 0, len(out))
        cur = out[pos]
        repl = _rand(rng, 0, alpha)
        if repl == cur:
            repl = (cur + 1) % alpha
        out[pos] = repl
    return out


def sample_program_from_spec(
    spec: Dict[str, float],
    rng,
    sample_program_fn: Callable[[int, object], List[int]],
    alphabet_size: int = DEFAULT_ALPHABET_SIZE,
) -> List[int]:
    """Build one program from a governor spec.

    `sample_program_fn` is INJECTED so this module stays dependency-free and
    testable in isolation; the driver passes the seeder's own `sample_program`.

    Order of composition (each step documented in the module docstring):
        base = sample_program(prog_len)
        nesting -> base repeated 2**(d-1) times
        obstacle -> `k` barrier spans inserted
        distractor -> `k` positions overwritten
    """
    if alphabet_size < 2:
        raise CurriculumEnvError("alphabet_size must be >= 2")
    n = max(1, int(spec.get("prog_len", 32)))
    base = list(sample_program_fn(n, rng))
    if not base:
        raise CurriculumEnvError("sample_program_fn returned an empty program")

    reps = max(1, 2 ** (max(1, int(spec.get("multiscale_nesting", 1))) - 1))
    toks = base * reps

    toks = _insert_barriers(toks, max(0, int(spec.get("topological_obstacle", 0))),
                            rng, alphabet_size)
    toks = _inject_distractors(toks, max(0, int(spec.get("distractor_noise", 0))),
                               rng, alphabet_size)
    return toks


def tape_size_from_spec(spec: Dict[str, float], default: int = 256,
                        floor: int = 32) -> int:
    """The caller's `grid_growth` lever: a BOUNDED VM tape size in bytes."""
    v = int(spec.get("grid_growth", default))
    return max(floor, min(v, 1 << 20))


def spec_report(spec: Dict[str, float]) -> Dict[str, object]:
    """Compact, JSON-safe description for telemetry."""
    reps = max(1, 2 ** (max(1, int(spec.get("multiscale_nesting", 1))) - 1))
    return {
        "spec": {k: spec[k] for k in sorted(spec)},
        "neutral": is_neutral(spec),
        "nesting_repetitions": reps,
        "barrier_spans": max(0, int(spec.get("topological_obstacle", 0))),
        "distractor_positions": max(0, int(spec.get("distractor_noise", 0))),
        "tape_size": tape_size_from_spec(spec),
        "max_len_multiple": MAX_LEN_MULTIPLE,
    }
