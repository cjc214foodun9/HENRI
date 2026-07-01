"""
ground_truth.py  —  PHASE 0

Installs an OBJECTIVE verification gate. Until now the loop's pass/fail
decision came from wave distance to a stored target vector, which (a) was
computed from the tautological interferometer and (b) measures embedding
similarity, not correctness. This module makes correctness the judge.

The wave machinery is NOT discarded — it remains the representation and
routing substrate. But the binary CONVERGED/VETOED decision, and the scalar
reward that drives learning, now come from actually running the candidate
against the task's input/output examples (ARC grids) or unit tests.

Returned object: GroundTruthResult
    passed          : bool   -- all examples solved exactly
    reward          : float  -- graded score in [0, 1] (mean per-example score)
    error_energy    : float  -- 1 - reward, the loss the loop should minimize
    detail          : dict   -- per-example breakdown for telemetry/debugging

Integration contract (used by Phase 0 wiring in cognitive_swarm):
    gt = GroundTruthGate()
    result = gt.evaluate_arc(candidate_code, task_examples, repl)
    # or
    result = gt.evaluate_tests(candidate_code, unit_tests, repl)

`repl` is the existing universal_repl executor exposing
    repl.execute_block(code_str) -> {"success", "stdout", "stderr",
                                     "error_message", ...}
We do NOT introduce a second sandbox; we reuse the project's REPL so all
execution stays inside the existing air-gapped environment.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import json
import ast


@dataclass
class GroundTruthResult:
    passed: bool
    reward: float                      # in [0, 1]
    error_energy: float                # 1 - reward, what the loop minimizes
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        return self.passed


def _grid_cell_accuracy(pred, truth) -> float:
    """
    Graded similarity between two ARC grids (lists of lists of ints).
    Returns fraction of cells that match when shapes are equal; 0.0 if shapes
    differ (wrong output dimensions = no partial credit for cells, but the
    caller can still see it was 'close' via the shape flag in detail).
    """
    if pred is None or truth is None:
        return 0.0
    try:
        if len(pred) != len(truth):
            return 0.0
        total = 0
        correct = 0
        for prow, trow in zip(pred, truth):
            if len(prow) != len(trow):
                return 0.0
            for pc, tc in zip(prow, trow):
                total += 1
                if pc == tc:
                    correct += 1
        if total == 0:
            return 0.0
        return correct / total
    except TypeError:
        # Non-grid output; fall back to exact-equality scoring.
        return 1.0 if pred == truth else 0.0


def _parse_grid_from_stdout(stdout: str):
    """
    The candidate is expected to print its answer grid as JSON (or a Python
    literal) on the LAST non-empty line of stdout. We parse that line.

    This keeps the contract simple and language-agnostic: the model's program
    must `print(json.dumps(answer))` (or print a Python list literal) as its
    final output.
    """
    lines = [ln for ln in stdout.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1].strip()
    # Try JSON first, then a safe Python literal.
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(last)
        except (ValueError, SyntaxError):
            continue
    return None


class GroundTruthGate:
    """
    Objective verifier. Two evaluation modes:

      evaluate_arc(code, examples, repl)
          examples: list of {"input": grid, "output": grid}
          For each example, the candidate program is run with the example's
          `input` grid bound as the global `INPUT_GRID`, and must print its
          predicted output grid as its final stdout line. Per-example score is
          cell accuracy; reward is the mean across examples; passed requires
          every example to be an exact (1.0) match.

      evaluate_tests(code, tests, repl)
          tests: list of {"call": "...", "expect": <value>} OR list of
          assertion strings. reward is fraction of tests passed; passed
          requires all to pass.
    """

    def __init__(self, exact_pass_threshold: float = 1.0):
        # exact_pass_threshold lets you accept >=0.99 as "passed" if you want
        # to tolerate, e.g., a single mislabeled cell. Default demands exact.
        self.exact_pass_threshold = exact_pass_threshold

    # ----- ARC mode --------------------------------------------------------

    def evaluate_arc(self, code: str, examples: List[Dict[str, Any]], repl) -> GroundTruthResult:
        if not examples:
            return GroundTruthResult(
                passed=False, reward=0.0, error_energy=1.0,
                detail={"reason": "no examples supplied; cannot establish ground truth"},
            )

        per_example = []
        scores = []
        for i, ex in enumerate(examples):
            inp = ex.get("input")
            expected = ex.get("output")

            # Bind the input grid, run candidate, then expect the final printed
            # line to be the predicted grid.
            harness = (
                "import json\n"
                f"INPUT_GRID = {json.dumps(inp)}\n"
                "# --- candidate program below ---\n"
                f"{code}\n"
            )
            res = repl.execute_block(harness)

            if not res.get("success", False):
                err = (res.get("error_message") or res.get("stderr") or "").strip()
                per_example.append({
                    "index": i, "score": 0.0, "shape_ok": False,
                    "runtime_error": err[:300],
                })
                scores.append(0.0)
                continue

            pred = _parse_grid_from_stdout(res.get("stdout", ""))
            score = _grid_cell_accuracy(pred, expected)
            shape_ok = (
                pred is not None and expected is not None
                and hasattr(pred, "__len__") and hasattr(expected, "__len__")
                and len(pred) == len(expected)
            )
            per_example.append({
                "index": i, "score": score, "shape_ok": bool(shape_ok),
                "predicted": pred if score < 1.0 else "exact",
            })
            scores.append(score)

        reward = sum(scores) / len(scores)
        passed = all(s >= self.exact_pass_threshold for s in scores)
        return GroundTruthResult(
            passed=passed,
            reward=reward,
            error_energy=max(0.0, 1.0 - reward),
            detail={"mode": "arc", "n_examples": len(examples), "per_example": per_example},
        )

    # ----- Unit-test mode --------------------------------------------------

    def evaluate_tests(self, code: str, tests: List[Any], repl) -> GroundTruthResult:
        if not tests:
            return GroundTruthResult(
                passed=False, reward=0.0, error_energy=1.0,
                detail={"reason": "no tests supplied; cannot establish ground truth"},
            )

        results = []
        passed_count = 0
        for i, t in enumerate(tests):
            if isinstance(t, dict):
                # {"call": "...", "expect": <value>}
                call = t["call"]
                expect = t.get("expect")
                harness = (
                    "import json\n"
                    f"{code}\n"
                    f"__GT_RESULT__ = ({call})\n"
                    "print(json.dumps(__GT_RESULT__, default=str))\n"
                )
                res = repl.execute_block(harness)
                ok = False
                if res.get("success", False):
                    got = _parse_grid_from_stdout(res.get("stdout", ""))
                    ok = (got == expect)
                results.append({"index": i, "passed": ok,
                                 "error": (res.get("error_message") or res.get("stderr") or "")[:200] if not ok else None})
            else:
                # Bare assertion string; passes iff it executes without error.
                harness = f"{code}\nassert ({t})\n"
                res = repl.execute_block(harness)
                ok = bool(res.get("success", False))
                results.append({"index": i, "passed": ok,
                                 "error": (res.get("error_message") or res.get("stderr") or "")[:200] if not ok else None})
            if ok:
                passed_count += 1

        reward = passed_count / len(tests)
        passed = (passed_count == len(tests))
        return GroundTruthResult(
            passed=passed,
            reward=reward,
            error_energy=max(0.0, 1.0 - reward),
            detail={"mode": "tests", "n_tests": len(tests), "results": results},
        )
