"""UHR-05 contract tests — candidate-code assembly (off-by-one repair).

MEASURED DEFECT these tests pin down (own execution, 2026-09-24):

  `s1_scicode_scaffold_runner.py` assembled the candidate payload with

        prior_candidate = [source.produce((problem, k), []) for k in range(idx)]   # EXCLUSIVE

  while the REFERENCE arm on the next lines uses `range(idx + 1)` (INCLUSIVE). Its
  helper then read

        own = prior_outputs[idx] if idx < len(prior_outputs) else ""

  With a list of length `idx`, `idx < len(...)` is ALWAYS FALSE, so `own` silently became
  "" -- and `source.produce` was never called for the current step at all (at idx=0 it was
  never called). The published tests for a sub-step require that sub-step's OWN function to
  be defined, so every scored item failed as
  `CANDIDATE_UNDEFINED_SYMBOL:<the current step's function>` -- a plain required-function
  NameError. That is the measured sub-class on all 15 scored items.

  It is an ASYMMETRY between the two arms, not a design choice: the reference arm includes
  the current step, and it passes for the two attemptable items.

WHY THIS IS A BLOCK ON MEASUREMENT, NOT A RESULT
  With the registered NullCandidateSource the payload is byte-identical either way
  (`deps + "\\n"`), so pass@1 is 0.0 before and after. These tests only pin the addressing.

FALSIFYING CONTROLS
  * `test_falsifying_control_prefix_addressing_excludes_current` asserts the OLD addressing
    does NOT include the current step. If it ever fails, the defect was not what was measured.
  * `test_callsite_source_is_inclusive` is a static guard against regression.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys

import pytest

V2 = pathlib.Path(__file__).resolve().parents[2]
RUNNER = V2 / "experiments" / "verification" / "s1_scicode_scaffold_runner.py"
CORPUS = V2 / "data" / "official_benchmarks" / "scicode" / "problems_dev.jsonl"


@pytest.fixture(scope="module")
def M():
    sys.path.insert(0, str(V2))
    spec = importlib.util.spec_from_file_location("s1runner_t", str(RUNNER))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["s1runner_t"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def probs():
    if not CORPUS.exists():
        pytest.skip("pinned SciCode corpus absent")
    return [json.loads(line) for line in
            CORPUS.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]


def _probe(M):
    """A source emitting EXACTLY ONE distinctive marker per step (no duplication)."""

    class S(M.CandidateSource):
        produces_code = True

        def produce(self, item, prior_outputs):
            prob, k = item
            header = M._dataset_field(prob["sub_steps"][k], "function_header") or ""
            match = re.search(r"def\s+(\w+)", header)
            if match:
                return f"def {match.group(1)}(*a, **k):\n    return None  # PROBE_STEP_{k}\n"
            return f"# PROBE_STEP_{k}\n"

    S.name = "ProbeSource"
    return S()


def test_current_step_code_is_included(M, probs):
    """THE regression gate: the current step's produced code must reach the payload."""
    prob, idx = probs[0], 0
    src = _probe(M)
    prior = [src.produce((prob, k), []) for k in range(idx + 1)]
    code = M.build_candidate_code(prob, idx, prior)
    assert f"PROBE_STEP_{idx}" in code, "current step's code missing from the payload"


def test_prior_step_code_also_included(M, probs):
    prob, idx = probs[0], 2
    src = _probe(M)
    prior = [src.produce((prob, k), []) for k in range(idx + 1)]
    code = M.build_candidate_code(prob, idx, prior)
    for k in range(idx + 1):
        assert f"PROBE_STEP_{k}" in code, f"step {k} missing"


def test_no_duplication_of_current_step(M, probs):
    """`prior` must exclude the current step, or `own` would emit it twice."""
    prob, idx = probs[0], 2
    src = _probe(M)
    prior = [src.produce((prob, k), []) for k in range(idx + 1)]
    code = M.build_candidate_code(prob, idx, prior)
    assert code.count(f"PROBE_STEP_{idx}") == 1, "current step duplicated"
    assert code.count(f"PROBE_STEP_{idx - 1}") == 1, "prior step duplicated"


def test_prior_precedes_current(M, probs):
    prob, idx = probs[0], 3
    src = _probe(M)
    prior = [src.produce((prob, k), []) for k in range(idx + 1)]
    code = M.build_candidate_code(prob, idx, prior)
    assert code.find(f"PROBE_STEP_{idx - 1}") < code.find(f"PROBE_STEP_{idx}")


def test_dependencies_prepended(M, probs):
    prob = probs[0]
    deps = (prob.get("required_dependencies") or "").strip()
    code = M.build_candidate_code(prob, 0, [""])
    assert deps and code.startswith(deps)


def test_negative_control_absent_source_yields_deps_only(M, probs):
    """NULL-CONTROL: an empty source must not crash and must still yield deps."""
    prob = probs[0]
    code = M.build_candidate_code(prob, 0, [""])
    assert "PROBE_STEP" not in code
    assert code.endswith("\n")


def test_falsifying_control_prefix_addressing_excludes_current(M, probs):
    """THE FALSIFIER: the OLD addressing must NOT include the current step."""
    prob, idx = probs[0], 0
    src = _probe(M)
    old_prior = [src.produce((prob, k), []) for k in range(idx)]   # pre-fix addressing
    code = M.build_candidate_code(prob, idx, old_prior)
    assert f"PROBE_STEP_{idx}" not in code, "pre-fix path unexpectedly included the step"


def test_callsite_source_is_inclusive(M):
    """Static guard: the call site must stay inclusive, and keep the [:idx] slice."""
    txt = RUNNER.read_text(encoding="utf-8")
    # Anchor on the CANDIDATE call site specifically. A bare `range(idx + 1)` substring
    # also occurs elsewhere in this file (the reference-arm join), so asserting on that
    # alone would pass for the wrong reason -- a vacuous guard.
    assert ("prior_candidate = [source.produce((problem, k), []) "
            "for k in range(idx + 1)]") in txt, \
        "candidate call site regressed to exclusive range(idx)"
    assert "for p in prior_outputs[:idx]" in txt, "assembly lost the [:idx] slice"
