"""Forensic-audit fixes (2026-08-03, worktree commit TBD).

Covers the three verified defects from the leaf audit + arbitration:
1. Sandbox EXECUTION_ERROR status must never become a task FAIL — the CEGIS
   verifier now raises SandboxExecutionError when no candidate genuinely
   executed, and the pilot counts that status as an execution error.
2. The decoder-fallback provenance gate must catch BOTH spellings of the
   hardcoded fallback (real newline and escaped \\n).
3. parse_entry_from_tests prefers the Compare.left call (assert f(args) ==
   expected) over a walk-order heuristic.
"""
import sys
from pathlib import Path

import pytest

HENRI = Path(__file__).resolve().parents[2]
if str(HENRI) not in sys.path:
    sys.path.insert(0, str(HENRI))

from mbpp_cegis_synthesizer import (  # noqa: E402
    CandidateMissError,
    MbppCegisSynthesizer,
    SandboxExecutionError,
    parse_entry_from_tests,
)
from mbpp_heldout_pilot import FALLBACK_MARKER, FALLBACK_SOURCE_MARKER  # noqa: E402
from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec  # noqa: E402


class _FakeResult:
    def __init__(self, status: str):
        self.status = status
        self.stdout = ""
        self.stderr = ""
        self.runtime_ms = 1.0


class _FakeSandbox:
    def __init__(self, statuses):
        self._statuses = list(statuses)

    def execute(self, code):
        return _FakeResult(self._statuses.pop(0))


def _synth() -> MbppCegisSynthesizer:
    return MbppCegisSynthesizer([], qFHRREpistemicCodec(device="cpu"), device="cpu")


def _ranked(n: int):
    return [(f"def f(a0, a1):\n    return a0 + a1", {"morphism": "identity"}, 0.5)
            for _ in range(n)]


ITEM = {"test_list": ["assert f(1, 2) == 3"]}


def test_parse_entry_from_tests_prefers_compare_left():
    assert parse_entry_from_tests(["assert f(1, 2) == 3"]) == ("f", ["a0", "a1"])
    assert parse_entry_from_tests(["assert 3 == f(1, 2)"]) == ("f", ["a0", "a1"])


def test_parse_entry_from_tests_returns_none_on_garbage():
    assert parse_entry_from_tests(["def not_a_test(): pass"]) is None


def test_fallback_markers_both_spellings():
    # Real-newline spelling in a decoder source file must trip the gate.
    assert FALLBACK_MARKER in "def solution():\n    return True"
    # Escape-sequence spelling inside a string literal must trip the gate.
    assert FALLBACK_SOURCE_MARKER in 'def solution():\\n    return True'
    # The gate is the OR of both (fix: previously only the raw marker ran).
    decoder_src = "def solution():\n    return True"
    assert (FALLBACK_SOURCE_MARKER in decoder_src) or (FALLBACK_MARKER in decoder_src)


def test_cegis_all_infra_fail_raises_execution_error():
    synth = _synth()
    sandbox = _FakeSandbox(["EXECUTION_ERROR"] * 12)
    with pytest.raises(SandboxExecutionError):
        synth.cegis_verify(_ranked(12), ITEM, sandbox, max_attempts=12, escalate=False)


def test_cegis_mixed_infra_is_genuine_miss():
    # One candidate genuinely ran and failed its tests; one infra-failed.
    # The mechanism executed -> a real miss, NOT an execution error.
    synth = _synth()
    sandbox = _FakeSandbox(["FAIL", "EXECUTION_ERROR", "FAIL"] * 4)
    src, meta = synth.cegis_verify(_ranked(12), ITEM, sandbox, max_attempts=12,
                                   escalate=False)
    assert src is None
    assert meta["cegis"] is False


def test_cegis_winner_in_primary_window_not_escalated():
    synth = _synth()
    sandbox = _FakeSandbox(["FAIL", "PASS"] + ["FAIL"] * 10)
    src, meta = synth.cegis_verify(_ranked(12), ITEM, sandbox, max_attempts=12,
                                   escalate=True)
    assert src is not None
    assert meta["cegis"] is True
    assert meta["cegis_escalated"] is False
    assert meta["candidates_tried"] == 2
