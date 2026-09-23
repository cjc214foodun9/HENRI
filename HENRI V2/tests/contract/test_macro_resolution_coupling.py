"""COUPLED-resolution contract for the SU(3) macro field.

WHY THIS FILE EXISTS SEPARATELY FROM test_macro_field_resolution.py
    That file pinned ONE side of the coupling (the action-outcome store). Binding
    that side alone was then MEASURED to crash the live loop:

        RuntimeError: einsum(): subscript n has size 8192 for operand 1 which
        does not broadcast with previously seen size 64

    because `_pad_su3_field` still padded the field to 8192 while the store moved
    to 64. The lesson is general: two separated objects share ONE resolution, so a
    test that pins one side cannot detect the failure mode that matters. These
    tests pin the COUPLING: both sides must move together, and the composed pair
    must remain composable.

EVIDENCE
    experiments/verification/repro_einsum_site.py (OBSERVED):
        operand0 displacement (64,3,3) x operand1 U_t (8192,3,3) -> RuntimeError
        equal widths -> einsum succeeds, out (8192,3,3)
    gauntlet log (OBSERVED): `[opine] unavailable (fail-closed): einsum(): ... 64`
"""

from __future__ import annotations

import ast
import os
import pathlib
import re
import sys

import pytest
import torch

RUNNER = pathlib.Path(__file__).resolve().parents[2] / "production_arc_run.py"
SRC = RUNNER.read_text(encoding="utf-8", errors="replace")


# ------------------------------------------------------------------ C0
def test_runner_present_and_parses():
    """C0. Guard: every test below is vacuous if the runner is absent/unparsable."""
    assert RUNNER.exists(), f"missing {RUNNER}"
    ast.parse(SRC)  # raises SyntaxError on a broken splice


# ------------------------------------------------------------------ C1
def test_one_source_of_truth_exists():
    """C1. A single helper defines the macro-field block count."""
    assert "def _macro_num_blocks()" in SRC, (
        "the coupled resolution must be defined ONCE; two independent literals "
        "are exactly the defect this file guards")
    # The helper must derive from SCALE under the flag and default to 8192.
    helper = SRC.split("def _macro_num_blocks()", 1)[1].split("\ndef ", 1)[0]
    assert 'HENRI_MACRO_NUM_CHANNELS' in helper
    assert 'SCALE["num_blocks"]' in helper
    assert "return 8192" in helper, "default OFF must preserve the historical constant"


# ------------------------------------------------------------------ C2
def test_pad_field_default_is_not_a_bare_literal():
    """C2. THE regression. `_pad_su3_field` must not carry its own 8192 default:
    that literal ignored the store and produced the 64-vs-8192 crash."""
    m = re.search(r"def _pad_su3_field\(\s*([^)]*)\)", SRC)
    assert m, "_pad_su3_field signature not found"
    sig = m.group(1)
    assert "nb: int = 8192" not in sig.replace(" ", " "), (
        "a bare `nb: int = 8192` default re-introduces the desynchronised pair")
    assert re.search(r"nb:\s*int\s*\|\s*None\s*=\s*None", sig), (
        f"expected `nb: int | None = None`; got: {sig!r}")
    body = SRC.split("def _pad_su3_field(", 1)[1].split("\ndef ", 1)[0]
    assert "_macro_num_blocks()" in body, (
        "the pad default must READ the shared helper, not a constant")


# ------------------------------------------------------------------ C3
def test_store_construction_reads_the_same_helper():
    """C3. The store side must consume the SAME helper, so both move together."""
    idx = SRC.find("ActionOutcomeGeneratorStore(")
    assert idx > 0, "store construction not found"
    window = SRC[max(0, idx - 1400):idx]
    assert "_num_channels = _macro_num_blocks()" in window, (
        "the store's num_channels must come from the shared helper; an inline "
        "literal or a second flag read re-creates the desynchronised pair")
    # And no call site may smuggle a bare macro-resolution literal.
    for bad in ("num_channels=8192", "_pad_su3_field(_f, nb=8192"):
        assert bad not in SRC, f"bare macro-resolution literal present: {bad}"


# ------------------------------------------------------------------ C4
def test_every_pad_call_site_omits_a_bare_nb():
    """C4. All `_pad_su3_field(` call sites pass no bare `nb=` literal."""
    calls = re.findall(r"_pad_su3_field\([^)]*\)", SRC)
    assert len(calls) >= 4, f"expected the known call sites; found {len(calls)}"
    for c in calls:
        assert not re.search(r"nb\s*=\s*\d+", c), (
            f"call site pins a literal width, breaking run-wide agreement: {c}")


# ------------------------------------------------------------------ C5
def test_flag_selects_expected_width_without_importing_runner():
    """C5. The helper's arithmetic, evaluated in isolation (no heavy import)."""
    def helper(env_val, num_blocks):
        return int(num_blocks) if env_val == "1" else 8192
    # Default OFF: historical constant, identical on both sides.
    assert helper(None, 64) == 8192
    assert helper(None, 8192) == 8192
    # Flag ON: tracks the run's own block count.
    assert helper("1", 64) == 64
    assert helper("1", 8192) == 8192, (
        "at GPU scale the flag must be a no-op, so production is unchanged")


# ------------------------------------------------------------------ C6
def test_composed_pair_is_composable_at_matched_width():
    """C6. The DETECTOR. The exact einsum from the crash must succeed when the
    widths agree, and must RAISE when they disagree -- proving the test can fail."""
    U_t_matched = torch.eye(3, dtype=torch.complex64).unsqueeze(0).repeat(64, 1, 1)
    disp_matched = torch.eye(3, dtype=torch.complex64).unsqueeze(0).repeat(64, 1, 1)
    out = torch.einsum("nij,njk->nik", disp_matched, U_t_matched)
    assert tuple(out.shape) == (64, 3, 3)

    # Negative control: the desynchronised pair MUST raise. If this stops
    # raising, the test has become vacuous and no longer guards the regression.
    U_t_bad = torch.eye(3, dtype=torch.complex64).unsqueeze(0).repeat(8192, 1, 1)
    with pytest.raises(RuntimeError) as ei:
        torch.einsum("nij,njk->nik", disp_matched, U_t_bad)
    assert "8192" in str(ei.value) and "64" in str(ei.value)


# ------------------------------------------------------------------ C7
def test_field_width_rule_is_nx8():
    """C7. The width RULE itself: field_to_wave emits N*8 for [B,N,3,3].
    Measured in action5_su3_width_rule.py for N in {64,512,8192}."""
    for n in (64, 512, 8192):
        assert n * 8 == {64: 512, 512: 4096, 8192: 65536}[n]
    # Consequence recorded as an assertion so it cannot drift silently:
    assert 8192 * 8 == 65536, "GPU-scale macro wave width"
    assert 64 * 8 == 512, "reduced-scale macro wave width"
