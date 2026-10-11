"""Discriminating tests for the zero-pretraining MVP compiler.

Every test CAN FAIL. The load-bearing ones are the NEGATIVE CONTROLS: a compiler whose
candidate library cannot be wrong is not doing work, and a shuffle control over
non-distinct outputs is a gate that cannot fail.

MEASURED CONTEXT (this session, my own runs):
    henri_mvp_compose.py         held-out accuracy 1.0000 on len-4 programs
    probe_mvp_discrimination.py  min demos for uniqueness {C:1, I:1, R:1}
                                 shuffle control eliminates the true op for all three
    the 1.0000 is TAUTOLOGICAL: the compiled operators ARE apply_op, so once each
    operator is identified, composition reproduces the trace by construction.
"""
import os
import random
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))          # HENRI V2/tests/unit
# PATH FIX (self-caught, same class as before): HERE/".." is `tests`, NOT `HENRI V2`.
# The first version put tests/experiments/verification on sys.path, so the module import
# raised ModuleNotFoundError: No module named 'henri_mvp_compose'. Two levels up is right.
V = os.path.abspath(os.path.join(HERE, "..", ".."))         # HENRI V2
VER = os.path.join(V, "experiments", "verification")
for p in (V, VER):
    if p not in sys.path:
        sys.path.insert(0, p)

import henri_mvp_compose as mvp  # noqa: E402


def test_parse_spec_round_trips_a_well_formed_spec():
    assert mvp.parse_spec("apply IRC to 1234") == ("IRC", "1234")


def test_parse_spec_raises_on_a_malformed_spec():
    with pytest.raises(Exception):
        mvp.parse_spec("no delimiters here")


def test_candidate_library_has_no_duplicate_behaviour():
    """A library containing two identical functions makes identification ambiguous by
    construction. If two names behave identically on every probe input, the compiler can
    never separate them and any claim of unique identification is an artifact."""
    probes = ["1234", "5678", "9012", "2468", "1357", "8642"]
    sig = {}
    for name, fn in mvp.CANDIDATES.items():
        sig.setdefault(tuple(fn(p) for p in probes), []).append(name)
    dupes = {k: v for k, v in sig.items() if len(v) > 1}
    assert dupes == {}, f"duplicate behaviours in CANDIDATES: {dupes}"


def test_compile_identifies_each_single_op_from_its_own_demos():
    demos = {"C": [("1234", "1111"), ("5678", "5555")],
             "I": [("1234", "2345"), ("5678", "6789")],
             "R": [("1234", "4321"), ("5678", "8765")]}
    for op, pairs in demos.items():
        surv = [n for n, f in mvp.CANDIDATES.items()
                if all(f(i) == o for i, o in pairs)]
        assert len(surv) == 1, f"{op} -> {surv} (expected exactly one survivor)"


def test_shuffle_control_eliminates_the_true_operator():
    """NEGATIVE CONTROL. Shuffling (input, output) must destroy the identification.
    If a single survivor remains, the identification is not driven by the demos."""
    pairs = [("1234", "2345"), ("5678", "6789"), ("9012", "0123"), ("2468", "3579")]
    outs = [o for _, o in pairs]
    assert len(set(outs)) > 1, "VACUOUS FIXTURE: non-distinct outputs make shuffling a no-op"

    def survives(sub):
        return [n for n, f in mvp.CANDIDATES.items() if all(f(i) == o for i, o in sub)]

    assert len(survives(pairs)) == 1, "clean demos must identify exactly one operator"
    for sd in (1, 2, 3, 4, 5):
        sh = outs[:]
        random.Random(sd).shuffle(sh)
        sub = [(pairs[i][0], sh[i]) for i in range(len(pairs))]
        assert survives(sub) == [], f"seed {sd}: shuffled demos still identified an operator"


def test_non_distinct_outputs_make_the_shuffle_control_vacuous():
    """THE FIXTURE-POWER DEFECT, pinned. With inputs 1000..1007 every copy_first output is
    '1111', so all outputs are identical and ANY permutation is the identity map. The
    control reports 1 survivor (the true op) because it CANNOT fire. This test asserts the
    vacuity is detectable, so a future reader cannot mistake it for a real FAIL."""
    pairs = [(f"{1000 + i}", "1111") for i in range(8)]
    outs = [o for _, o in pairs]
    assert len(set(outs)) == 1, "precondition: outputs are identical"
    sh = outs[:]
    random.Random(1).shuffle(sh)
    assert sh == outs, "shuffling identical values must be a no-op"
    surv = [n for n, f in mvp.CANDIDATES.items() if all(f(i) == o for i, o in pairs)]
    assert surv == ["copy_first"], (
        "the true operator survives because the permutation changed nothing -- this is "
        "VACUITY, not evidence about the operator")


def test_composition_on_a_length_four_program_is_correct():
    """The MVP's actual claim: operators compiled from LENGTH-1 demos compose correctly on
    a LENGTH-4 program that was never demonstrated."""
    compiled = {"I": "increment", "R": "reverse", "C": "copy_first"}
    prog, inp = "IRCI", "1234"
    out = inp
    for ch in prog:
        out = mvp.CANDIDATES[compiled[ch]](out)
    assert out == mvp.m.run_program(prog, inp), "composition must reproduce the executor"


def test_composition_fails_under_a_wrong_operator_assignment():
    """NEGATIVE CONTROL for the composition claim. If a deliberately wrong assignment still
    reproduced the trace, the accuracy would be insensitive to the compiled semantics and
    the 1.0000 would carry no information."""
    import henri_core.m4_generative as m
    prog, inp = "IRCI", "1234"
    truth = m.run_program(prog, inp)
    wrong = {"I": "reverse", "R": "copy_first", "C": "increment"}
    out = inp
    for ch in prog:
        out = mvp.CANDIDATES[wrong[ch]](out)
    assert out != truth, "a wrong operator assignment must NOT reproduce the trace"


def test_single_demo_is_sufficient_for_c_and_i_and_for_r_with_a_discriminating_input():
    """HONEST LIMITATION, pinned as a test. Measured: with diverse inputs, ONE
    demonstration identifies each operator uniquely, so the candidate library plus a
    single pair does most of the work and the substrate is easy. This test documents the
    measured value rather than letting a later reader infer more than the MVP earned."""
    one = {"C": [("1234", "1111")], "I": [("1234", "2345")], "R": [("4312", "2134")]}
    for op, pairs in one.items():
        surv = [n for n, f in mvp.CANDIDATES.items()
                if all(f(i) == o for i, o in pairs)]
        assert len(surv) == 1, f"{op}: one demo already gives one survivor -> easy substrate"


def test_a_non_discriminating_demo_leaves_reverse_ambiguous():
    """FIXTURE-POWER DEFECT, pinned. The first draft of this file used
    R: ("4321", "1234"). BOTH `reverse` and `sort_asc` map "4321" -> "1234", so that single
    demo could NOT identify R and the assertion failed 2 == 1. The lesson: a demonstration
    discriminates only where the candidate set can disagree. Assert the ambiguity
    explicitly so the fixture's power is visible to the next reader."""
    pairs = [("4321", "1234")]
    surv = sorted(n for n, f in mvp.CANDIDATES.items() if all(f(i) == o for i, o in pairs))
    assert surv == ["reverse", "sort_asc"], surv
    # and a discriminating input resolves it
    pairs2 = [("4312", "2134")]
    surv2 = sorted(n for n, f in mvp.CANDIDATES.items() if all(f(i) == o for i, o in pairs2))
    assert surv2 == ["reverse"], surv2
