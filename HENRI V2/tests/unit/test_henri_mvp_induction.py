"""Discriminating tests for henri_mvp_induction -- structural operator inference.

Every test CAN FAIL. The load-bearing ones pin the two MEASURED LIMITATIONS so a later
reader cannot mistake a substrate-designed result for a general one:
  1. SHIFT(s=1) and AFFINE(a=1,b=1) are the SAME function -> I is NOT uniquely identified.
  2. The hypothesis space spans {I,R,C}, so 1.0000 on this substrate is expected.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", ".."))     # two levels: tests/unit -> HENRI V2
VER = os.path.join(V, "experiments", "verification")
for p in (V, VER):
    if p not in sys.path:
        sys.path.insert(0, p)

import henri_mvp_induction as ind  # noqa: E402


def test_infer_returns_callable_and_meta_in_that_order():
    """The first draft stored (name, fn, meta) and called the tuple element as a function,
    raising TypeError: 'str' object is not callable during composition. Pin the order."""
    best, surv = ind.infer_operator([("1234", "2345")])
    assert best is not None
    fn, meta, name = best
    assert callable(fn), "element 0 must be the callable"
    assert isinstance(meta, dict) and "form" in meta
    assert isinstance(name, str)
    assert fn("1234") == "2345"


def test_shift_and_affine_are_the_same_function():
    """MEASURED LIMITATION 1, pinned. I reports 2 survivors because SHIFT(s=1) and
    AFFINE(a=1,b=1) compute the identical map. Assert the identity so the non-uniqueness
    is visible and no reader mistakes 'survivors==1' for a universal certificate."""
    sh = lambda d: "".join(str((int(c) + 1) % 10) for c in d)
    af = lambda d: "".join(str((1 * int(c) + 1) % 10) for c in d)
    for probe in ("1234", "9012", "0000", "9999", "5005"):
        assert sh(probe) == af(probe), probe
    _, surv = ind.infer_operator([("1234", "2345"), ("5678", "6789")])
    names = sorted(n for _, _, n in surv)
    assert "SHIFT" in names and "AFFINE" in names, names
    assert len(surv) > 1, "I must report more than one survivor; a unique claim would be false"


def test_reverse_is_identified_by_a_position_permutation():
    best, surv = ind.infer_operator([("4312", "2134"), ("1234", "4321")])
    assert best is not None
    assert best[1]["form"] == "PERMUTATION"
    assert best[1]["params"]["perm"] == [3, 2, 1, 0]


def test_copy_first_is_identified_only_by_the_repeat_first_form():
    """C failed with 0 survivors before REPEAT_FIRST was added -- a COVERAGE gap, not an
    inference failure. This test documents that the space must span the operator."""
    best, surv = ind.infer_operator([("1234", "1111"), ("5678", "5555")])
    assert best is not None
    assert best[1]["form"] == "REPEAT_FIRST"
    assert best[0]("9876") == "9999"


def test_non_discriminating_demo_yields_multiple_or_no_survivors():
    """NEGATIVE CONTROL. With inputs 1000..1007 every copy_first output is '1111', so all
    demos are identical and the fit cannot separate REPEAT_FIRST from CONSTANT. Assert the
    ambiguity rather than a false unique identification."""
    pairs = [(f"{1000 + i}", "1111") for i in range(8)]
    _, surv = ind.infer_operator(pairs)
    names = sorted(n for _, _, n in surv)
    assert "CONSTANT" in names and "REPEAT_FIRST" in names, names
    assert len(surv) >= 2, "identical outputs cannot identify a unique form"


def test_no_hypothesis_is_imported_from_the_executor():
    """PURITY CHECK. The module must define its own structural forms; importing apply_op
    or OPS into the hypothesis set would make the inference a library lookup. Assert the
    module does not reference the executor's operator names as hypotheses."""
    src = open(ind.__file__, encoding="utf-8").read()
    for forbidden in ("apply_op(", "OPS[", "m.apply_op"):
        assert forbidden not in src, f"executor reference {forbidden!r} leaks into inference"


def test_hypothesis_space_forms_are_named_structurally():
    names = sorted(n for n, _ in ind.HYPOTHESES)
    assert names == ["AFFINE", "CONSTANT", "IDENTITY", "PERMUTATION",
                     "REPEAT_FIRST", "REPEAT_LAST", "SHIFT"], names
