"""Phase 1 contract tests for the Demonstration Ingress Contract.

Gate contract: SPEC-2026-10-01-PHASE1-TRANSDUCTION
(`experiments/verification/arc_phase1_transduction_prereg.md`).

Verifies the CONTRACT the specification asks for (typed separation + fail
closed) WITHOUT fabricating demonstrations, which the committed
`arc_demo_preflight.py` invariant forbids.
"""

import pytest

from henri.ingress.demo_contract import (
    BLOCKED_ALIASES_EVAL_BUFFER,
    BLOCKED_EMPTY,
    BLOCKED_NO_PROVENANCE,
    SUPPORT_OK,
    DemoIngressContract,
    DemoIngressError,
    assert_no_buffer_aliasing,
    build_from_public_api,
)


def _pairs(n=3):
    return [([[1, 2], [3, 4]], [[4, 3], [2, 1]]) for _ in range(n)]


def test_d1_valid_contract_passes():
    c = DemoIngressContract(support_pairs=_pairs(), eval_budget=100,
                            provenance="task_json.train", task_id="t1")
    assert c.validate() == SUPPORT_OK
    assert c.require_support() == SUPPORT_OK
    assert c.support_pair_count() == 3
    assert c.as_dict()["fabricated"] is False


def test_d2_empty_support_fails_closed():
    """The specification's Gap 2 symptom: an empty context buffer must ABORT."""
    c = DemoIngressContract(support_pairs=[], eval_budget=100,
                            provenance="task_json.train", task_id="t2")
    assert c.validate() == BLOCKED_EMPTY
    with pytest.raises(DemoIngressError, match="DEMO_BLOCKED_EMPTY"):
        c.require_support()


def test_d3_missing_provenance_fails_closed():
    c = DemoIngressContract(support_pairs=_pairs(), eval_budget=10, provenance="")
    assert c.validate() == BLOCKED_NO_PROVENANCE
    with pytest.raises(DemoIngressError):
        c.require_support()


def test_d4_unknown_provenance_is_refused():
    c = DemoIngressContract(support_pairs=_pairs(), eval_budget=10,
                            provenance="made_up_source")
    assert c.validate() == BLOCKED_NO_PROVENANCE


def test_d5_evaluation_buffer_provenance_is_refused():
    """The exact conflation the specification names: support == eval buffer."""
    c = DemoIngressContract(support_pairs=_pairs(), eval_budget=10,
                            provenance="evaluation_buffer")
    assert c.validate() == BLOCKED_ALIASES_EVAL_BUFFER


def test_d6_aliased_buffer_is_detected_by_identity():
    pairs = _pairs()
    c = DemoIngressContract(support_pairs=pairs, eval_budget=10,
                            provenance="task_json.train")
    with pytest.raises(DemoIngressError, match="BLOCKED_ALIASES_EVAL_BUFFER"):
        assert_no_buffer_aliasing(c, pairs)     # the SAME object


def test_d6b_element_wise_alias_is_detected():
    """A copy holding the same objects is still the Gap 2 defect."""
    pairs = _pairs()
    c = DemoIngressContract(support_pairs=list(pairs), eval_budget=10,
                            provenance="task_json.train")
    with pytest.raises(DemoIngressError, match="BLOCKED_ALIASES_EVAL_BUFFER"):
        assert_no_buffer_aliasing(c, pairs)


def test_d6c_distinct_buffer_is_allowed():
    """TAUTOLOGY GUARD: genuinely distinct buffers must NOT be flagged."""
    c = DemoIngressContract(support_pairs=_pairs(3), eval_budget=10,
                            provenance="task_json.train")
    assert_no_buffer_aliasing(c, _pairs(3))     # equal values, distinct objects
    assert_no_buffer_aliasing(c, None)


def test_d7_malformed_pair_is_rejected():
    c = DemoIngressContract(support_pairs=[([[1]],)], eval_budget=1,
                            provenance="task_json.train")
    with pytest.raises(DemoIngressError, match="not an .input, output. pair"):
        c.validate()


def test_d8_eval_budget_is_separate_from_support():
    """The two budgets are distinct typed fields and cannot be conflated."""
    c = DemoIngressContract(support_pairs=_pairs(2), eval_budget=250,
                            provenance="task_json.train")
    d = c.as_dict()
    assert d["support_pair_count"] == 2
    assert d["eval_budget"] == 250
    assert d["support_pair_count"] != d["eval_budget"]


def test_d9_build_from_public_api_reads_only_public_attrs():
    class _Game:
        examples = [{"input": [[1]], "output": [[2]]}]

    c = build_from_public_api(_Game(), task_id="g1", eval_budget=50)
    assert c.validate() == SUPPORT_OK
    assert c.provenance == "game.examples"


def test_d10_build_from_public_api_with_no_demos_stays_blocked():
    """No fabrication: an env without public demos yields BLOCKED, not a fake pair."""
    class _Game:
        pass

    c = build_from_public_api(_Game(), task_id="g2", eval_budget=50)
    assert c.validate() == BLOCKED_NO_PROVENANCE
    assert c.support_pair_count() == 0
    with pytest.raises(DemoIngressError):
        c.require_support()
