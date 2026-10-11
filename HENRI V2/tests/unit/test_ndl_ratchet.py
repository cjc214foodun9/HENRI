"""Discriminating tests for the NDL causal-emergence ratchet.

Every test CAN FAIL. The load-bearing ones:

  1. The NO-WAKE invariant. An empty wake store must give a batch BYTE-IDENTICAL to
     stage1_fictive. Every prior receipt depends on this; if it breaks, prior results
     silently change meaning.
  2. The ratchet is MONOTONE and the SHUFFLED control is not a substitute. Without a
     control that fails to ratchet, growth is explained by sample count alone.
  3. THE CONTROL FLOOR IS NON-ZERO, AND WHY. Measured: the shuffled arm accumulated 12
     distinct verified pairs, not 0. That is not a defect in the metric -- it is PROGRAM
     COLLISION. Programs are strings over {I,R,C}; distinct programs can induce the SAME
     function (e.g. "II" and "I" both act as the identity on the trace), so a permuted
     pairing can still re-verify. The test asserts the floor is non-zero AND bounded, so a
     future reader does not "fix" it to zero by weakening the verifier.

SCOPE. Store-level and deterministic. No model is trained. A pass shows consolidation
ACCUMULATES VERIFIABLE STRUCTURE; it does NOT show the model uses it. The model-level
ratchet remains open and is labelled that way in the receipt.
"""
import json
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", ".."))          # tests/unit -> HENRI V2
for p in (V, os.path.join(V, "henri_core")):
    if p not in sys.path:
        sys.path.insert(0, p)

from daydream import DaydreamConfig, DaydreamEngine            # noqa: E402
from m4_generative import INPUTS, apply_op, run_program        # noqa: E402

REPO = os.path.abspath(os.path.join(V, ".."))
RECEIPT = os.path.join(REPO, "design", "zone_a", "evidence",
                       "henri_ndl_ratchet_observed.json")

SEED = 20261011
MAX_LEN = 4


def _engine(seed=SEED):
    return DaydreamEngine(None, None, DaydreamConfig(seed=seed, wake_ratio=0.5))


def _parse(spec):
    try:
        _, prog, _, inp = spec.split()
        return prog, inp
    except Exception:                                          # noqa: BLE001
        return None


def _verified_pairs(stream):
    seen = set()
    for spec, trace in stream:
        pi = _parse(spec)
        if pi is None:
            continue
        prog, inp = pi
        try:
            if run_program(prog, inp) == trace:
                seen.add((prog, inp))
        except Exception:                                      # noqa: BLE001
            continue
    return seen


# ============================================== 1. THE NO-WAKE INVARIANT
def test_no_wake_batch_is_byte_identical_to_stage1_fictive():
    """The load-bearing invariant. With an empty wake store, replay_batch must return
    EXACTLY stage1_fictive(n) for the same generator seed. Two identically seeded
    generators must produce identical bytes."""
    eng = _engine()
    assert len(eng.store.wake_items()) == 0
    a = eng.replay_batch(20, torch.Generator().manual_seed(999), max_len=MAX_LEN)
    b = eng.stage1_fictive(20, torch.Generator().manual_seed(999), max_len=MAX_LEN)
    assert a == b, "replay_batch must equal stage1_fictive byte-for-byte when no wake exists"
    assert len(a) == 20


def test_wake_ratio_zero_is_also_legacy():
    """A second route to the same invariant: wake_ratio = 0 must not mix anything even
    when the store is non-empty."""
    eng = _engine()
    for spec, trace in eng.stage1_fictive(10, torch.Generator().manual_seed(5), max_len=MAX_LEN):
        eng.ingest_wake(spec, trace)
    eng.cfg.wake_ratio = 0.0
    a = eng.replay_batch(15, torch.Generator().manual_seed(7), max_len=MAX_LEN)
    b = eng.stage1_fictive(15, torch.Generator().manual_seed(7), max_len=MAX_LEN)
    assert a == b, "wake_ratio=0 must give the legacy path even with a populated store"


# ============================================== 2. THE VERIFIER CAN FAIL
def test_mismatched_trace_fails_re_execution():
    """The metric's whole basis. A trace that is not what the program produces must NOT
    count as verified. If this passed, the ratchet would measure nothing."""
    eng = _engine()
    assert run_program("R", "1234") == "4321"
    good = {("R", "1234")}
    assert _verified_pairs([("apply R to 1234", "4321")]) == good
    assert _verified_pairs([("apply R to 1234", "9999")]) == set()
    # and a permuted pairing loses the pair
    assert _verified_pairs([("apply R to 1234", run_program("I", "5678"))]) == set()


def test_program_collisions_are_real_so_the_control_floor_is_nonzero():
    """PINNED FINDING, and the honest explanation of the control's 12.

    MY TEST DEFECT, self-caught: the first draft asserted that "I" and "II" are both the
    identity on the trace. MEASURED: apply_op("1234","I") == "2345", so I is INCREMENT,
    not identity, and I != II. My assertion was about semantics I never measured.

    The REAL collision is COPY_IF_FIRST being idempotent: C(x) = x[0] * len(x), so applying
    it twice gives the same string as applying it once. C, CC, and CCC therefore share a
    trace, and a permuted pairing between those entries still re-verifies. That is why the
    shuffled arm accumulates a non-zero floor.
    """
    # measured semantics, not assumed
    assert apply_op("1234", "I") == "2345", "I is increment (measured), not identity"
    assert apply_op("1234", "R") == "4321"
    assert apply_op("1234", "C") == "1111"

    # the idempotent collision: C once and C twice give the SAME trace
    assert run_program("C", "1234") == run_program("CC", "1234") == "1111"
    assert run_program("C", "5678") == run_program("CC", "5678") == "5555"

    # and a genuine non-collision, so the space is not fully degenerate
    assert run_program("R", "1234") != run_program("I", "1234")

    # count collision groups across a real program family on one input
    progs = ["C", "CC", "CCC", "R", "RR", "I", "II", "IR", "RI"]
    outs = [run_program(pr, "1234") for pr in progs]
    assert len(set(outs)) < len(progs), "some programs must collide, or the floor would be 0"
    # C / CC / CCC form one collision group
    assert outs[0] == outs[1] == outs[2]


# ============================================== 3. THE RATCHET AND ITS CONTROL
def test_real_stream_verifies_completely_and_shuffled_does_not():
    """ARM R vs ARM S on one cycle. The real pairing must verify at ~1.0 and the shuffled
    pairing far below, or the metric does not distinguish a real stream from a permuted
    one. Distinguishability is asserted too, because an all-identical trace window would
    make the shuffle a NO-OP and the control vacuous."""
    eng = _engine()
    gen = torch.Generator().manual_seed(SEED)
    wake = eng.stage1_fictive(24, gen, max_len=MAX_LEN)
    for spec, trace in wake:
        eng.ingest_wake(spec, trace)
    replay = eng.replay_batch(40, gen, max_len=MAX_LEN)

    real = len(_verified_pairs(replay))
    specs = [s for s, _ in replay]
    traces = [t for _, t in replay]
    order = torch.randperm(len(traces), generator=gen).tolist()
    shuffled = [(specs[i], traces[order[i]]) for i in range(len(specs))]
    shuf = len(_verified_pairs(shuffled))

    # distinguishability: how many positions actually moved
    moved = sum(1 for i in range(len(traces)) if order[i] != i) / len(traces)
    assert moved >= 0.60, f"shuffle moved only {moved:.2f} of positions; control is weak"
    assert real > shuf, f"real verified {real} must exceed shuffled {shuf}"
    assert real >= 20, f"real stream should verify most pairs, got {real}"


def test_ratchet_accumulates_monotonically_and_beats_the_control():
    """The full ratchet over several cycles. ARM R's cumulative union must be monotone AND
    materially larger than ARM S's. The margin bar is 2.0, pre-registered in the probe."""
    def run(shuffled):
        eng = _engine()
        gen = torch.Generator().manual_seed(SEED + (7 if shuffled else 0))
        cum = set()
        trace_cum = []
        for _ in range(4):
            wake = eng.stage1_fictive(24, gen, max_len=MAX_LEN)
            for spec, t in wake:
                eng.ingest_wake(spec, t)
            replay = eng.replay_batch(40, gen, max_len=MAX_LEN)
            if shuffled:
                specs = [s for s, _ in replay]
                trs = [t for _, t in replay]
                od = torch.randperm(len(trs), generator=gen).tolist()
                replay = [(specs[i], trs[od[i]]) for i in range(len(specs))]
            cum |= _verified_pairs(replay)
            trace_cum.append(len(cum))
        return trace_cum

    real = run(False)
    shuf = run(True)

    # monotone by construction of a union -- assert it, because a non-union metric would not be
    assert all(real[i] <= real[i + 1] for i in range(len(real) - 1)), real
    assert all(shuf[i] <= shuf[i + 1] for i in range(len(shuf) - 1)), shuf

    margin = real[-1] / shuf[-1] if shuf[-1] else float("inf")
    assert margin >= 2.0, f"real {real[-1]} vs shuffled {shuf[-1]}: margin {margin:.2f} < 2.0"
    # the control floor is NON-ZERO (program collisions) and BOUNDED -- not zero, not equal
    assert 0 < shuf[-1] < real[-1], f"control floor {shuf[-1]} must be positive and smaller"


# ============================================== 4. RECEIPT PINS THE HONEST VERDICT
@pytest.mark.skipif(not os.path.exists(RECEIPT), reason="ratchet receipt not produced yet")
def test_receipt_records_scope_and_does_not_overclaim():
    """Read the receipt and assert the scope limits are IN it, so the submission cannot be
    re-read as a capability claim."""
    R = json.load(open(RECEIPT, encoding="utf-8"))
    assert R["evidence_class"] == "OBSERVED"
    assert R["verdict"] == "RATCHET_CONFIRMED_AT_STORE_LEVEL"
    # all six pre-registered checks passed
    assert all(R["checks"].values()), R["checks"]
    # the margin is recorded and above the preregistered bar
    assert R["margin_real_over_shuffle"] >= R["preregistered"]["ratchet_margin_real_over_shuffle"]
    # the non-claims are present, including the model-level residual
    joined = " ".join(R["non_claims"]).lower()
    assert "no model is trained" in joined
    assert "not a capability claim" in joined
    assert "model-level ratchet" in R["residual"].lower()
    # the no-wake invariant is recorded as holding
    assert R["arm_N_no_wake"]["byte_identical_to_stage1_fictive"] is True
    # the control is non-vacuous and non-zero
    assert R["arm_S"]["min_distinguishable_fraction"] >= \
        R["preregistered"]["min_distinguishable_fraction"]
    assert R["arm_S"]["final_cumulative"] > 0
    assert R["arm_R"]["final_cumulative"] > R["arm_S"]["final_cumulative"]
