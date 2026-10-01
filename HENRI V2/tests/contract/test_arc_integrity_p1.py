"""P1 ARC-integrity guards: C2 soft-target wiring, fail-closed egress, trace schema.

Covers the three score-integrity contracts added in the P1 commit:
1. SagnacMCTSPlanner.search() must use the corrected soft-target SGLD
   protocol, never the inert all-zero-label CE path (regression guard).
2. decode_wave_to_response must FAIL CLOSED on the legacy marker branches
   (option/math/generic) unless HENRI_SYNTHETIC_EGRESS=1 is set; marker
   outputs are never score-eligible. The python/code branch is covered by the
   SAME contract (added with STEP 2): it reaches decode_autoregressive_sequence,
   whose code vocabulary is a 10-token canned stub map and whose grammar mask
   leaves every out-of-vocab id selectable beyond the 3-token prefix.
3. ARCEpisodeTrace validates schema and rejects pre-existing task-specific
   persistence (unseen-task governance).
"""

import inspect

import pytest
import torch

import sagnac_mcts_planner as sp
from henri_benchmark_registry import ARCEpisodeTrace
from henri_decoder import DecoderEgressFailClosedError, HENRIUnifiedEgressTransducer


def test_planner_no_all_zero_label_wart():
    """Regression guard: search() must not resurrect the C1-class inert
    all-zero bootstrap labels or the CE-only adapt_in_context call."""
    src = inspect.getsource(sp)
    assert "demo_token_ids = [0]" not in src, (
        "all-zero bootstrap labels must not return"
    )
    assert "adapt_in_context_sgld_wave" in src, (
        "corrected soft-target SGLD must be wired"
    )
    assert "self.decoder.adapt_in_context(demo_waves" not in src, (
        "CE-only inert call must not return"
    )


def test_synthetic_egress_fails_closed_by_default():
    """Production default: marker egress raises DecoderEgressFailClosedError."""
    trans = HENRIUnifiedEgressTransducer(
        d_model=64, hidden_dim=16, vocab_size=64, device="cpu",
        checkpoint_policy="disabled",
    )
    wave = torch.randn(64)
    with pytest.raises(DecoderEgressFailClosedError):
        trans.decode_wave_to_response(wave, "Please solve the math problem.")


def _transducer():
    return HENRIUnifiedEgressTransducer(
        d_model=64, hidden_dim=16, vocab_size=64, device="cpu",
        checkpoint_policy="disabled",
    )


# STEP 2 contract (HENRI-ARCH-2026-CRITICAL-DIRECTIVE-V1): the python/code branch
# must be gated by the SAME flag as its three siblings, not closed incidentally.
#
# MEASURED BEFORE THE FIX: with the flag unset the branch raised
# DecoderEgressFailClosedError with "out-of-vocab token id N" -- i.e. it was
# closed only because argmax happened to land outside a 10-token canned code map.
# With the flag SET it raised the same way (so the branch was unreachable even for
# synthetic fixtures), and it never set score_eligible/synthetic_marker at all.
# This test pins the FLAG GATE, so the incidental guard cannot stand in for it.
import os as _os  # noqa: E402

PY_PROMPT = "Write a python function to add two numbers."


def test_python_egress_fails_closed_by_the_flag_not_incidentally(monkeypatch):
    """The python/code branch must raise on the FLAG GATE, like its siblings.

    Discriminating assertion: the exception message must be the flag-gate text.
    An "out-of-vocab" message means the branch is only accidentally closed and
    would emit a canned `def solution(): return <literal>` stub for some wave
    state -- with no ineligibility marking.
    """
    monkeypatch.delenv("HENRI_SYNTHETIC_EGRESS", raising=False)
    trans = _transducer()
    wave = torch.randn(64)
    with pytest.raises(DecoderEgressFailClosedError) as ei:
        trans.decode_wave_to_response(wave, PY_PROMPT)
    msg = str(ei.value)
    assert "disabled by default" in msg, (
        "python/code branch is not gated by HENRI_SYNTHETIC_EGRESS; it raised "
        f"{msg!r} instead of the flag-gate error"
    )


def test_python_egress_output_is_never_score_eligible(monkeypatch):
    """Invariant that must hold whether or not the branch returns.

    Two acceptable outcomes with the flag SET: fail closed (out-of-vocab), or
    return marked synthetic+ineligible. A score-eligible python stub is forbidden:
    `decode_autoregressive_sequence` itself refuses to emit a stub solution
    ("a synthetic `return True` solution would fabricate task outcomes").
    """
    monkeypatch.setenv("HENRI_SYNTHETIC_EGRESS", "1")
    trans = _transducer()
    wave = torch.randn(64)
    try:
        _text, telem = trans.decode_wave_to_response(wave, PY_PROMPT)
    except DecoderEgressFailClosedError:
        return  # acceptable fail-closed outcome
    assert telem.get("synthetic_marker") is True
    assert telem.get("score_eligible") is False


def test_synthetic_egress_flag_marks_ineligible(monkeypatch):
    """With HENRI_SYNTHETIC_EGRESS=1 the marker branch runs but is marked
    score-ineligible."""
    monkeypatch.setenv("HENRI_SYNTHETIC_EGRESS", "1")
    trans = HENRIUnifiedEgressTransducer(
        d_model=64, hidden_dim=16, vocab_size=64, device="cpu",
        checkpoint_policy="disabled",
    )
    wave = torch.randn(64)
    text, telem = trans.decode_wave_to_response(
        wave, "What is the correct option letter?"
    )
    assert telem.get("synthetic_marker") is True
    assert telem.get("score_eligible") is False
    assert "correct option" in text


def _valid_trace_kwargs(**overrides):
    kwargs = {
        "schema_id": "henri.arc-episode-trace.v1",
        "episode_id": "env-abc-1",
        "commit_sha256": "0" * 64,
        "task_input_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "split_id": "arcade-public-seen",
        "task_specific_persistence_preexisting": False,
        "demo_pair_count": 2,
        "candidate_count": 10,
        "veto_count": 0,
        "evaluator_reached": True,
        "evaluator_status": "BUDGET_EXHAUSTED",
    }
    kwargs.update(overrides)
    return kwargs


def test_arc_episode_trace_validates():
    trace = ARCEpisodeTrace(**_valid_trace_kwargs())
    assert trace.schema_id == "henri.arc-episode-trace.v1"
    assert trace.model_dump()["commit_sha256"] == "0" * 64


def test_arc_episode_trace_rejects_preexisting_state():
    with pytest.raises(ValueError):
        ARCEpisodeTrace(**_valid_trace_kwargs(task_specific_persistence_preexisting=True))
