"""P1 ARC-integrity guards: C2 soft-target wiring, fail-closed egress, trace schema.

Covers the three score-integrity contracts added in the P1 commit:
1. SagnacMCTSPlanner.search() must use the corrected soft-target SGLD
   protocol, never the inert all-zero-label CE path (regression guard).
2. decode_wave_to_response must FAIL CLOSED on the legacy marker branches
   (option/math/generic) unless HENRI_SYNTHETIC_EGRESS=1 is set; marker
   outputs are never score-eligible.
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
