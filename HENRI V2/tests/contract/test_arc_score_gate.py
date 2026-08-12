"""Gate 2 contract tests: ARC score-eligibility gate (arc_score_gate.py).

Covers missing, corrupt, incompatible, loaded-but-unused, and
loaded-and-active cases. The causal-path audit (2026-08-09) established
that the learned egress component is NOT on the ARC action path; the
loaded-but-unused case must block with
LOADED_COMPONENT_NOT_ON_ACTION_PATH even when all checkpoint fields look
healthy.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "HENRI V2"))

import pytest  # noqa: E402

from arc_score_gate import (  # noqa: E402
    ACTION_HEAD_NOT_CALIBRATED,
    ARC_LEARNED_COMPONENT_ON_ACTION_PATH,
    CHECKPOINT_HASH_MISSING,
    CHECKPOINT_INCOMPATIBLE,
    CHECKPOINT_LOAD_FAILED,
    CHECKPOINT_MISSING,
    ELIGIBLE,
    LOADED_COMPONENT_NOT_ON_ACTION_PATH,
    POLICY_NOT_REQUIRED,
    TRAINED_DECODER_INACTIVE,
    arc_score_eligibility,
)

A = "A" * 64  # fake 64-hex sha


def elig(**kw):
    defaults = dict(
        learned_component_on_action_path=True,
        checkpoint_policy="required",
        checkpoint_load_status="LOADED",
        trained_decoder_active=True,
        checkpoint_sha256=A,
        state_dict_sha256=A,
        trained_action_head_active=True,
    )
    defaults.update(kw)
    return arc_score_eligibility(**defaults)


# --- causal path -------------------------------------------------------------
def test_causal_path_blocks_even_when_loaded():
    # Loaded-and-healthy checkpoint, but component NOT on the action path.
    r = elig(learned_component_on_action_path=False)
    assert r == {"score_eligible": False,
                 "score_block_reason": LOADED_COMPONENT_NOT_ON_ACTION_PATH}


def test_constant_reflects_current_audit():
    # The shipped constant must remain False until a trained component is
    # causally wired into the ARC action path.
    assert ARC_LEARNED_COMPONENT_ON_ACTION_PATH is False


# --- policy / load status ----------------------------------------------------
def test_policy_not_required_blocks():
    r = elig(checkpoint_policy="auto")
    assert r["score_eligible"] is False
    assert r["score_block_reason"] == POLICY_NOT_REQUIRED


def test_missing_checkpoint_blocks():
    r = elig(checkpoint_load_status="SKIPPED_NO_CHECKPOINT")
    assert r["score_block_reason"] == CHECKPOINT_MISSING
    r = elig(checkpoint_load_status=None)
    assert r["score_block_reason"] == CHECKPOINT_MISSING


def test_incompatible_checkpoint_blocks():
    r = elig(checkpoint_load_status="SKIPPED_INCOMPATIBLE_ARCHITECTURE")
    assert r["score_block_reason"] == CHECKPOINT_INCOMPATIBLE


def test_load_failed_blocks():
    r = elig(checkpoint_load_status="LOAD_FAILED")
    assert r["score_block_reason"] == CHECKPOINT_LOAD_FAILED


def test_trained_decoder_inactive_blocks():
    r = elig(trained_decoder_active=False)
    assert r["score_block_reason"] == TRAINED_DECODER_INACTIVE


def test_missing_hashes_block():
    r = elig(checkpoint_sha256=None)
    assert r["score_block_reason"] == CHECKPOINT_HASH_MISSING
    r = elig(state_dict_sha256=None)
    assert r["score_block_reason"] == CHECKPOINT_HASH_MISSING


# --- fully eligible ----------------------------------------------------------
def test_loaded_and_active_eligible():
    r = elig()
    assert r == {"score_eligible": True, "score_block_reason": ELIGIBLE}


# --- block reason priority ---------------------------------------------------
def test_causal_path_priority_over_missing_checkpoint():
    # Even with a missing checkpoint, the causal-path fact is the FIRST
    # violated condition and determines the reason.
    r = elig(learned_component_on_action_path=False,
             checkpoint_load_status=None)
    assert r["score_block_reason"] == LOADED_COMPONENT_NOT_ON_ACTION_PATH


# --- Phase 7.4: semantic action-head dominance ------------------------------
def test_generic_egress_without_action_head_blocks():
    # Phase 7.4 contract: a LOADED generic decoder + egress ON must NEVER
    # independently grant eligibility. All decoder fields look healthy but
    # the calibrated semantic action head is inactive -> blocked.
    r = elig(trained_action_head_active=False)
    assert r == {"score_eligible": False,
                 "score_block_reason": ACTION_HEAD_NOT_CALIBRATED}


def test_generic_egress_off_head_off_blocks():
    # Everything off: blocked with the action-head reason (after decoder
    # checks pass only when the decoder is on the path; with egress OFF the
    # policy check blocks first with POLICY_NOT_REQUIRED).
    r = elig(checkpoint_policy=None, trained_action_head_active=False)
    assert r["score_eligible"] is False
    assert r["score_block_reason"] == POLICY_NOT_REQUIRED


def test_action_head_active_with_full_chain_eligible():
    # Positive control: only a provenance-validated calibrated semantic
    # action head plus a fully healthy decoder chain flips eligibility.
    r = elig()
    assert r == {"score_eligible": True, "score_block_reason": ELIGIBLE}


def test_decoder_hash_priority_over_head():
    # Decoder-provenance checks precede the head check; a missing decoder
    # hash blocks with CHECKPOINT_HASH_MISSING even when the head is OFF.
    r = elig(checkpoint_sha256=None, trained_action_head_active=False)
    assert r["score_block_reason"] == CHECKPOINT_HASH_MISSING
