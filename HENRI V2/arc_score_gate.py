"""ARC score-eligibility gate (Gate 2, deterministic, CPU-testable).

Separates EXECUTION evidence (a run happened) from CAPABILITY evidence
(a learned component on the action-producing path produced the outcome).

Causal-path audit (2026-08-09, commits e6a346c/001fec8): the ARC action
path in production_arc_run.py uses HolographicActionDecoder (random phase
engrams, darwinian_phase_swarm.py:319). HENRIUnifiedEgressTransducer is NOT
instantiated in the ARC runner. A trained checkpoint file on disk is
therefore NOT causally on the action-producing path. Any score emitted
while that remains true must be blocked with
LOADED_COMPONENT_NOT_ON_ACTION_PATH.

The gate never silently suppresses raw environment outcomes; it only
labels them. All traces and scorecards carry score_eligible and
score_block_reason explicitly.
"""

from __future__ import annotations

from typing import Dict, Optional

# Blocked reasons (ordered by priority in arc_score_eligibility).
LOADED_COMPONENT_NOT_ON_ACTION_PATH = "LOADED_COMPONENT_NOT_ON_ACTION_PATH"
POLICY_NOT_REQUIRED = "CHECKPOINT_POLICY_NOT_REQUIRED"
CHECKPOINT_MISSING = "CHECKPOINT_MISSING"
CHECKPOINT_INCOMPATIBLE = "CHECKPOINT_INCOMPATIBLE"
CHECKPOINT_LOAD_FAILED = "CHECKPOINT_LOAD_FAILED"
TRAINED_DECODER_INACTIVE = "TRAINED_DECODER_INACTIVE"
CHECKPOINT_HASH_MISSING = "CHECKPOINT_HASH_MISSING"
ELIGIBLE = ""

# Phase 7.2: a SANS-calibrated action head (self-generated epistemic play)
# proves action-state correlation only, never task semantics. Held-out
# accuracy on the SANS buffer predicts which random action was taken, not
# whether actions achieve task progress. Score eligibility stays blocked
# until external task outcomes (exact_pass / level completions) are observed
# with the head active.
SANS_HEAD_NOT_TASK_VALIDATED = "SANS_HEAD_NOT_TASK_VALIDATED"


# Causal-path audit (2026-08-09, e6a346c/001fec8): the ARC action path in
# production_arc_run.py uses HolographicActionDecoder (random phase engrams,
# darwinian_phase_swarm.py:319). HENRIUnifiedEgressTransducer is NOT
# instantiated in the ARC runner. Set this True ONLY when a trained,
# checkpoint-validated learned component is actually consumed on the
# action-producing path of production_arc_run.py.
ARC_LEARNED_COMPONENT_ON_ACTION_PATH = False


def arc_score_eligibility(
    *,
    learned_component_on_action_path: bool,
    checkpoint_policy: Optional[str] = None,
    checkpoint_load_status: Optional[str] = None,
    trained_decoder_active: bool = False,
    checkpoint_sha256: Optional[str] = None,
    state_dict_sha256: Optional[str] = None,
) -> Dict[str, object]:
    """Return {score_eligible, score_block_reason} for an ARC run.

    Eligibility requires ALL of:
    - learned_component_on_action_path is True (causal path audit)
    - checkpoint_policy == "required"
    - checkpoint_load_status == "LOADED"
    - trained_decoder_active is True
    - checkpoint_sha256 and state_dict_sha256 are present

    The first violated condition determines the block reason.
    """
    if not learned_component_on_action_path:
        return {
            "score_eligible": False,
            "score_block_reason": LOADED_COMPONENT_NOT_ON_ACTION_PATH,
        }
    if checkpoint_policy != "required":
        return {
            "score_eligible": False,
            "score_block_reason": POLICY_NOT_REQUIRED,
        }
    if checkpoint_load_status != "LOADED":
        if checkpoint_load_status in (None, "SKIPPED_NO_CHECKPOINT"):
            reason = CHECKPOINT_MISSING
        elif checkpoint_load_status == "SKIPPED_INCOMPATIBLE_ARCHITECTURE":
            reason = CHECKPOINT_INCOMPATIBLE
        else:
            reason = CHECKPOINT_LOAD_FAILED
        return {"score_eligible": False, "score_block_reason": reason}
    if not trained_decoder_active:
        return {
            "score_eligible": False,
            "score_block_reason": TRAINED_DECODER_INACTIVE,
        }
    if not checkpoint_sha256 or not state_dict_sha256:
        return {
            "score_eligible": False,
            "score_block_reason": CHECKPOINT_HASH_MISSING,
        }
    return {"score_eligible": True, "score_block_reason": ELIGIBLE}
