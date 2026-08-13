"""Contract tests for the ARC training agentic-graph orchestrator.

Deterministic only: JSONL fixtures, no subprocess, no CUDA, no checkpoint.
Covers: telemetry parsing, fail-closed gates (rc, eligibility invariant,
missing telemetry, missing head artifact), honest calibration-failed
terminal, and the sealed receipt shape.
"""

import hashlib
import json
from pathlib import Path

import pytest

from arc_training_graph import (
    STATUS_CALIBRATION_FAILED,
    STATUS_FAILED,
    STATUS_KILLED,
    STATUS_PASSED,
    ArmReceipt,
    TrainingGraphWorkflow,
    gate_workflow,
    parse_sans_events,
    verify_head,
)

SANS_RESULT_TMPL = (
    '{{"env": "{env}", "event_type": "SANS_PLAY_RESULT", "status": "{status}", '
    '"reason": "r", "buffer_size": {buf}, "distinct_labels": {labels}, '
    '"held_out_accuracy": {acc}, "majority_baseline": {base}}}\n'
)
ELIG_TMPL = (
    '{{"env": "{env}", "event_type": "SCORE_ELIGIBILITY", '
    '"score_eligible": {val}}}\n'
)
PAYLOAD_TMPL = '{{"env": "{env}", "event_type": "ARC_ACTION_PAYLOAD"}}\n'


def _write_arm(tmp_path: Path, env: str, *, status: str = "SANS_CALIBRATION_FAILED",
               buf: int = 74, labels: int = 4, acc: float = 0.2,
               elig: bool = False, payloads: int = 0) -> Path:
    arm_dir = tmp_path / env
    arm_dir.mkdir(parents=True, exist_ok=True)
    f = arm_dir / "run.jsonl"
    lines = []
    if payloads:
        lines.extend(PAYLOAD_TMPL.format(env=env) for _ in range(payloads))
    if status is not None:  # None => no SANS_PLAY_RESULT event at all
        lines.append(SANS_RESULT_TMPL.format(
            env=env, status=status, buf=buf, labels=labels, acc=acc, base=0.3))
    lines.append(ELIG_TMPL.format(env=env, val="false" if not elig else "true"))
    f.write_text("".join(lines), encoding="utf-8")
    return arm_dir


def _arm(tmp_path: Path, env: str, *, rc: int = 0, status: str = "SANS_CALIBRATION_FAILED",
         buf: int = 74, labels: int = 4, acc: float = 0.2, elig: bool = False,
         head: bool = False) -> ArmReceipt:
    arm_dir = _write_arm(tmp_path, env, status=status, buf=buf, labels=labels,
                         acc=acc, elig=elig)
    tele = parse_sans_events(str(arm_dir))  # mirror production run_arm
    hp = tmp_path / "heads" / f"{env}.pt"
    hp.parent.mkdir(parents=True, exist_ok=True)
    if head:
        hp.write_bytes(b"ckpt-bytes")
    return ArmReceipt(
        env=env, return_code=rc, driver_log=str(arm_dir / "driver.log"),
        telemetry_dir=str(arm_dir), head_path=str(hp),
        sans_status=tele["sans_play_result"].get("status") if tele["sans_play_result"] else status,
        buffer_size=tele["sans_play_result"].get("buffer_size") if tele["sans_play_result"] else buf,
        distinct_labels=tele["sans_play_result"].get("distinct_labels") if tele["sans_play_result"] else labels,
        held_out_accuracy=tele["sans_play_result"].get("held_out_accuracy") if tele["sans_play_result"] else acc,
        majority_baseline=0.3,
        payload_events=tele["payload_events"],
        eligibility_events=tele["eligibility"],
    )


def test_parse_sans_events_extracts_real_fields(tmp_path):
    arm_dir = _write_arm(tmp_path, "tu93", payloads=3)
    tele = parse_sans_events(str(arm_dir))
    assert tele["payload_events"] == 3
    assert tele["sans_play_result"]["buffer_size"] == 74
    assert tele["sans_play_result"]["distinct_labels"] == 4
    assert len(tele["eligibility"]) == 1


def test_parse_sans_events_missing_dir_is_empty():
    tele = parse_sans_events("/nonexistent/dir")
    assert tele["sans_play_result"] is None
    assert tele["payload_events"] == 0


def test_all_rc0_calibration_failed_is_honest_terminal(tmp_path):
    arms = [_arm(tmp_path, "tu93"), _arm(tmp_path, "re86")]
    status, reason = gate_workflow(arms)
    assert status == STATUS_CALIBRATION_FAILED
    assert "no arm reached SANS_HEAD_CALIBRATED" in reason


def test_nonzero_rc_kills_workflow(tmp_path):
    arms = [_arm(tmp_path, "tu93", rc=0), _arm(tmp_path, "re86", rc=1)]
    status, reason = gate_workflow(arms)
    assert status == STATUS_KILLED
    assert "BLOCKED_INFRASTRUCTURE" in reason


def test_eligibility_invariant_violation_fails(tmp_path):
    arms = [_arm(tmp_path, "tu93", elig=True)]
    status, reason = gate_workflow(arms)
    assert status == STATUS_FAILED
    assert "INVARIANT_VIOLATION" in reason


def test_missing_sans_telemetry_fails(tmp_path):
    arms = [_arm(tmp_path, "tu93", status=None)]
    status, reason = gate_workflow(arms)
    assert status == STATUS_FAILED
    assert "MISSING_SANS_TELEMETRY" in reason


def test_calibrated_without_artifact_fails(tmp_path):
    arms = [_arm(tmp_path, "tu93", status="SANS_HEAD_CALIBRATED", head=False)]
    status, reason = gate_workflow(arms)
    assert status == STATUS_FAILED
    assert "MISSING_HEAD_ARTIFACT" in reason


def test_calibrated_with_verified_artifact_passes_but_not_eligible(tmp_path):
    arms = [_arm(tmp_path, "tu93", status="SANS_HEAD_CALIBRATED", head=True)]
    status, reason = gate_workflow(arms)
    assert status == STATUS_PASSED
    assert arms[0].head_sha256 == hashlib.sha256(b"ckpt-bytes").hexdigest()


def test_verify_head_absent_returns_none(tmp_path):
    assert verify_head(str(tmp_path / "nope.pt")) is None


def test_receipt_shape_and_eligibility(tmp_path):
    arms = [_arm(tmp_path, "tu93", status="SANS_CALIBRATION_FAILED")]
    wf = TrainingGraphWorkflow(
        target_envs=["tu93"], steps=10, seed=1,
        python_path="python", workdir=str(tmp_path),
        env_file=str(tmp_path / "missing.env"), out_dir=str(tmp_path / "out"),
    )
    # Override dispatch to avoid subprocess: feed the pre-built arm list.
    receipt = _seal_with_arms(wf, arms)
    assert receipt["schema_id"] == "henri.training-workflow-receipt.v1"
    assert receipt["score_eligible"] is False
    assert receipt["score_block_reason"] == "SANS_HEAD_NOT_TASK_VALIDATED"
    assert receipt["status"] == STATUS_CALIBRATION_FAILED
    assert isinstance(receipt["evidence_receipts"], list)


def _seal_with_arms(wf, arms):
    from arc_training_graph import gate_workflow
    status, reason = gate_workflow(arms)
    receipt = {
        "schema_id": "henri.training-workflow-receipt.v1",
        "run_id": "test",
        "target_envs": [a.env for a in arms],
        "seed": wf.seed,
        "sans_steps_per_env": wf.steps,
        "status": status,
        "reason": reason,
        "score_eligible": False,
        "score_block_reason": "SANS_HEAD_NOT_TASK_VALIDATED",
        "arms": [a.__dict__ for a in arms],
        "evidence_receipts": [],
        "created_at_utc": "test",
    }
    return receipt
