"""ARC Training Agentic-Graph Orchestrator (Phase 7.9b).

Deterministic control-plane workflow for the HENRI ARC training program.
Dispatches the REAL `production_arc_run.py` SANS-collection subprocess per
target environment, parses the REAL JSONL telemetry, applies fail-closed
gates, verifies persisted calibration artifacts, and seals a
provenance-carrying receipt. It NEVER fabricates rows, deltas, calibration,
or eligibility.

Origin and disposition ("training graph.txt", sha256 f66b396a...):
- The supplied 5-node state-graph shell (Init/Collect/Audit/Calibrate/
  Evaluate) matches the live HENRI contracts for fail-closed multi-arm
  discipline, per-arm receipts, and the eligibility invariant. Adopted.
- The supplied simulated content is REJECTED: hardcoded rows=72 / delta=0.038,
  a time.sleep()-based calibration, the sagnac delta<0.10 gate (falsified
  bound: delta >= 1-1/D on unit vectors, sagnacfunctor.txt, P4-2), and
  score_eligible=True from internal metrics (violates the live semantic
  action-head eligibility dominance rule).

Live contract (OBSERVED on main ba9d78f, 2026-08-12/13):
- Runner: "HENRI V2/production_arc_run.py" --envs 1 --steps 3 from the repo
  root; all HENRI_* env assignments precede the interpreter (launcher-env
  discipline).
- ZONE_C_ENV=prod + ZONE_C_PROD_DSN (from the env file) are required for the
  live telemetry sink; missing -> BLOCKED_INFRASTRUCTURE, never a surrogate.
- The run itself performs SANS calibration (arc_sans_play.py: committed
  buffer -> seeded hold-out split -> AdamW CE -> provenance persist). The
  orchestrator verifies the persisted artifact (existence + sha256); it does
  not train.
- score_eligible=false is an invariant at this stage
  (SANS_HEAD_NOT_TASK_VALIDATED).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic_graph.evidence_receipts import file_hash_receipt

STATUS_PENDING = "PENDING"
STATUS_IN_FLIGHT = "IN_FLIGHT"
STATUS_COLLECTED = "COLLECTED"
STATUS_CALIBRATED = "CALIBRATED"
STATUS_PASSED = "PASSED"
STATUS_KILLED = "KILLED"
STATUS_FAILED = "FAILED"
STATUS_CALIBRATION_FAILED = "CALIBRATION_FAILED"

RECEIPT_SCHEMA = "henri.training-workflow-receipt.v1"


@dataclass
class ArmReceipt:
    """Deterministic per-environment execution + telemetry receipt."""

    env: str
    return_code: int
    driver_log: str = ""
    telemetry_dir: str = ""
    head_path: str = ""
    jsonl_sha256s: List[str] = field(default_factory=list)
    sans_status: Optional[str] = None
    buffer_size: Optional[int] = None
    distinct_labels: Optional[int] = None
    held_out_accuracy: Optional[float] = None
    majority_baseline: Optional[float] = None
    payload_events: int = 0
    eligibility_events: List[Dict[str, Any]] = field(default_factory=list)
    head_sha256: Optional[str] = None
    error: str = ""


def parse_sans_events(telemetry_dir: str) -> Dict[str, Any]:
    """Parse REAL SANS JSONL telemetry from a run's telemetry directory.

    Returns a dict with the LAST SANS_PLAY_RESULT event, all SANS_PLAY
    events, the ARC_ACTION_PAYLOAD event count, and all SCORE_ELIGIBILITY
    events. Unknown lines are skipped; no synthesis.
    """
    out: Dict[str, Any] = {
        "sans_play_result": None,
        "sans_play": [],
        "payload_events": 0,
        "eligibility": [],
    }
    tdir = Path(telemetry_dir)
    if not tdir.is_dir():
        return out
    for p in sorted(tdir.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            et = d.get("event_type")
            if et == "SANS_PLAY_RESULT":
                out["sans_play_result"] = d
            elif et == "SANS_PLAY":
                out["sans_play"].append(d)
            elif et == "ARC_ACTION_PAYLOAD":
                out["payload_events"] += 1
            elif et == "SCORE_ELIGIBILITY":
                out["eligibility"].append(d)
    return out


def _load_env_file(path: str) -> Dict[str, str]:
    """Parse KEY=VALUE lines (optionally 'export '-prefixed)."""
    p = Path(path)
    if not p.is_file():
        return {}
    out: Dict[str, str] = {}
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" in line:
            k, _, v = line.partition("=")
            k = k.strip()
            if k:
                out[k] = v.strip()
    return out


def run_arm(
    env: str,
    *,
    steps: int,
    seed: int,
    python_path: str,
    workdir: str,
    env_file: str,
    out_dir: str,
    timeout: int = 3600,
) -> ArmReceipt:
    """Launch the REAL production_arc_run.py for one environment.

    The env contract mirrors the proven 7.9 launcher: env assignments
    precede the interpreter; per-arm telemetry dir and head path; live Zone C
    via ZONE_C_ENV=prod + the env-file DSN.
    """
    root = Path(workdir)
    runner = root / "HENRI V2" / "production_arc_run.py"
    arm_dir = Path(out_dir) / env
    head_dir = Path(out_dir) / "heads"
    arm_dir.mkdir(parents=True, exist_ok=True)
    head_dir.mkdir(parents=True, exist_ok=True)
    head_path = head_dir / f"{env}.pt"

    wf_env = {
        "ZONE_C_ENV": "prod",
        "PYTHONPATH": "HENRI V2",
        "HENRI_ARC_EGRESS": "1",
        "HENRI_ARC_SANS_PLAY": "1",
        "HENRI_ARC_SANS_STEPS": str(steps),
        "HENRI_ARC_SANS_MODE": "random",
        "HENRI_ARC_ACTION_PAYLOADS": "1",
        "HENRI_ARC_SANS_HEAD_PATH": str(head_path),
        "HENRI_SINGLE_ENV": env,
        "HENRI_SEED": str(seed),
        "HENRI_TELEMETRY_DIR": str(arm_dir),
    }
    env_full = {**os.environ, **_load_env_file(env_file), **wf_env}
    if env_full.get("ZONE_C_ENV") == "prod" and not env_full.get("ZONE_C_PROD_DSN"):
        return ArmReceipt(
            env=env, return_code=-1,
            error="BLOCKED_INFRASTRUCTURE: ZONE_C_ENV=prod but ZONE_C_PROD_DSN "
                  "not set (env file missing or incomplete)",
        )

    cmd = [python_path, str(runner), "--envs", "1", "--steps", "3"]
    log_path = arm_dir / "driver.log"
    try:
        proc = subprocess.run(
            cmd, cwd=str(root), env=env_full,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        log_path.write_text(f"TIMEOUT after {timeout}s\n{exc}", encoding="utf-8")
        return ArmReceipt(env=env, return_code=-1, driver_log=str(log_path),
                          telemetry_dir=str(arm_dir), head_path=str(head_path),
                          error="TIMEOUT")
    log_path.write_text(
        f"RC={proc.returncode}\n--- STDOUT ---\n{proc.stdout}\n--- STDERR ---\n{proc.stderr}",
        encoding="utf-8",
    )

    tele = parse_sans_events(str(arm_dir))
    spr = tele["sans_play_result"] or {}
    arm = ArmReceipt(
        env=env, return_code=proc.returncode,
        driver_log=str(log_path), telemetry_dir=str(arm_dir),
        head_path=str(head_path),
        sans_status=spr.get("status"),
        buffer_size=spr.get("buffer_size"),
        distinct_labels=spr.get("distinct_labels"),
        held_out_accuracy=spr.get("held_out_accuracy"),
        majority_baseline=spr.get("majority_baseline"),
        payload_events=tele["payload_events"],
        eligibility_events=tele["eligibility"],
    )
    for p in sorted(arm_dir.glob("*.jsonl")):
        arm.jsonl_sha256s.append(hashlib.sha256(p.read_bytes()).hexdigest())
    return arm


def verify_head(head_path: str) -> Optional[str]:
    """Return the sha256 of a persisted action-head artifact, or None."""
    p = Path(head_path)
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def gate_workflow(arms: List[ArmReceipt]) -> tuple:
    """Apply the fail-closed gates; return (status, reason)."""
    for a in arms:
        if a.return_code != 0:
            return (STATUS_KILLED,
                    f"BLOCKED_INFRASTRUCTURE: non-zero exit on {a.env} "
                    f"(rc={a.return_code}) {a.error}")
        for ev in a.eligibility_events:
            if ev.get("score_eligible") is not False:
                return (STATUS_FAILED,
                        f"INVARIANT_VIOLATION: score_eligible not false on {a.env}")
        if a.sans_status is None:
            return (STATUS_FAILED,
                    f"MISSING_SANS_TELEMETRY: no SANS_PLAY_RESULT on {a.env}")

    calibrated = []
    for a in arms:
        if a.sans_status == "SANS_HEAD_CALIBRATED":
            sha = verify_head(a.head_path)
            if sha is None:
                return (STATUS_FAILED,
                        f"MISSING_HEAD_ARTIFACT: {a.env} reports "
                        f"SANS_HEAD_CALIBRATED but {a.head_path} absent")
            a.head_sha256 = sha
            calibrated.append(a.env)

    if calibrated:
        return (STATUS_PASSED,
                f"{len(calibrated)} calibrated head(s) persisted and verified "
                f"({', '.join(calibrated)}); score_eligible=false "
                f"(SANS_HEAD_NOT_TASK_VALIDATED)")
    return (STATUS_CALIBRATION_FAILED,
            "no arm reached SANS_HEAD_CALIBRATED with a persisted artifact "
            "(honest negative; collection may still be usable)")


class TrainingGraphWorkflow:
    """Deterministic ARC training workflow: INIT -> COLLECT -> AUDIT ->
    CALIBRATE-VERIFY -> SEAL. No model inference anywhere in the loop."""

    def __init__(self, *, target_envs: List[str], steps: int, seed: int,
                 python_path: str, workdir: str, env_file: str,
                 out_dir: str, timeout: int = 3600):
        self.target_envs = target_envs
        self.steps = steps
        self.seed = seed
        self.python_path = python_path
        self.workdir = workdir
        self.env_file = env_file
        self.out_dir = out_dir
        self.timeout = timeout

    def run(self) -> Dict[str, Any]:
        run_id = f"p79b_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        arms: List[ArmReceipt] = []
        for env in self.target_envs:
            arms.append(run_arm(
                env, steps=self.steps, seed=self.seed,
                python_path=self.python_path, workdir=self.workdir,
                env_file=self.env_file, out_dir=self.out_dir,
                timeout=self.timeout,
            ))
        status, reason = gate_workflow(arms)

        receipts = []
        for a in arms:
            for p in sorted(Path(a.telemetry_dir).glob("*.jsonl")):
                try:
                    receipts.append(asdict(file_hash_receipt(
                        str(p), f"{a.env}-{p.name}",
                    )))
                except Exception:
                    pass
            try:
                receipts.append(asdict(file_hash_receipt(
                    a.driver_log, f"{a.env}-driver",
                )))
            except Exception:
                pass

        receipt = {
            "schema_id": RECEIPT_SCHEMA,
            "run_id": run_id,
            "target_envs": self.target_envs,
            "seed": self.seed,
            "sans_steps_per_env": self.steps,
            "status": status,
            "reason": reason,
            "score_eligible": False,
            "score_block_reason": "SANS_HEAD_NOT_TASK_VALIDATED",
            "arms": [asdict(a) for a in arms],
            "evidence_receipts": receipts,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(self.out_dir) / f"{run_id}_receipt.json"
        out_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        return receipt


def main() -> int:
    parser = argparse.ArgumentParser(
        description="HENRI ARC training agentic-graph workflow orchestrator")
    parser.add_argument("--envs", nargs="+",
                        default=["tu93", "re86", "ls20", "ka59"])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--python", default="/venv/main/bin/python")
    parser.add_argument("--workdir", default=str(Path.cwd()))
    parser.add_argument("--env-file", default="/workspace/zonec_prod.env")
    parser.add_argument("--out-dir", default="/tmp/p79b_workflow")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    wf = TrainingGraphWorkflow(
        target_envs=args.envs, steps=args.steps, seed=args.seed,
        python_path=args.python, workdir=args.workdir,
        env_file=args.env_file, out_dir=args.out_dir, timeout=args.timeout,
    )
    receipt = wf.run()
    print(f"run_id: {receipt['run_id']}")
    print(f"status: {receipt['status']}")
    print(f"reason: {receipt['reason']}")
    print(f"score_eligible: {receipt['score_eligible']}")
    for a in receipt["arms"]:
        print(f"  {a['env']}: rc={a['return_code']} "
              f"status={a['sans_status']} buffer={a['buffer_size']} "
              f"payloads={a['payload_events']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
