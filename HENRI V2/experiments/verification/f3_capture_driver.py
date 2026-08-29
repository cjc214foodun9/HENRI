"""F3 capture driver — per-env bounded attempts (orchestration layer, CPU-only).

Enforces SPEC-2026-08-29-F3-BROAD-BANK section 3 budget BY CONSTRUCTION
(the launcher previously ran the runner once with --steps 500, producing
ACTION6-starved envs and cap overruns):

  - per-env floor:  >= 100 records/env. Retries with a fresh attempt dir +
                    fresh telemetry (HENRI_SINGLE_ENV pins the env), up to
                    --max-attempts (default 5).
  - per-env cap:    each attempt runs the runner with --steps 150; the merge
                    tool additionally trims per-env rows to --env-cap 150.
  - ACTION6 payloads: HENRI_ARC_ACTION_PAYLOADS=1 (sanctioned
                    step_with_payload path; bare ACTION6 raises KeyError 'x'
                    in arcengine — the F3-v1 bp35 crash).
  - N budget:       12 envs x [100, 150] => [1200, 1800] after merge.
  - fail-loud:      an env still below floor after MAX_ATTEMPTS aborts with
                    AssertionError (K3 boundary) — artifacts preserved, never
                    synthesized. Zero-pretraining invariant holds: rows are
                    live authorized captures only.

No model load, no torch in this process: subprocess orchestration + row
accounting. The production runner does the CUDA work per attempt.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from f3_merge_banks import merge_banks

RUNNER = "HENRI V2/production_arc_run.py"
PYTHON = "/venv/main/bin/python"
DEFAULT_ENVS = [
    "lp85-305b61c3", "cd82-fb555c5d", "sk48-d8078629", "ar25-0c556536",
    "ft09-0d8bbf25", "sb26-7fbdac44", "g50t-5849a774", "bp35-0a0ad940",
    "tr87-cd924810", "ka59-38d34dbb", "wa30-ee6fef47", "sc25-635fd71a",
]


def count_env_rows(jsonl_paths: Sequence[Path], env: str) -> int:
    """Count meta rows whose env field equals the target env (exact match)."""
    n = 0
    for p in jsonl_paths:
        for line in p.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(rec.get("env", "?")) == env:
                n += 1
    return n


def attempt_dir(root: Path, env: str, attempt: int) -> Path:
    return root / env / f"attempt_{attempt}"


def derive_seed(base: int, env_index: int, attempt: int) -> int:
    """Deterministic per-(env, attempt) seed pinned in the receipt."""
    return base + env_index * 1000 + attempt


def _run_attempt(
    env: str,
    attempt: int,
    adir: Path,
    steps: int,
    seed: int,
    python: str,
    runner: str,
) -> Dict:
    adir.mkdir(parents=True, exist_ok=True)
    env_map = dict(os.environ)
    env_map.update({
        "HENRI_SINGLE_ENV": env,
        "HENRI_ARC_TRAJECTORY_BANK": "1",
        "HENRI_ARC_ACTION_PAYLOADS": "1",
        "HENRI_SEED": str(seed),
        "HENRI_TELEMETRY_DIR": str(adir),
        "PYTHONPATH": "HENRI V2",
    })
    cmd = [python, runner, "--envs", "1", "--steps", str(steps)]
    log_path = adir / "attempt.log"
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, env=env_map, stdout=logf, stderr=subprocess.STDOUT)
    rc = proc.returncode
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"CAPTURE_ATTEMPT_DONE rc={rc}\n")
    n = count_env_rows(sorted(adir.glob("trajectories_*.jsonl")), env)
    receipt = {
        "env": env,
        "attempt": attempt,
        "rc": rc,
        "seed": seed,
        "rows_env": n,
        "log": str(log_path),
        "seconds": round(time.time() - t0, 1),
    }
    with open(adir / "attempt.json", "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, default=str)
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--envs", nargs="*", default=DEFAULT_ENVS)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--floor", type=int, default=100)
    ap.add_argument("--env-cap", type=int, default=150)
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--python", default=PYTHON)
    ap.add_argument("--runner", default=RUNNER)
    args = ap.parse_args()

    attempts_root = Path(args.attempts_dir)
    attempts_root.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"production_run_{int(time.time())}"
    envs = list(args.envs)
    per_env: Dict[str, Dict] = {}
    blocked: List[str] = []

    for ei, env in enumerate(envs):
        receipts = []
        cum_rows = 0
        for attempt in range(1, args.max_attempts + 1):
            adir = attempt_dir(attempts_root, env, attempt)
            rec = _run_attempt(
                env, attempt, adir, args.steps,
                derive_seed(args.seed, ei, attempt),
                args.python, args.runner,
            )
            receipts.append(rec)
            cum_rows += rec["rows_env"]
            print(f"[driver] {env} attempt {attempt}: rc={rec['rc']} "
                  f"rows_env={rec['rows_env']} cum={cum_rows} "
                  f"({rec['seconds']}s)")
            if cum_rows >= args.floor:
                break
        per_env[env] = {
            "attempts": receipts,
            "rows_total": cum_rows,
            "floor_reached": cum_rows >= args.floor,
        }
        if not per_env[env]["floor_reached"]:
            blocked.append(env)

    if blocked:
        # Fail-loud (K3): preserve everything, emit BLOCKED, exit non-zero.
        driver_receipt = {
            "schema_id": "f3-capture-driver.v1",
            "run_id": run_id,
            "verdict": "BLOCKED_RECORD_FLOOR",
            "blocked_envs": blocked,
            "per_env": per_env,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(attempts_root / "f3_capture_driver_receipt.json", "w",
                  encoding="utf-8") as f:
            json.dump(driver_receipt, f, indent=2, default=str)
        print(json.dumps(driver_receipt, indent=2, default=str))
        print("F3_CAPTURE_DRIVER_BLOCKED")
        sys.exit(2)

    merge = merge_banks(
        str(attempts_root), args.out, run_id,
        env_cap=args.env_cap, expect_envs=envs,
    )
    driver_receipt = {
        "schema_id": "f3-capture-driver.v1",
        "run_id": run_id,
        "verdict": "CAPTURE_OK",
        "per_env": per_env,
        "merged": {
            "record_count": merge["record_count"],
            "envs": merge["envs"],
            "per_env_counts": merge["per_env_counts"],
            "per_action_counts": merge["per_action_counts"],
            "npz_sha256": merge["npz_sha256"],
            "jsonl_sha256": merge["jsonl_sha256"],
        },
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(attempts_root / "f3_capture_driver_receipt.json", "w",
              encoding="utf-8") as f:
        json.dump(driver_receipt, f, indent=2, default=str)
    print(json.dumps(driver_receipt, indent=2, default=str))
    print("F3_CAPTURE_DRIVER_DONE")


if __name__ == "__main__":
    main()
