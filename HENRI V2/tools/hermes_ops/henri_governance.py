"""HENRI human-governance bridge for Photon and Kanban workflows.

This is a deterministic command-line boundary. It records an explicit human
decision in the hash-linked HENRI audit ledger and optionally appends the same
compact decision to a Kanban task comment. It does not approve commands by
itself and never treats a model response as human approval.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / "AppData/Local/hermes")))
AUDIT_SCRIPT = HERMES_HOME / "scripts" / "henri_audit.py"


def _run(*args: str) -> tuple[int, str]:
    try:
        p = subprocess.run(list(args), capture_output=True, text=True, timeout=20, check=False)
        return p.returncode, (p.stdout or p.stderr or "").strip()
    except Exception as exc:
        return 1, f"{type(exc).__name__}: {exc}"


def _verify_chain() -> str:
    rc, text = _run(sys.executable, str(AUDIT_SCRIPT), "verify")
    if rc != 0:
        raise RuntimeError(f"audit chain verification failed: {text}")
    return text


def record_decision(
    *, task_id: str, decision: str, scope: list[str], assumptions: list[str],
    acceptance: list[str], rejection: list[str], channel: str, reason: str,
) -> dict[str, Any]:
    verify = _verify_chain()
    payload = {
        "task_id": task_id,
        "decision": decision,
        "scope": scope,
        "assumptions": assumptions,
        "acceptance_criteria": acceptance,
        "rejection_criteria": rejection,
        "channel": channel,
        "reason": reason,
        "evidence": "human operator supplied this decision; audit chain verified first",
    }
    rc, output = _run(
        sys.executable, str(AUDIT_SCRIPT), "record", "human_governance",
        "HUMAN_DECISION", json.dumps(payload, sort_keys=True),
    )
    if rc != 0:
        raise RuntimeError(output)
    return {"decision": payload, "chain_verification": verify, "seal": output}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("decision", choices=["approve", "reject", "block"])
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--scope", action="append", default=[])
    parser.add_argument("--assumption", action="append", default=[])
    parser.add_argument("--accept", action="append", default=[])
    parser.add_argument("--reject", action="append", default=[])
    parser.add_argument("--channel", default="photon")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    result = record_decision(
        task_id=args.task_id,
        decision={"approve": "APPROVE", "reject": "REJECT", "block": "BLOCK"}[args.decision],
        scope=args.scope,
        assumptions=args.assumption,
        acceptance=args.accept,
        rejection=args.reject,
        channel=args.channel,
        reason=args.reason,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
