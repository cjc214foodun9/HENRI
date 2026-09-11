"""Deterministic HENRI mobile progress report.

This script reads durable evidence only. It does not call a model and does not
infer task success from internal coherence, tool names, or session text alone.
It is suitable for no-agent cron delivery and for Photon-sized status replies.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / "AppData/Local/hermes")))
HENRI_ROOT = Path(os.environ.get("HENRI_ROOT", r"C:\Users\chan\Desktop\HENRI 7B SWARM"))
AUDIT_SCRIPT = HERMES_HOME / "scripts" / "henri_audit.py"
AUDIT_LEDGER = HERMES_HOME / "audit" / "henri_audit_chain.jsonl"
KANBAN_DB = HERMES_HOME / "kanban.db"
STATE_DB = HERMES_HOME / "state.db"
MAX_FIELD = 280


def _clip(value: Any, limit: int = MAX_FIELD) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _run(*args: str, cwd: Path | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(
            list(args), cwd=str(cwd) if cwd else None,
            capture_output=True, text=True, timeout=15, check=False,
        )
        return p.returncode, (p.stdout or p.stderr or "").strip()
    except Exception as exc:
        return 1, f"{type(exc).__name__}: {exc}"


def _git_state() -> dict[str, str]:
    rc, branch = _run("git", "branch", "--show-current", cwd=HENRI_ROOT)
    rc2, head = _run("git", "log", "-1", "--format=%h %s", cwd=HENRI_ROOT)
    rc3, status = _run("git", "status", "--short", cwd=HENRI_ROOT)
    return {
        "branch": _clip(branch or "UNKNOWN", 120) if rc == 0 else "BLOCKED",
        "head": _clip(head or "UNKNOWN", 180) if rc2 == 0 else "BLOCKED",
        "worktree": "clean" if rc3 == 0 and not status else _clip(status, 220),
    }


def _repo_root_from_state() -> str | None:
    """Read the newest repository root recorded by Hermes, if available."""
    if not STATE_DB.exists():
        return None
    try:
        db = sqlite3.connect(f"file:{STATE_DB}?mode=ro", uri=True)
        row = db.execute(
            "SELECT git_repo_root FROM sessions "
            "WHERE git_repo_root IS NOT NULL AND git_repo_root <> '' "
            "ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        db.close()
        return str(row[0]) if row and row[0] else None
    except Exception:
        return None


def _audit() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not AUDIT_LEDGER.exists():
        return {"status": "BLOCKED", "reason": "audit ledger missing"}, []
    try:
        records = [json.loads(line) for line in AUDIT_LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as exc:
        return {"status": "BLOCKED", "reason": f"ledger unreadable: {exc}"}, []
    if not records:
        return {"status": "BLOCKED", "reason": "audit ledger empty"}, []
    rc, verify = _run(sys.executable, str(AUDIT_SCRIPT), "verify")
    head = records[-1]
    summary = {
        "status": "OBSERVED" if rc == 0 else "BLOCKED",
        "verify": _clip(verify, 180),
        "head_idx": head.get("idx"),
        "head_hash": str(head.get("hash", ""))[:16],
        "last_event": f"{head.get('actor')}::{head.get('action')}",
    }
    return summary, records[-5:]


def _kanban() -> dict[str, Any]:
    if not KANBAN_DB.exists():
        return {"status": "BLOCKED", "reason": "kanban database missing"}
    try:
        db = sqlite3.connect(f"file:{KANBAN_DB}?mode=ro", uri=True)
        rows = db.execute(
            "SELECT id,title,status,assignee,current_step_key,goal_mode "
            "FROM tasks WHERE status NOT IN ('done','archived') "
            "ORDER BY priority DESC, created_at ASC LIMIT 3"
        ).fetchall()
        db.close()
    except Exception as exc:
        return {"status": "BLOCKED", "reason": f"kanban unreadable: {exc}"}
    if not rows:
        return {"status": "OBSERVED", "tasks": [], "reason": "no active task"}
    return {
        "status": "OBSERVED",
        "tasks": [
            {
                "id": row[0], "title": _clip(row[1], 100), "state": row[2],
                "assignee": row[3] or "unassigned", "step": row[4] or "unset",
                "goal_mode": bool(row[5]),
            }
            for row in rows
        ],
    }


def _latest_ci() -> dict[str, str]:
    # Cron history is durable Hermes evidence. Use the CLI output without
    # parsing unbounded logs; a missing row is explicitly reported.
    rc, text = _run("hermes", "cron", "runs", "8027351ab01e", "--limit", "1")
    if rc != 0:
        return {"status": "BLOCKED", "detail": _clip(text, 220)}
    return {"status": "OBSERVED", "detail": _clip(text, 300)}


def build_report() -> dict[str, Any]:
    global HENRI_ROOT
    HENRI_ROOT = Path(_repo_root_from_state() or HENRI_ROOT)
    audit, events = _audit()
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mission": "HENRI development workflow governance",
        "stage": "AUDIT_CHAIN_STATUS",
        "evidence": {
            "audit": audit,
            "recent_events": [
                {"idx": e.get("idx"), "event": f"{e.get('actor')}::{e.get('action')}", "payload": e.get("payload", {})}
                for e in events
            ],
            "git": _git_state(),
            "kanban": _kanban(),
            "ci": _latest_ci(),
        },
        "uncertainty": [
            "This report does not establish external task success unless a real outcome event is present.",
            "Telemetry metrics are not inferred from the audit chain when no telemetry event is sealed.",
        ],
    }


def render(report: dict[str, Any], max_chars: int = 1800) -> str:
    ev = report["evidence"]
    audit = ev["audit"]
    tasks = ev["kanban"].get("tasks", [])
    task = tasks[0] if tasks else None
    lines = [
        "HENRI PROGRESS",
        f"STAGE: {report['stage']}",
        "EVIDENCE: OBSERVED (deterministic collector)",
        f"AUDIT: {audit.get('status')} {audit.get('last_event', 'none')} head={audit.get('head_hash', 'none')}",
        f"CODE: {ev['git']['branch']} {ev['git']['head']} | {ev['git']['worktree']}",
        f"CI: {ev['ci']['status']} {_clip(ev['ci'].get('detail', ''), 240)}",
    ]
    if task:
        lines.append(f"GOAL/TASK: {task['id']} {task['state']} {task['title']} step={task['step']}")
    else:
        lines.append("GOAL/TASK: BLOCKED no active Kanban task")
    if audit.get("status") != "OBSERVED":
        lines.append("BLOCKER: audit chain is not verified")
    lines.append("NEXT: inspect the latest audit event, then approve or block the active task")
    lines.append("UNCERTAINTY: no external outcome is claimed without a sealed outcome event")
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-chars", type=int, default=1800)
    args = parser.parse_args()
    report = build_report()
    print(json.dumps(report, sort_keys=True) if args.json else render(report, args.max_chars))


if __name__ == "__main__":
    main()
