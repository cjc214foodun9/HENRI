"""Read-only HENRI audit-and-task context collector.

This collector reduces durable HENRI development state without a model and
without mutating code, Kanban, graph events, remote execution, or telemetry.
Missing or malformed sources are reported as BLOCKED. It is suitable for a
no-agent cron job or a bounded Photon status request.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from agentic_event_store import verify_local_events  # noqa: E402

HERMES_HOME = Path(
    os.environ.get("HERMES_HOME", str(Path.home() / "AppData/Local/hermes"))
)
HENRI_ROOT = Path(
    os.environ.get("HENRI_ROOT", r"C:\Users\chan\Desktop\HENRI 7B SWARM")
)
VAULT_ROOT = Path(
    os.environ.get(
        "OBSIDIAN_VAULT_PATH",
        r"G:\My Drive\HENRI_Research_Vault\HENRI_Research_Vault",
    )
)
AUDIT_SCRIPT = HERMES_HOME / "scripts" / "henri_audit.py"
AUDIT_LEDGER = HERMES_HOME / "audit" / "henri_audit_chain.jsonl"
GRAPH_CLI = HENRI_ROOT / "scripts" / "agentic_graph_cli.py"
GRAPH_PROJECTION = VAULT_ROOT / "_agentic" / "graph_projection.json"
KANBAN_DB = HERMES_HOME / "kanban.db"
STATE_DB = HERMES_HOME / "state.db"
CI_JOB_ID = os.environ.get("HENRI_CI_JOB_ID", "8027351ab01e")
VAULT_HEALTH_URL = os.environ.get("HENRI_VAULT_HEALTH_URL", "http://127.0.0.1:8000/health")
MAX_DETAIL = 260


def _clip(value: Any, limit: int = MAX_DETAIL) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _run(*args: str, cwd: Path | None = None, timeout: int = 15) -> tuple[int, str]:
    try:
        result = subprocess.run(
            list(args),
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return result.returncode, (result.stdout or result.stderr or "").strip()
    except Exception as exc:
        return 1, f"{type(exc).__name__}: {exc}"


def _git() -> dict[str, Any]:
    if not HENRI_ROOT.exists():
        return {"status": "BLOCKED", "detail": f"repository missing: {HENRI_ROOT}"}
    rc_branch, branch = _run("git", "branch", "--show-current", cwd=HENRI_ROOT)
    rc_head, head = _run("git", "log", "-1", "--format=%h %s", cwd=HENRI_ROOT)
    rc_status, status = _run("git", "status", "--short", cwd=HENRI_ROOT)
    if rc_branch or rc_head or rc_status:
        return {"status": "BLOCKED", "detail": _clip(branch or head or status)}
    return {
        "status": "OBSERVED",
        "branch": _clip(branch, 100),
        "head": _clip(head, 180),
        "worktree": "clean" if not status else _clip(status, 220),
    }


def _audit() -> dict[str, Any]:
    if not AUDIT_LEDGER.exists():
        return {"status": "BLOCKED", "detail": f"ledger missing: {AUDIT_LEDGER}"}
    rc, output = _run(sys.executable, str(AUDIT_SCRIPT), "verify")
    try:
        records = [
            json.loads(line)
            for line in AUDIT_LEDGER.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception as exc:
        return {"status": "BLOCKED", "detail": f"ledger unreadable: {exc}"}
    if rc != 0 or not records:
        return {"status": "BLOCKED", "detail": _clip(output or "audit verification failed")}
    head = records[-1]
    return {
        "status": "OBSERVED",
        "record_count": len(records),
        "head_hash": str(head.get("hash", ""))[:16],
        "last_event": f"{head.get('actor')}::{head.get('action')}",
        "verify": _clip(output, 180),
    }


def _graph() -> dict[str, Any]:
    if not GRAPH_PROJECTION.exists():
        return {"status": "BLOCKED", "detail": f"projection missing: {GRAPH_PROJECTION}"}
    try:
        projection = json.loads(GRAPH_PROJECTION.read_text(encoding="utf-8"))
        if not isinstance(projection, dict):
            raise ValueError("projection is not an object")
        valid, validation = verify_local_events(VAULT_ROOT)
        event_count = len(list((VAULT_ROOT / "_agentic" / "events").glob("*.json")))
        if not valid:
            return {"status": "BLOCKED", "detail": _clip(validation)}
        return {
            "status": "OBSERVED",
            "event_count": event_count,
            "node_count": projection.get("node_count", 0),
            "edge_count": projection.get("edge_count", 0),
            "projection_hash": str(projection.get("projection_hash", ""))[:16],
            "validation": _clip(validation, 160),
        }
    except Exception as exc:
        return {"status": "BLOCKED", "detail": f"projection unreadable: {exc}"}


def _kanban(task_id: str) -> dict[str, Any]:
    if not KANBAN_DB.exists():
        return {"status": "BLOCKED", "detail": f"Kanban DB missing: {KANBAN_DB}"}
    try:
        db = sqlite3.connect(f"file:{KANBAN_DB}?mode=ro", uri=True)
        row = db.execute(
            "SELECT id,title,status,assignee,current_step_key,goal_mode "
            "FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        db.close()
    except Exception as exc:
        return {"status": "BLOCKED", "detail": f"Kanban unreadable: {exc}"}
    if not row:
        return {"status": "BLOCKED", "detail": f"task not found: {task_id}"}
    return {
        "status": "OBSERVED",
        "task_id": row[0],
        "title": _clip(row[1], 120),
        "state": row[2],
        "assignee": row[3] or "unassigned",
        "step": row[4] or "unset",
        "goal_mode": bool(row[5]),
    }


def _ci() -> dict[str, Any]:
    rc, output = _run("hermes", "cron", "runs", CI_JOB_ID, "--limit", "1")
    if rc != 0:
        return {"status": "BLOCKED", "detail": _clip(output)}
    return {"status": "OBSERVED", "job_id": CI_JOB_ID, "detail": _clip(output, 300)}


def _vault() -> dict[str, Any]:
    try:
        with urllib.request.urlopen(VAULT_HEALTH_URL, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("health response is not an object")
        return {
            "status": "OBSERVED" if payload.get("status") == "ok" else "BLOCKED",
            "indexed_chunks": payload.get("indexed_chunks"),
            "event_store": _clip(payload.get("local_event_store"), 160),
        }
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "BLOCKED", "detail": f"vault health unavailable: {exc}"}
    except Exception as exc:
        return {"status": "BLOCKED", "detail": f"vault health error: {type(exc).__name__}: {exc}"}


def _telemetry() -> dict[str, Any]:
    candidates = [
        HENRI_ROOT / "telemetry_summary.json",
        HENRI_ROOT / "telemetry" / "latest_summary.json",
        HENRI_ROOT / "telemetry_logs" / "latest_summary.json",
    ]
    for path in candidates:
        if path.exists():
            return {"status": "OBSERVED", "artifact": str(path)}
    return {"status": "BLOCKED", "detail": "no approved telemetry summary found"}


def build_report(task_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "collector": "henri_agentic_context_collector",
        "scope": "read_only",
        "evidence": {
            "audit": _audit(),
            "graph": _graph(),
            "kanban": _kanban(task_id),
            "git": _git(),
            "ci": _ci(),
            "telemetry": _telemetry(),
            "vault": _vault(),
        },
        "uncertainty": [
            "This collector does not establish external task success.",
            "Telemetry is not inferred when no approved summary exists.",
            "The collector does not mutate code, Kanban, graph events, or remote execution state.",
        ],
        "next_action": "Inspect the evidence, then approve or block the active task.",
    }


def render(report: dict[str, Any], max_chars: int = 1800, review: bool = False) -> str:
    ev = report["evidence"]
    audit = ev["audit"]
    graph = ev["graph"]
    kanban = ev["kanban"]
    ci = ev["ci"]
    vault = ev["vault"]
    lines = [
        "HENRI AGENTIC CONTEXT",
        f"SCOPE: OBSERVED read-only deterministic collector{' | MANUAL GOVERNANCE REVIEW' if review else ''}",
        f"AUDIT: {audit.get('status')} records={audit.get('record_count', '?')} head={audit.get('head_hash', 'none')}",
        f"GRAPH: {graph.get('status')} nodes={graph.get('node_count', '?')} edges={graph.get('edge_count', '?')} hash={graph.get('projection_hash', 'none')}",
        f"TASK: {kanban.get('status')} {kanban.get('task_id', '?')} {kanban.get('state', '?')} step={kanban.get('step', '?')}",
        f"CODE: {ev['git'].get('status')} {ev['git'].get('branch', '')} {ev['git'].get('head', '')} | {ev['git'].get('worktree', '')}",
        f"CI: {ci.get('status')} {_clip(ci.get('detail', ''), 220)}",
        f"VAULT: {vault.get('status')} chunks={vault.get('indexed_chunks', '?')}",
        f"TELEMETRY: {ev['telemetry'].get('status')} {_clip(ev['telemetry'].get('detail', ''), 160)}",
        "NEXT: inspect the evidence, then approve or block the active task",
        "UNCERTAINTY: no external outcome is claimed without a sealed outcome event",
    ]
    if review:
        lines.insert(-2, "DECISION: approve or block only after inspecting the listed evidence")
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", default="t_a27660c4")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--review",
        action="store_true",
        help="mark this invocation as a manual HENRI governance review",
    )
    parser.add_argument("--max-chars", type=int, default=1800)
    args = parser.parse_args()
    report = build_report(args.task_id)
    print(
        json.dumps(report, ensure_ascii=False, sort_keys=True)
        if args.json
        else render(report, args.max_chars, review=args.review)
    )


if __name__ == "__main__":
    main()
