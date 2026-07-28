#!/usr/bin/env python3
"""Deterministic state-change router for the HENRI holonic loop (Tier 0).

This module performs ZERO model inference. It reads live state, compares it
with the previous observation, and emits a compact route plan only when a
watched signal changes. When nothing changed it prints nothing and exits 0,
which is the Hermes ``no_agent`` cron silent-watchdog contract.

Purpose in the holonic design
-----------------------------
The failure mode this prevents is a timer-driven model call. A 10-minute cron
that invokes an agent on every tick pays tokens for 144 no-op turns per day.
This router converts the timer into a *poll* and lets a *state change* be the
only trigger for inference.

What this module is NOT
-----------------------
- It is not evidence that a holon ran.
- It is not evidence of HENRI task progress.
- It does not approve anything and does not mutate the repository.
- It does not call TrustGraph, the CUDA target, or any LLM.

It only answers: "what changed, and which bounded route is admissible?"

Signals watched
---------------
================  ==========================================================
signal            source
================  ==========================================================
drive_inbox       G:/My Drive/HENRI_Inbox  (name, size, mtime signature)
git_head          repository HEAD commit
audit_head        Hermes hash-linked ledger head + record count
event_head        local event store: per-stream count + latest event time
vault_server      http://127.0.0.1:8000/health reachability
trustgraph_api    http://127.0.0.1:8088/ reachability
================  ==========================================================

Exit codes: 0 = success (silent or routed). 2 = the router itself failed.
A failed *probe* (server down) is a state value, not a router failure.

CLI
---
    python scripts/henri_state_router.py            # poll, print routes if changed
    python scripts/henri_state_router.py --json     # full machine-readable plan
    python scripts/henri_state_router.py --seal     # also seal a ROUTE_PLANNED event
    python scripts/henri_state_router.py --show     # print current state, no diff
    python scripts/henri_state_router.py --reset    # forget prior state (next poll is a baseline)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

REPO_ROOT = SCRIPT_DIR.parent
GDRIVE_INBOX = Path(r"G:\My Drive\HENRI_Inbox")
STATE_FILE = SCRIPT_DIR / ".henri_router_state.json"
VAULT_SERVER = "http://127.0.0.1:8000"
TRUSTGRAPH_API = "http://127.0.0.1:8088"
AUDIT_LEDGER = (
    Path(os.environ.get("HERMES_HOME", str(Path.home() / "AppData/Local/hermes")))
    / "audit"
    / "henri_audit_chain.jsonl"
)
SUPPORTED_SOURCES = {".pdf", ".md", ".txt"}
PROBE_TIMEOUT = 3.0

# ---------------------------------------------------------------------------
# Route table.
#
# Each entry maps a changed signal to ONE bounded route. `tier` follows the
# four-tier execution cascade: 0 = deterministic collector, 1 = single leaf
# holon, 2 = parent synthesis, 3 = MoA. `approval` states whether a sealed
# human decision is required before the route may mutate code or spend GPU
# time. `silent` marks routes that must not produce a mobile notification.
# ---------------------------------------------------------------------------
ROUTE_TABLE: dict[str, dict[str, Any]] = {
    "drive_inbox.added": {
        "route": "research_leaf",
        "tier": 1,
        "action": "hash source, ingest, project, run one bounded research leaf",
        "approval": False,
        "notify": True,
    },
    "drive_inbox.removed": {
        "route": "none",
        "tier": 0,
        "action": "record source withdrawal; do not re-ingest",
        "approval": False,
        "notify": False,
    },
    "git_head.changed": {
        "route": "remote_verify",
        "tier": 0,
        "action": "poll HENRI CI for this commit; reduce telemetry when complete",
        "approval": False,
        "notify": True,
    },
    "audit_head.extended": {
        "route": "none",
        "tier": 0,
        "action": "ledger grew as expected; no inference",
        "approval": False,
        "notify": False,
    },
    "audit_head.broken": {
        "route": "circuit_breaker",
        "tier": 0,
        "action": "STOP: audit chain failed verification; preserve artifacts",
        "approval": True,
        "notify": True,
    },
    "event_head.changed": {
        "route": "none",
        "tier": 0,
        "action": "event store grew; inspect only if a routed stream changed",
        "approval": False,
        "notify": False,
    },
    "vault_server.down": {
        "route": "degraded",
        "tier": 0,
        "action": "semantic retrieval BLOCKED; local events remain valid",
        "approval": False,
        "notify": False,
    },
    "vault_server.up": {
        "route": "none",
        "tier": 0,
        "action": "semantic retrieval restored",
        "approval": False,
        "notify": False,
    },
    "trustgraph_api.down": {
        "route": "degraded",
        "tier": 0,
        "action": "TrustGraph path BLOCKED; use local event/Chroma path only",
        "approval": False,
        "notify": False,
    },
    "trustgraph_api.up": {
        "route": "none",
        "tier": 0,
        "action": "TrustGraph reachable; still requires a verified flow before use",
        "approval": False,
        "notify": False,
    },
}

# Streams whose growth is decision-relevant enough to route a leaf.
ROUTED_STREAMS = {"research", "claim_audit", "telemetry", "execution"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256_short(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Probes. Each returns a plain-data value. A probe failure is a VALUE, not an
# exception, so a down service produces a route rather than crashing the poll.
# ---------------------------------------------------------------------------

def probe_drive_inbox() -> dict[str, Any]:
    if not GDRIVE_INBOX.exists():
        return {"available": False, "sources": {}}
    sources: dict[str, str] = {}
    for path in sorted(GDRIVE_INBOX.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SOURCES:
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        sources[path.name] = _sha256_short(f"{path.name}::{st.st_size}::{int(st.st_mtime)}")
    return {"available": True, "sources": sources}


def probe_git_head() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15, check=False,
        )
        if proc.returncode != 0:
            return {"available": False, "commit": None}
        return {"available": True, "commit": proc.stdout.strip()}
    except Exception:
        return {"available": False, "commit": None}


def probe_audit_head() -> dict[str, Any]:
    """Read the ledger head directly and verify the hash link.

    The router reads the ledger rather than shelling out so that a poll costs
    one file read. Chain integrity is recomputed here because a broken chain is
    a circuit-breaker condition, not a cosmetic detail.
    """
    if not AUDIT_LEDGER.exists():
        return {"available": False, "count": 0, "head": None, "intact": None}
    records: list[dict[str, Any]] = []
    try:
        with AUDIT_LEDGER.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as exc:
        return {"available": False, "count": 0, "head": None, "intact": False,
                "error": f"{type(exc).__name__}: {exc}"}
    if not records:
        return {"available": True, "count": 0, "head": None, "intact": True}
    prev = "0" * 64
    intact = True
    for i, rec in enumerate(records):
        if rec.get("idx") != i or rec.get("prev_hash") != prev:
            intact = False
            break
        body = (
            f"{rec['idx']}|{rec['ts']}|{rec['actor']}|{rec['action']}|"
            f"{json.dumps(rec['payload'], sort_keys=True)}|{rec['prev_hash']}"
        )
        if hashlib.sha256(body.encode("utf-8")).hexdigest() != rec.get("hash"):
            intact = False
            break
        prev = rec["hash"]
    return {
        "available": True,
        "count": len(records),
        "head": records[-1].get("hash"),
        "intact": intact,
    }


def _vault_root() -> Path:
    configured = os.environ.get("OBSIDIAN_VAULT_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "Documents" / "HENRI_Research_Vault"


def probe_event_head() -> dict[str, Any]:
    events_dir = _vault_root() / "_agentic" / "events"
    if not events_dir.exists():
        return {"available": False, "streams": {}}
    streams: dict[str, dict[str, Any]] = {}
    for path in events_dir.glob("*.json"):
        try:
            event = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        stream = event.get("stream", "unknown")
        entry = streams.setdefault(stream, {"count": 0, "latest": ""})
        entry["count"] += 1
        if event.get("event_time", "") > entry["latest"]:
            entry["latest"] = event["event_time"]
    return {"available": True, "streams": streams}


def _http_reachable(url: str) -> dict[str, Any]:
    """A reachable endpoint that answers 404 is still UP.

    The TrustGraph API gateway returns 404 at its root. Treating that as down
    would produce a false circuit-breaker signal, so any HTTP status counts as
    reachable and only a transport error counts as down.
    """
    try:
        with urllib.request.urlopen(url, timeout=PROBE_TIMEOUT) as response:
            return {"up": True, "status": response.status}
    except urllib.error.HTTPError as exc:
        return {"up": True, "status": exc.code}
    except Exception as exc:
        return {"up": False, "status": None, "error": type(exc).__name__}


def probe_vault_server() -> dict[str, Any]:
    return _http_reachable(f"{VAULT_SERVER}/health")


def probe_trustgraph() -> dict[str, Any]:
    return _http_reachable(f"{TRUSTGRAPH_API}/")


def collect_state() -> dict[str, Any]:
    return {
        "observed_at": _utc_now(),
        "drive_inbox": probe_drive_inbox(),
        "git_head": probe_git_head(),
        "audit_head": probe_audit_head(),
        "event_head": probe_event_head(),
        "vault_server": probe_vault_server(),
        "trustgraph_api": probe_trustgraph(),
    }


# ---------------------------------------------------------------------------
# Deterministic diff.
# ---------------------------------------------------------------------------

def diff_state(previous: dict[str, Any] | None, current: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the list of changed signals. An empty list means a silent tick.

    A missing previous state is a BASELINE: it emits no routes. This prevents a
    first run (or a reset) from routing every existing source as if it were new.
    """
    if not previous:
        return []
    changes: list[dict[str, Any]] = []

    prev_sources = (previous.get("drive_inbox") or {}).get("sources") or {}
    curr_sources = (current.get("drive_inbox") or {}).get("sources") or {}
    added = [n for n, sig in curr_sources.items() if prev_sources.get(n) != sig]
    removed = [n for n in prev_sources if n not in curr_sources]
    if added:
        changes.append({"signal": "drive_inbox.added", "detail": {"sources": sorted(added)}})
    if removed:
        changes.append({"signal": "drive_inbox.removed", "detail": {"sources": sorted(removed)}})

    prev_commit = (previous.get("git_head") or {}).get("commit")
    curr_commit = (current.get("git_head") or {}).get("commit")
    if curr_commit and prev_commit and curr_commit != prev_commit:
        changes.append({
            "signal": "git_head.changed",
            "detail": {"from": prev_commit[:12], "to": curr_commit[:12]},
        })

    prev_audit = previous.get("audit_head") or {}
    curr_audit = current.get("audit_head") or {}
    if curr_audit.get("intact") is False:
        changes.append({
            "signal": "audit_head.broken",
            "detail": {"count": curr_audit.get("count"), "error": curr_audit.get("error")},
        })
    elif curr_audit.get("count", 0) > prev_audit.get("count", 0):
        changes.append({
            "signal": "audit_head.extended",
            "detail": {
                "added": curr_audit.get("count", 0) - prev_audit.get("count", 0),
                "head": (curr_audit.get("head") or "")[:16],
            },
        })

    prev_streams = (previous.get("event_head") or {}).get("streams") or {}
    curr_streams = (current.get("event_head") or {}).get("streams") or {}
    grown = {
        name: data["count"] - (prev_streams.get(name, {}).get("count", 0))
        for name, data in curr_streams.items()
        if data["count"] > prev_streams.get(name, {}).get("count", 0)
    }
    if grown:
        changes.append({
            "signal": "event_head.changed",
            "detail": {"streams": grown, "routed": sorted(set(grown) & ROUTED_STREAMS)},
        })

    for name in ("vault_server", "trustgraph_api"):
        was_up = (previous.get(name) or {}).get("up")
        is_up = (current.get(name) or {}).get("up")
        if was_up is not None and is_up != was_up:
            changes.append({
                "signal": f"{name}.{'up' if is_up else 'down'}",
                "detail": {"status": (current.get(name) or {}).get("status")},
            })
    return changes


def plan_routes(changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map changed signals to bounded routes using the static route table."""
    routes: list[dict[str, Any]] = []
    for change in changes:
        entry = ROUTE_TABLE.get(change["signal"])
        if entry is None:
            routes.append({
                "signal": change["signal"], "detail": change["detail"],
                "route": "unrouted", "tier": 0, "approval": True, "notify": True,
                "action": "unknown signal; a human must classify it before routing",
            })
            continue
        routes.append({**entry, "signal": change["signal"], "detail": change["detail"]})
    return routes


def _load_state() -> dict[str, Any] | None:
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_state(state: dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _seal_route_plan(routes: list[dict[str, Any]]) -> str | None:
    """Seal one ROUTE_PLANNED event. Returns the audit hash, or None if blocked.

    Sealing is opt-in. Routing every silent poll into the ledger would dilute
    the governance record with non-decisions.
    """
    try:
        from agentic_event_store import append_event
    except Exception:
        return None
    try:
        event = append_event(
            "ROUTE_PLANNED",
            {
                "route_count": len(routes),
                "signals": [r["signal"] for r in routes],
                "routes": [r["route"] for r in routes],
                "max_tier": max((r["tier"] for r in routes), default=0),
                "approval_required": any(r["approval"] for r in routes),
            },
            stream="graph",
            actor="henri_state_router",
            causal_status="observed",
            vault_path=str(_vault_root()),
        )
        return event.get("audit_hash")
    except Exception:
        return None


def render_compact(routes: list[dict[str, Any]], seal: str | None) -> str:
    """Render the mobile-executive form: decisions, not a terminal mirror."""
    notify = [r for r in routes if r.get("notify")]
    if not notify:
        return ""
    lines = ["\U0001f7e2 HENRI ROUTE PLAN"]
    for r in notify:
        gate = " [APPROVAL REQUIRED]" if r["approval"] else ""
        lines.append(f"- {r['signal']} -> {r['route']} (tier {r['tier']}){gate}")
        lines.append(f"  {r['action']}")
        detail = json.dumps(r["detail"], sort_keys=True)
        lines.append(f"  detail: {detail[:200]}")
    if seal:
        lines.append(f"SEAL: {seal[:16]}")
    lines.append("EVIDENCE: OBSERVED (state diff only; no model call, no task outcome)")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the full plan as JSON")
    parser.add_argument("--seal", action="store_true", help="seal a ROUTE_PLANNED event")
    parser.add_argument("--show", action="store_true", help="print current state without diffing")
    parser.add_argument("--reset", action="store_true", help="discard stored state and exit")
    args = parser.parse_args()

    if args.reset:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        print(json.dumps({"status": "reset", "state_file": str(STATE_FILE)}))
        return 0

    try:
        current = collect_state()
    except Exception as exc:
        print(json.dumps({"status": "ROUTER_FAILED", "error": f"{type(exc).__name__}: {exc}"}))
        return 2

    if args.show:
        print(json.dumps(current, indent=2, sort_keys=True))
        return 0

    previous = _load_state()
    changes = diff_state(previous, current)
    routes = plan_routes(changes)
    _save_state(current)

    if not routes:
        # Silent healthy tick. Baseline runs land here by design.
        if args.json:
            print(json.dumps({
                "status": "baseline" if previous is None else "no_change",
                "observed_at": current["observed_at"], "routes": [],
            }, indent=2))
        return 0

    seal = _seal_route_plan(routes) if args.seal else None

    if args.json:
        print(json.dumps({
            "status": "routed", "observed_at": current["observed_at"],
            "route_count": len(routes), "routes": routes, "audit_hash": seal,
        }, indent=2, sort_keys=True))
        return 0

    compact = render_compact(routes, seal)
    if compact:
        print(compact)
    return 0


if __name__ == "__main__":
    sys.exit(main())
