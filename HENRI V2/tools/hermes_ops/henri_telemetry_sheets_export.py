#!/usr/bin/env python3
"""HENRI MBPP telemetry -> Google Sheets bridge.

Reads a run directory (item_results.jsonl + run_evidence.json), computes the
decision-relevant rollup (run-level, per-item, CEGIS inner-loop: attempts,
escalation, winner body), and writes two tabs to Google Sheets via the
google-workspace skill's google_api.py:
  - "run_summary": one row per run (run id, passed, failed, exec errors,
    score_eligible, complexity_lambda, escalated winners/misses, failure
    histogram, mean attempts, top winner bodies).
  - "item_details": one row per item (task_id, pass, attempts, escalated,
    winner body).

Credentials: the skill's setup.py must have completed the one-time OAuth
flow (NO_GWS_CONFIG / NO_GCLOUD_CREDS currently -> BLOCKED with setup path).
The remote run host NEVER holds Google credentials; this exporter runs
locally after telemetry sync.

Usage: python henri_telemetry_sheets_export.py --run-dir <dir> [--label run16b]
"""

import argparse
import collections
import json
import os
import sys
from pathlib import Path


def load_run(run_dir: Path) -> dict:
    rows = []
    with open(run_dir / "item_results.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    evidence = {}
    ev_path = run_dir / "run_evidence.json"
    if ev_path.exists():
        evidence = json.loads(ev_path.read_text(encoding="utf-8"))
    return {"rows": rows, "evidence": evidence}


def summarize(run: dict, label: str) -> dict:
    rows = run["rows"]
    ev = run["evidence"]
    n = len(rows)
    passed = sum(1 for r in rows if r.get("pass"))
    fails = [r for r in rows if not r.get("pass")]
    exec_errors = sum(1 for r in fails if str(r.get("failure_reason", "")).startswith("EXECUTION_ERROR"))
    hist = collections.Counter(str(r.get("failure_reason", ""))[:50] for r in fails)
    tel = [r.get("telemetry", {}) for r in rows]
    escalated_win = sum(1 for t, r in zip(tel, rows) if r.get("pass") and t.get("cegis_escalated"))
    escalated_miss = sum(1 for t, r in zip(tel, rows) if not r.get("pass") and t.get("cegis_escalated"))
    attempts = [t.get("candidates_tried") for t in tel if isinstance(t.get("candidates_tried"), int)]
    bodies = [t.get("body") for t, r in zip(tel, rows) if r.get("pass") and t.get("body")]
    return {
        "run": label,
        "total": n,
        "passed": passed,
        "failed": n - passed,
        "exec_errors": exec_errors,
        "score_eligible": bool(ev.get("score_eligible", ev.get("status") == "OBSERVED")),
        "complexity_lambda": os.environ.get("HENRI_COMPLEXITY_LAMBDA", "default"),
        "escalated_winners": escalated_win,
        "escalated_misses": escalated_miss,
        "mean_attempts": round(sum(attempts) / len(attempts), 2) if attempts else None,
        "failure_histogram": dict(hist.most_common(6)),
        "winner_bodies": bodies[:10],
    }


def item_rows(run: dict) -> list[list]:
    out = []
    for r in run["rows"]:
        t = r.get("telemetry", {})
        out.append([
            r.get("task_id"),
            1 if r.get("pass") else 0,
            t.get("candidates_tried"),
            1 if t.get("cegis_escalated") else 0,
            t.get("body"),
            r.get("failure_reason"),
        ])
    return out


def push_sheets(summary: dict, items: list[list]) -> tuple[str, str]:
    """Write to Google Sheets; returns (status, detail). Raises on missing creds."""
    sys.path.insert(0, r"C:\Users\chan\AppData\Local\hermes\skills\productivity\google-workspace\scripts")
    import google_api

    creds = google_api.get_credentials()
    svc = google_api.build_service("sheets", "v4")
    title = "HENRI MBPP Telemetry"
    body = {"properties": {"title": title}}
    created = svc.spreadsheets().create(body=body).execute()
    sheet_id = created["spreadsheetId"]
    url = created.get("spreadsheetUrl", "")
    headers = [list(summary.keys()), [str(v) for v in summary.values()]]
    svc.spreadsheets().values().update(
        spreadsheetId=sheet_id, range="Sheet1!A1",
        valueInputOption="RAW", body={"values": headers}).execute()
    # item details tab
    svc.spreadsheets().batchUpdate(spreadsheetId=sheet_id, body={
        "requests": [{"addSheet": {"properties": {"title": "item_details"}}}]}).execute()
    item_headers = [["task_id", "pass", "candidates_tried", "cegis_escalated", "body", "failure_reason"]]
    svc.spreadsheets().values().update(
        spreadsheetId=sheet_id, range="item_details!A1",
        valueInputOption="RAW", body={"values": item_headers + items}).execute()
    return sheet_id, url


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--label", default="run")
    args = ap.parse_args()
    run = load_run(Path(args.run_dir))
    summary = summarize(run, args.label)
    print("=== rollup ===")
    for k, v in summary.items():
        if k == "winner_bodies":
            print(f"{k}: {v}")
        elif k == "failure_histogram":
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v}")
    try:
        sheet_id, url = push_sheets(summary, item_rows(run))
        print(f"SHEET_PUSHED id={sheet_id} url={url}")
    except Exception as exc:
        print(f"SHEETS_BLOCKED: {type(exc).__name__}: {exc}")
        print("SETUP_REQUIRED: python 'C:\\Users\\chan\\AppData\\Local\\hermes\\skills\\productivity\\google-workspace\\scripts\\setup.py' (one-time OAuth), then rerun.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
