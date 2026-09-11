#!/usr/bin/env python3
"""henri_sync_manifest.py — declarative HENRI artifact topology + drift verifier.

Answers one falsifiable question: does the four-surface topology (local disk,
GitHub, Google Drive, Vast.ai) actually obey its declared rules RIGHT NOW?

Verbs:
  show     print the manifest
  verify   check the live repo against the manifest and report drift

Deterministic. Read-only. No network, no LLM.
Evidence class: OBSERVED (filesystem/git) -> DERIVED (drift verdict).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM")

# ---------------------------------------------------------------- manifest
# Surface roles. One writer per path class. Nothing is mirrored "everywhere".
MANIFEST = {
    "version": 1,
    "updated": "2026-09-11",
    "surfaces": {
        "github": {
            "role": "canonical versioned code, skills, experiments, docs",
            "authoritative_for": ["HENRI V2/**/*.py", "HENRI V2/docs/**",
                                  "experiments/**", "tests/**", "docs/**",
                                  "docker/**", "scripts/**",
                                  "HENRI V2/agentic_graph/**/*.py",
                                  "HENRI V2/agentic_graph/**/*.ttl"],
            "never_holds": ["*.pt", "*.zip", "*.pdf", "2605.26340",
                            "artificial-analysis-openapi.yaml",
                            "*.jsonl telemetry", "raw scorecard bodies",
                            "*.pt checkpoints", "Obsidian_Vault/**",
                            ".vault_vector_db/**"],
        },
        "local": {
            "role": "dev workspace, Obsidian vault, scratch; not authoritative",
            "authoritative_for": [],
            "never_holds": [],
        },
        "drive": {
            "role": "papers, vault projection, human-readable digests, figures",
            "authoritative_for": ["HENRI_Inbox/**", "HENRI_Research_Vault/**",
                                  "HENRI_Telemetry/**"],
            "access": "junctions: HENRI V2/Drive_Inbox, Drive_Research_Vault, Drive_Telemetry",
        },
        "vast": {
            "role": "ephemeral compute ONLY; must hold no unique state",
            "authoritative_for": [],
            "never_holds": ["unpushed source edits", "unpulled telemetry"],
            "egress_path": "capture -> scripts/henri_capture_to_repo.py -> commit",
        },
    },
    "flows": [
        {"from": "local", "to": "github", "trigger": "commit",
         "paths": ["HENRI V2/**", "experiments/**", "docs/**"]},
        {"from": "local", "to": "vast", "trigger": "experiment launch",
         "paths": ["HENRI V2/**"], "mechanism": "mutagen sync or git"},
        {"from": "vast", "to": "github", "trigger": "run complete",
         "paths": ["experiments/**", "telemetry summaries"],
         "mechanism": "henri_capture_to_repo.py -> commit (receipts, not raw blobs)"},
        {"from": "drive", "to": "local", "trigger": "research ingest",
         "paths": ["HENRI_Inbox/**"]},
        {"from": "local", "to": "drive", "trigger": "digest complete",
         "paths": ["figures/**", "*.md digests"]},
    ],
    "invariants": [
        "No binary >1 MiB (*.pt/*.zip/*.pdf) is git-tracked.",
        "No file over 1 MiB is inlined into agent context; reference by path.",
        "The live audit ledger is %LOCALAPPDATA%/hermes/audit/henri_audit_chain.jsonl; "
        "repo-root henri_audit_chain.json files are stale v0 genesis artifacts.",
        "Each path class has exactly one writer surface.",
    ],
    "context_budget": {
        "raw_inline_limit_bytes": 1_000_000,
        "rule": "files above this are path-referenced after deterministic reduction",
        "known_offenders": {
            "experiments/verification/mmlu_839_scorecard.json": 5_940_600,
            "experiments/verification/gpqa_839_scorecard.json": 84_011,
            "HENRI V2/telemetry_logs/vast_ai_sync_20260719_203821.jsonl": 741_076,
        },
    },
}


def sh(*args: str) -> str:
    try:
        r = subprocess.run(args, cwd=str(REPO), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=120)
        return r.stdout.strip()
    except Exception as e:
        return f"<error: {e}>"


def git_tracked(pattern: str) -> list[str]:
    out = sh("git", "ls-files", pattern)
    return [l for l in out.splitlines() if l.strip()] if out else []


def verify() -> int:
    fails: list[str] = []
    warns: list[str] = []
    ok: list[str] = []

    print("=" * 78)
    print("HENRI SYNC TOPOLOGY VERIFY  (OBSERVED git/filesystem -> DERIVED verdict)")
    print(f"repo={REPO}")
    print("=" * 78)

    if not (REPO / ".git").exists():
        print("FAIL: not a git repository")
        return 2

    print(f"\ngit branch: {sh('git','branch','--show-current')}")

    # 1. binary blobs must not be tracked
    print("\n[1] binary blobs git-tracked? (manifest: NO)")
    bad = []
    for pat in ("*.pt", "*.zip", "*.pdf", "*.m4a", "*.mp4"):
        t = git_tracked(pat)
        if t:
            bad += t
            print(f"    VIOLATION {pat}: {len(t)} tracked -> {t[:3]}")
        else:
            print(f"    ok  {pat}: 0 tracked")
    if bad:
        fails.append(f"{len(bad)} binary blob(s) tracked")

    # 2. known large context offenders and how they are handled
    print("\n[2] large-artifact handling (manifest: path-reference, never inline)")
    for rel, size in MANIFEST["context_budget"]["known_offenders"].items():
        p = REPO / rel
        if not p.exists():
            print(f"    absent  {rel}")
            continue
        real = p.stat().st_size
        tracked = bool(git_tracked(rel))
        verdict = "tracked" if tracked else "untracked"
        flag = "REVIEW" if tracked and real > MANIFEST["context_budget"]["raw_inline_limit_bytes"] else "ok"
        print(f"    {flag:6s} {rel}  {real:,} B  {verdict}")
        if flag == "REVIEW":
            warns.append(f"{rel} tracked at {real:,} B (> inline limit)")

    # 3. sync-mechanism sprawl
    print("\n[3] sync mechanisms present (manifest target: one egress path)")
    mechs = []
    for pat in ("*sync*", "*scp*", "*mutagen*", "*capture_to_repo*"):
        for f in REPO.rglob(pat):
            s = str(f).replace("\\", "/")
            if "/.git/" in s or "/trustgraph/" in s or "node_modules" in s:
                continue
            if f.is_file() and f.suffix in (".py", ".sh", ".ps1"):
                mechs.append(s.replace(str(REPO).replace("\\", "/") + "/", ""))
    for m in sorted(set(mechs)):
        print(f"    {m}")
    if len(set(mechs)) > 3:
        warns.append(f"{len(set(mechs))} sync/capture mechanisms -> consolidate")

    # 4. audit ledger locations (duplicate/stale ledgers)
    print("\n[4] audit ledger locations (manifest: ONE live ledger)")
    live = Path(os.path.expandvars(r"%LOCALAPPDATA%\hermes\audit\henri_audit_chain.jsonl"))
    print(f"    LIVE  {live}  exists={live.exists()}  "
          f"bytes={live.stat().st_size if live.exists() else 0}")
    for rel in ("henri_audit_chain.json", "HENRI V2/henri_audit_chain.json"):
        p = REPO / rel
        if p.exists():
            try:
                n = len(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                n = "?"
            print(f"    STALE {rel}  records={n}  tracked={bool(git_tracked(rel))}")
            warns.append(f"stale genesis ledger present: {rel}")

    # 5. worktree duplication
    print("\n[5] worktrees (duplicate trees cost disk, not correctness)")
    wt = sh("git", "worktree", "list")
    for line in wt.splitlines():
        print(f"    {line}")
    wt_dir = REPO / ".worktrees"
    if wt_dir.exists():
        tracked_ignored = sh("git", "check-ignore", "-v", ".worktrees")
        print(f"    .worktrees git-ignored: {'YES' if tracked_ignored else 'NO'}")
        if not tracked_ignored:
            warns.append(".worktrees/ is untracked but NOT ignored -> shows in git status")

    # 6. Drive junctions
    print("\n[6] Drive junction targets")
    for name in ("Drive_Inbox", "Drive_Research_Vault", "Drive_Telemetry"):
        p = REPO / "HENRI V2" / name
        exists = p.exists()
        ignored = bool(sh("git", "check-ignore", "-v", f"HENRI V2/{name}"))
        print(f"    {name:22s} resolves={exists}  gitignored={ignored}")
        if exists and not ignored:
            fails.append(f"{name} resolves and is NOT gitignored")

    print("\n" + "=" * 78)
    print(f"VERDICT: {'FAIL' if fails else 'PASS'}   failures={len(fails)}  warnings={len(warns)}")
    for f in fails:
        print(f"  FAIL  {f}")
    for w in warns:
        print(f"  WARN  {w}")
    if not fails and not warns:
        print("  topology matches the manifest")
    print("=" * 78)
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("verb", choices=["show", "verify"], default="verify", nargs="?")
    ap.add_argument("--json", dest="json_out")
    a = ap.parse_args()
    if a.verb == "show":
        print(json.dumps(MANIFEST, indent=2))
        return 0
    rc = verify()
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(MANIFEST, indent=2), encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main())
