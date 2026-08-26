#!/usr/bin/env python3
"""HENRI telemetry/data capture -> GitHub main (experiments/capture).

Deterministic, bounded, stdlib-only. Run by cron (no-agent, every 30m) via a
wrapper in the Hermes scripts dir. Sources:
  1. G:\\My Drive\\HENRI_Inbox         - newest files (<= 200 KB each, <= 10)
  2. G:\\My Drive\\HENRI_Research_Vault - newest notes (<= 300 KB each, <= 5)
  3. G:\\My Drive\\HENRI_Telemetry      - newest telemetry files (<= 500 KB each, <= 5)
  4. Vast.ai (ssh alias vast-5090)     - /root/henri-archive newest (<= 2 MB each, <= 8)
                                        + remote telemetry inventory (no bulk pull)
Bounds: per-run copied bytes <= 4 MB. Retention: newest 12 snapshots.
Push: detached worktree at origin/main; reset --hard origin/main; add bounded
paths only; commit; push HEAD:main; verify against git ls-remote. Fail-closed:
any error -> receipt records it, no push, exit 1.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(r"C:/Users/chan/Desktop/HENRI 7B SWARM")
WT = Path(r"C:/Users/chan/henri-worktrees/telemetry-capture")
DRIVE = Path(r"G:/My Drive")
SOURCES = {
    "drive_inbox": (DRIVE / "HENRI_Inbox", 200_000, 10),
    "drive_research": (DRIVE / "HENRI_Research_Vault", 300_000, 5),
    "drive_telemetry": (DRIVE / "HENRI_Telemetry", 500_000, 5),
}
VAST_ARCHIVE = "/root/henri-archive"
MAX_RUN_BYTES = 4_000_000
RETENTION = 12
LOCK = Path(os.environ.get("TEMP", "/tmp")) / "henri_capture.lock"
SSH_BASE = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "vast-5090"]
SCP_BASE = ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory(path: Path, limit: int = 300) -> list:
    out = []
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                p = Path(root) / name
                try:
                    st = p.stat()
                    out.append({"path": str(p.relative_to(path)), "bytes": st.st_size,
                                "mtime_utc": _dt.datetime.fromtimestamp(st.st_mtime, tz=_dt.timezone.utc).isoformat()})
                except OSError:
                    continue
    except OSError:
        return []
    out.sort(key=lambda r: r["mtime_utc"], reverse=True)
    return out[:limit]


def copy_newest(src_dir: Path, dst_dir: Path, max_bytes: int, count: int,
                run_budget: list, inv: list, copied: list, skipped: list) -> None:
    for rec in inv[:count]:
        p = src_dir / rec["path"]
        if rec["bytes"] > max_bytes:
            skipped.append(f"{rec['path']} (>{max_bytes}B)")
            continue
        if run_budget[0] + rec["bytes"] > MAX_RUN_BYTES:
            skipped.append(f"{rec['path']} (run budget)")
            break
        try:
            d = dst_dir / os.path.dirname(rec["path"])
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, d / os.path.basename(rec["path"]))
            run_budget[0] += rec["bytes"]
            copied.append({"path": rec["path"], "bytes": rec["bytes"],
                           "sha256": sha256(d / os.path.basename(rec["path"]))})
        except OSError as e:
            skipped.append(f"{rec['path']} ({e})")


def vast_capture(dst_dir: Path, run_budget: list, copied: list, skipped: list) -> dict:
    try:
        r = subprocess.run(SSH_BASE + ["ls", "-t", VAST_ARCHIVE, "|", "head", "-8"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return {"status": "BLOCKED", "detail": r.stderr.strip()[:300]}
        names = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()][:8]
        arch = dst_dir / "henri-archive"
        arch.mkdir(parents=True, exist_ok=True)
        arch_copied = 0
        for n in names:
            if run_budget[0] >= MAX_RUN_BYTES:
                skipped.append(f"vast:{n} (run budget)")
                break
            tmp = arch / n
            rr = subprocess.run(SCP_BASE + [f"vast-5090:{VAST_ARCHIVE}/{n}", str(tmp)],
                                capture_output=True, text=True, timeout=120)
            if rr.returncode == 0 and tmp.exists() and tmp.stat().st_size <= 2_000_000:
                run_budget[0] += tmp.stat().st_size
                copied.append({"path": f"henri-archive/{n}", "bytes": tmp.stat().st_size,
                               "sha256": sha256(tmp)})
                arch_copied += 1
            else:
                skipped.append(f"vast:{n} (scp rc={rr.returncode})")
                tmp.unlink(missing_ok=True)
        inv = subprocess.run(SSH_BASE + ["find", "/workspace", "-maxdepth", "3",
                                         "-name", "*.jsonl", "-mtime", "-3", "-size", "-5M"],
                             capture_output=True, text=True, timeout=60)
        inv_lines = [ln.strip() for ln in inv.stdout.splitlines() if ln.strip()][:30] if inv.returncode == 0 else []
        return {"status": "OK", "copied": arch_copied, "archive_files": arch_copied,
                "remote_jsonl_index": inv_lines}
    except (subprocess.TimeoutExpired, OSError) as e:
        return {"status": "BLOCKED", "detail": str(e)[:300]}


def main() -> int:
    # lock guard (stale after 25 min)
    try:
        if LOCK.exists() and (_dt.datetime.now().timestamp() - LOCK.stat().st_mtime) < 1500:
            return 0
        LOCK.touch()
    except OSError:
        pass

    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    receipt = {"schema": "henri.capture-receipt.v1", "run_utc": ts, "sources": {},
               "copied": [], "skipped": [], "bytes_total": 0}
    run_budget = [0]
    copied, skipped = [], []

    try:
        # ---- drive sources ----
        for key, (src, maxb, cnt) in SOURCES.items():
            if not src.exists():
                receipt["sources"][key] = {"status": "BLOCKED", "detail": "source missing"}
                continue
            inv = inventory(src)
            tmp = Path(tempfile.mkdtemp()) / key
            before = len(copied)
            copy_newest(src, tmp, maxb, cnt, run_budget, inv, copied, skipped)
            if copied and len(copied) > before:
                snap = WT / "experiments" / "capture" / ts / key
                snap.parent.mkdir(parents=True, exist_ok=True)
                if snap.exists():
                    shutil.rmtree(snap)
                shutil.move(tmp, snap)
            else:
                shutil.rmtree(tmp.parent, ignore_errors=True)
            receipt["sources"][key] = {"status": "OK", "inventory_count": len(inv),
                                       "copied": len(copied) - before}
        # ---- vast ----
        vdst = Path(tempfile.mkdtemp())
        vres = vast_capture(vdst / "vast", run_budget, copied, skipped)
        if vres["status"] == "OK" and any(c["path"].startswith("henri-archive/") for c in copied):
            snap = WT / "experiments" / "capture" / ts / "vast"
            snap.parent.mkdir(parents=True, exist_ok=True)
            if snap.exists():
                shutil.rmtree(snap)
            shutil.move(vdst / "vast", snap)
        shutil.rmtree(vdst, ignore_errors=True)
        vres["inventory_count"] = len(vres.get("remote_jsonl_index", []))
        receipt["sources"]["vast"] = vres
    except OSError as e:
        receipt["sources"]["local"] = {"status": "ERROR", "detail": str(e)[:300]}

    receipt["copied"] = copied
    receipt["skipped"] = skipped[:20]
    receipt["bytes_total"] = run_budget[0]

    # ---- git: write snapshot + manifest, commit, push ----
    if not (WT / ".git").exists():
        print("ERROR: capture worktree missing:", WT)
        return 1
    try:
        subprocess.run(["git", "-C", str(WT), "fetch", "origin", "main"],
                       capture_output=True, text=True, timeout=120, check=True)
        subprocess.run(["git", "-C", str(WT), "reset", "--hard", "origin/main"],
                       capture_output=True, text=True, timeout=120, check=True)
    except subprocess.CalledProcessError as e:
        print("ERROR: git sync:", e.stderr.strip()[:300])
        return 1

    snap_dir = WT / "experiments" / "capture" / ts
    if not snap_dir.exists():
        snap_dir.mkdir(parents=True)  # empty snapshot dir still records statuses
    (snap_dir / "receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")

    # retention: drop oldest snapshots beyond RETENTION
    snaps = sorted([p for p in (WT / "experiments" / "capture").glob("*") if p.is_dir()])
    for old in snaps[:-RETENTION]:
        shutil.rmtree(old, ignore_errors=True)

    # manifest (stable schema for Gemini)
    man_path = WT / "experiments" / "manifest.json"
    manifest = {"schema": "henri.capture-manifest.v1", "generated_utc": ts,
                "latest_snapshot": f"capture/{ts}",
                "sources": {k: v.get("status") for k, v in receipt["sources"].items()},
                "counts": {k: v.get("copied", 0) for k, v in receipt["sources"].items()},
                "bytes_total": receipt["bytes_total"],
                "retention_snapshots": RETENTION}
    man_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    try:
        subprocess.run(["git", "-C", str(WT), "add", "-A", "--",
                        "experiments/capture", "experiments/manifest.json",
                        "experiments/README.md", "experiments/docs"],
                       capture_output=True, text=True, timeout=120, check=True, env=env)
        if subprocess.run(["git", "-C", str(WT), "diff", "--cached", "--quiet"],
                          capture_output=True, text=True, timeout=60, env=env).returncode == 0:
            return 0  # nothing changed; silent
        subprocess.run(["git", "-C", str(WT), "commit", "-m",
                        f"capture: HENRI telemetry/data snapshot {ts}",
                        "--", "experiments/capture", "experiments/manifest.json",
                        "experiments/README.md", "experiments/docs"],
                       capture_output=True, text=True, timeout=120, check=True, env=env)
        push = subprocess.run(["git", "-C", str(WT), "push", "origin", "HEAD:main"],
                              capture_output=True, text=True, timeout=180, env=env)
        if push.returncode != 0:
            print("ERROR: push failed:", push.stderr.strip()[:300])
            return 1
        local = subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"],
                               capture_output=True, text=True, timeout=30, env=env).stdout.strip()
        remote = subprocess.run(["git", "ls-remote", "origin", "refs/heads/main"],
                                capture_output=True, text=True, timeout=60, env=env).stdout.split()[0]
        if local != remote:
            print(f"ERROR: push verify mismatch local={local} remote={remote}")
            return 1
        print(f"PUSHED {ts} sources={','.join(receipt['sources'])} bytes={receipt['bytes_total']} commit={local[:10]}")
        return 0
    except subprocess.CalledProcessError as e:
        print("ERROR:", e.stderr.strip()[:300])
        return 1


if __name__ == "__main__":
    sys.exit(main())
