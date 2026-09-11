"""E5 ground-state + legacy-bounds scan (deterministic, file-backed).

Real evidence only: every value is read from files on disk and hashed.
Writes ONE json receipt; the receipt is the evidence, not stdout.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

E4WT = Path(r"C:\Users\chan\henri-worktrees\e4-wt")
HV2 = E4WT / "HENRI V2"
LEDGER = Path(r"C:\Users\chan\AppData\Local\hermes\audit\henri_audit_chain.jsonl")
OUT = Path(r"C:\Users\chan\henri-telemetry\e3\e5_ground.json")


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=str(E4WT), capture_output=True,
                          text=True, timeout=90).stdout.strip()


def main() -> None:
    rec: dict = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    rec["git"] = {
        "head": git("rev-parse", "--short", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "main": git("rev-parse", "--short", "origin/main"),
        "remote_e4": git("ls-remote", "origin", "refs/heads/carrier/e4-construct"),
        "dirty_all": git("status", "--porcelain=v1", "-uall").splitlines(),
    }

    rows = [json.loads(l) for l in LEDGER.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    prev, ok = "0" * 64, True
    for i, r in enumerate(rows):
        body = (f"{r['idx']}|{r['ts']}|{r['actor']}|{r['action']}|"
                f"{json.dumps(r['payload'], sort_keys=True)}|{r['prev_hash']}")
        if (r["idx"] != i or r["prev_hash"] != prev
                or hashlib.sha256(body.encode("utf-8")).hexdigest() != r["hash"]):
            ok = False
            break
        prev = r["hash"]
    rec["chain"] = {"intact": ok, "records": len(rows),
                    "head": rows[-1]["hash"][:16] if rows else None,
                    "last6": [[r["idx"], r["action"]] for r in rows[-6:]]}

    # ---- every file under the repo carrying the legacy bounds ------------
    hits = []
    for p in sorted(HV2.rglob("*")):
        if not p.is_file() or p.suffix not in (".py", ".md", ".yml", ".yaml", ".ini", ".cfg"):
            continue
        try:
            t = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if "0.285" in t or "0.640" in t:
            ln = [(i + 1, l.strip()) for i, l in enumerate(t.splitlines())
                  if "0.285" in l or "0.640" in l]
            rel = str(p.relative_to(E4WT))
            kind = ("TEST" if p.name.startswith("test_") else
                    "RECEIPT" if ("audit" in p.name or "verdict" in p.name) else
                    "PREREG" if "prereg" in p.name else "CODE")
            hits.append({"file": rel, "kind": kind, "sha256": sha(p)[:16], "hits": ln})
    rec["bounds_files"] = hits

    # ---- CI ---------------------------------------------------------------
    wf = E4WT / ".github" / "workflows"
    ci = {"dir_exists": wf.exists(), "files": [], "files_with_bounds": []}
    if wf.exists():
        for f in sorted(wf.iterdir()):
            if f.is_file():
                ci["files"].append(f.name)
                t = f.read_text(encoding="utf-8", errors="replace")
                if "0.285" in t or "0.640" in t:
                    ci["files_with_bounds"].append(f.name)
    rec["ci"] = ci

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    print(f"\nWROTE {OUT} sha256={sha(OUT)[:16]}")


if __name__ == "__main__":
    main()
