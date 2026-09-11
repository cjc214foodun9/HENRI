#!/usr/bin/env python3
"""Final disposition: archive ignored data, then remove, then verify.

Honours experiments/docs/m3_worktree_disposition_preregistration.md ordering:
  triage -> salvage -> COMMIT salvage -> archive the remaining data -> remove -> prune

Why this step exists: `git status` does NOT show gitignored files. The triage
found up to 21.2 MB of ignored data per worktree. Those bytes are in no git
object and in no salvage copy, so `git worktree remove` would destroy them
silently. This archives the SUBSTANTIVE ignored files first.

Regenerable paths (__pycache__, .venv, node_modules, caches) are counted and
reported but NOT archived: they are rebuildable and would bloat the archive.
Exception: a worktree holding > IGNORE_LIMIT is skipped entirely rather than
archived, because it likely holds a dataset or model overlay.

Fail-closed: if a tar cannot be written and verified, that worktree is not removed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

REPO = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM")
H = Path(r"C:\Users\chan\AppData\Local\hermes")
TRIAGE = H / "reports" / "worktree_triage_uall_20260911.json"
MAN = REPO / "_archive" / "worktree_salvage_20260911" / "MANIFEST.json"
ARCDIR = H / "archive" / "worktree_ignored_20260911"
REPORT = H / "reports" / "worktree_disposition_20260911.json"
IGNORE_LIMIT = 100 * 1024 * 1024

REGEN_PARTS = ("__pycache__", "/.venv/", "/venv/", "node_modules", ".pytest_cache",
               ".mypy_cache", ".ruff_cache", ".tox", ".ipynb_checkpoints",
               "/dist/", "/build/", ".egg-info")


def git(args, cwd=REPO):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                          text=True, encoding="utf-8", errors="replace")


def sanitize(p: str) -> str:
    s = p.replace(":", "").replace("\\", "_").replace("/", "_").replace(" ", "_")
    return s.strip("_")


def is_regen(rel: str) -> bool:
    low = "/" + rel.replace("\\", "/").lower()
    return any(part in low for part in REGEN_PARTS) or low.endswith(".pyc")


def ignored_files(wt: Path) -> list[str]:
    r = git(["ls-files", "--others", "--ignored", "--exclude-standard", "-z"], wt)
    if r.returncode != 0:
        return []
    return [f for f in r.stdout.split("\0") if f]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    # ---- archive integrity gate (must pass before anything is removed) ----
    man = json.loads(MAN.read_text(encoding="utf-8"))
    salv = {(r["worktree"], r["path"]): r
            for r in man["records"] if r.get("status") == "SALVAGED"}
    broken = [r["salvaged_to"] for r in salv.values()
              if not (REPO / r["salvaged_to"]).is_file()
              or hashlib.sha256((REPO / r["salvaged_to"]).read_bytes()).hexdigest()
              != r["sha256"]]
    print(f"salvage gate: {len(salv)} files, integrity failures = {len(broken)}")
    if broken:
        print("FAIL-CLOSED: salvage archive damaged; nothing removed.")
        return 3

    tri = json.loads(TRIAGE.read_text(encoding="utf-8"))
    primary = REPO.resolve()
    ARCDIR.mkdir(parents=True, exist_ok=True)

    plan, tar_fail = [], 0
    for w in tri["worktrees"]:
        p = Path(w["path"])
        if not p.exists():
            continue
        try:
            if p.resolve() == primary:
                continue
        except OSError:
            continue

        ign = ignored_files(p)
        subs = [f for f in ign if not is_regen(f)]
        sub_bytes = sum((p / f).stat().st_size for f in subs if (p / f).is_file())

        rec = {"path": w["path"], "unique": w["unique"],
               "ignored_total": len(ign), "ignored_substantive": len(subs),
               "ignored_substantive_bytes": sub_bytes, "tar": None}

        if sub_bytes > IGNORE_LIMIT:
            rec["action"] = "SKIP_IGNORED_DATA"
            plan.append(rec)
            continue

        if subs:
            if not a.apply:
                rec["action"] = "WOULD_ARCHIVE+REMOVE"
                plan.append(rec)
                continue
            name = sanitize(w["path"]) + ".tar.gz"
            tp = ARCDIR / name
            try:
                with tarfile.open(tp, "w:gz") as tf:
                    for f in subs:
                        fp = p / f
                        if fp.is_file():
                            tf.add(fp, arcname=f)
                with tarfile.open(tp, "r:gz") as tf:      # verify readability
                    n = len(tf.getmembers())
                if n != len(subs):
                    raise RuntimeError(f"member count {n} != {len(subs)}")
                # ARCDIR lives outside the repo, so relative_to(REPO) raises.
                # Record a repo-relative path when possible, else the absolute one.
                try:
                    rec["tar"] = str(tp.relative_to(REPO)).replace("\\", "/")
                    rec["tar_root"] = "repo"
                except ValueError:
                    rec["tar"] = str(tp).replace("\\", "/")
                    rec["tar_root"] = "hermes"
                rec["tar_files"] = n
                rec["tar_bytes"] = tp.stat().st_size
                rec["action"] = "ARCHIVED+REMOVE"
            except Exception as e:
                rec["action"] = "SKIP_TAR_FAILED"
                rec["err"] = f"{type(e).__name__}: {e}"
                tar_fail += 1
        else:
            rec["action"] = "REMOVE"
        plan.append(rec)

    rem = [x for x in plan if x["action"].endswith("+REMOVE") or x["action"] == "REMOVE"]
    sk = [x for x in plan if x["action"].startswith("SKIP")]
    print(f"\nexamined={len(plan)}  remove={len(rem)}  skip={len(sk)}")
    print(f"\n{'action':22s} {'uniq':>5} {'ignN':>5} {'subMB':>7}  worktree")
    print("-" * 92)
    for x in sorted(plan, key=lambda y: -y["ignored_substantive_bytes"]):
        print(f"{x['action']:22s} {x['unique']:>5} {x['ignored_total']:>5} "
              f"{x['ignored_substantive_bytes']/1e6:>7.2f}  {x['path']}")
    tot_sub = sum(x["ignored_substantive_bytes"] for x in plan)
    print(f"\nsubstantive ignored bytes across all worktrees = {tot_sub:,}")
    for x in sk:
        print(f"  SKIP {x['path']}: {x.get('err', 'ignored data over limit')}")

    if not a.apply:
        print("\n(dry run)")
        REPORT.write_text(json.dumps({"plan": plan, "applied": False},
                                     indent=2), encoding="utf-8")
        return 0

    # ---- remove ----
    print("\n=== REMOVING (never `git branch -D`) ===")
    removed = failed = 0
    for x in rem:
        r = git(["worktree", "remove", "--force", x["path"]])
        if r.returncode != 0:
            r = git(["worktree", "remove", "--force", "--force", x["path"]])
        x["removed"] = r.returncode == 0
        if r.returncode == 0:
            removed += 1
        else:
            failed += 1
            x["err"] = (r.stderr or r.stdout).strip()[:140]
            print(f"  FAILED {x['path']}: {x['err']}")
    print(f"removed={removed}  failed={failed}")

    pr = git(["worktree", "prune", "--verbose"])
    print(f"prune: {(pr.stdout or pr.stderr).strip()[:200] or '(clean)'}")

    n_wt = git(["worktree", "list", "--porcelain"]).stdout.count("worktree ")
    n_br = len([l for l in git(["branch"]).stdout.splitlines() if l.strip()])
    print(f"worktrees remaining : {n_wt}")
    print(f"branches preserved  : {n_br}")

    # fresh source check: are any REMOVED paths still present?
    still = [x["path"] for x in rem if Path(x["path"]).exists()]
    print(f"paths still on disk : {len(still)}  (expected 0)")
    for s in still[:5]:
        print(f"  {s}")

    REPORT.write_text(json.dumps(
        {"plan": plan, "applied": True, "removed": removed, "failed": failed,
         "tar_failures": tar_fail, "worktrees_remaining": n_wt,
         "branches_preserved": n_br, "paths_still_present": still},
        indent=2), encoding="utf-8")
    print(f"\nreport: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
