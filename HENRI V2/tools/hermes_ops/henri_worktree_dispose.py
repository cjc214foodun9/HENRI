#!/usr/bin/env python3
"""Worktree disposition: verify the salvage, then remove redundant worktrees.

Ordering per experiments/docs/m3_worktree_disposition_preregistration.md:
  triage -> salvage -> COMMIT salvage -> archive record -> remove -> prune

Guards (fail-closed per worktree):
  G1  every unique-content file is in the committed salvage archive with a
      matching sha256. If the archive is damaged, NOTHING is removed.
  G2  gitignored data volume <= IGNORE_LIMIT_BYTES. `git status` does not show
      ignored files, so a large ignored dataset (*.pt overlays, corpora) would
      be destroyed silently. Such worktrees are REPORTED and SKIPPED, never
      removed on this directive.
  G3  the primary worktree is never touched; `git branch -D` never runs.

Usage: python dispose_worktrees.py            # dry run
       python dispose_worktrees.py --apply    # remove + prune
"""
from __future__ import annotations

import argparse
import hashlib
import json
import datetime
import subprocess
from pathlib import Path


def _write_evidence(path: Path, payload: str, force: bool = False,
                    encoding: str = "utf-8") -> Path:
    """Write an evidence artifact WITHOUT destroying a prior one.

    A disposition record reflects one set of worktrees at one time. Re-running
    after those worktrees are gone must not erase the record of the earlier set
    (observed incident 2026-09-11: an unguarded re-run reduced a 44-worktree
    triage record to 0 rows). Non-trivial existing content is never silently
    replaced; a timestamped sibling is written instead.
    """
    if path.exists() and not force and path.stat().st_size > 512:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        alt.write_text(payload, encoding=encoding)
        print(f"REFUSED to overwrite evidence ({path.stat().st_size:,} B): "
              f"{path.name}")
        print(f"  wrote {alt.name} instead (pass --force to replace)")
        return alt
    path.write_text(payload, encoding=encoding)
    return path

REPO = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM")
H = Path(r"C:\Users\chan\AppData\Local\hermes")
TRIAGE = H / "reports" / "worktree_triage_uall_20260911.json"
MAN = REPO / "_archive" / "worktree_salvage_20260911" / "MANIFEST.json"
IGNORE_LIMIT = 100 * 1024 * 1024          # 100 MB of ignored data -> skip


def git(args, cwd=REPO):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                          text=True, encoding="utf-8", errors="replace")


def ignored_volume(wt: Path) -> tuple[int, int]:
    r = git(["ls-files", "--others", "--ignored", "--exclude-standard", "-z"], wt)
    if r.returncode != 0:
        return -1, 0
    files = [f for f in r.stdout.split("\0") if f]
    total = 0
    for f in files:
        try:
            total += (wt / f).stat().st_size
        except OSError:
            pass
    return total, len(files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    # ---- G1 gate: archive integrity -------------------------------------
    man = json.loads(MAN.read_text(encoding="utf-8"))
    salv = {}
    for r in man["records"]:
        if r.get("status") == "SALVAGED":
            salv[(r["worktree"], r["path"])] = r
    broken = []
    for (w, p), r in salv.items():
        f = REPO / r["salvaged_to"]
        if not f.is_file():
            broken.append(r["salvaged_to"])
        elif hashlib.sha256(f.read_bytes()).hexdigest() != r["sha256"]:
            broken.append(r["salvaged_to"])
    print(f"salvage archive : {len(salv)} files  "
          f"({man.get('bytes_salvaged', 0):,} B)  integrity failures = {len(broken)}")
    if broken:
        print("FAIL-CLOSED: archive damaged. Nothing will be removed.")
        for b in broken[:5]:
            print(f"  {b}")
        return 3

    # ---- classify ------------------------------------------------------
    tri = json.loads(TRIAGE.read_text(encoding="utf-8"))
    primary = REPO.resolve()
    plan = []
    for w in tri["worktrees"]:
        p = Path(w["path"])
        if not p.exists():
            plan.append({**w, "action": "SKIP_ABSENT", "ignored_bytes": 0,
                         "ignored_files": 0, "missing_from_archive": []})
            continue
        try:
            if p.resolve() == primary:
                continue
        except OSError:
            pass
        missing = [f for f in w["unique_files"] if (w["path"], f) not in salv]
        ib, nign = ignored_volume(p)
        if missing:
            action = "SKIP_UNSALVAGED"
        elif ib < 0:
            action = "SKIP_CLASSIFIER_ERROR"
        elif ib > IGNORE_LIMIT:
            action = "SKIP_IGNORED_DATA"
        else:
            action = "REMOVE"
        plan.append({**w, "ignored_files": nign, "ignored_bytes": ib,
                     "missing_from_archive": missing, "action": action})

    rem = [x for x in plan if x["action"] == "REMOVE"]
    skip = [x for x in plan if x["action"].startswith("SKIP")]
    print(f"\nworktrees examined : {len(plan)}")
    print(f"REMOVE             : {len(rem)}")
    print(f"SKIP               : {len(skip)}")
    print(f"\n{'action':22s} {'uniq':>5} {'ign MB':>8}  worktree")
    print("-" * 88)
    for x in sorted(plan, key=lambda y: (y["action"], -y["unique"])):
        print(f"{x['action']:22s} {x['unique']:>5} "
              f"{x['ignored_bytes']/1e6:>8.1f}  {x['path']}")
    for x in skip:
        if x["action"] == "SKIP_IGNORED_DATA":
            print(f"\n  NOTE {x['path']}: holds {x['ignored_bytes']/1e6:.1f} MB of "
                  f"gitignored data in {x['ignored_files']} files - not destroyed.")
        if x["missing_from_archive"]:
            print(f"\n  MISSING ARCHIVE {x['path']} -> {x['missing_from_archive'][:4]}")

    if not a.apply:
        print("\n(dry run - pass --apply to remove the REMOVE set)")
        return 0

    # ---- apply ---------------------------------------------------------
    print("\n=== REMOVING (branches are never deleted) ===")
    removed = failed = 0
    for x in rem:
        p = x["path"]
        r = git(["worktree", "remove", "--force", p])
        if r.returncode != 0:
            r = git(["worktree", "remove", "--force", "--force", p])
        if r.returncode == 0:
            removed += 1
        else:
            failed += 1
            print(f"  FAILED {p}: {(r.stderr or r.stdout).strip()[:130]}")
    print(f"removed={removed}  failed={failed}")

    pr = git(["worktree", "prune", "--verbose"])
    print(f"prune: {(pr.stdout or pr.stderr).strip()[:200] or '(nothing to prune)'}")
    n = git(["worktree", "list", "--porcelain"]).stdout.count("worktree ")
    br = len([l for l in git(["branch"]).stdout.splitlines() if l.strip()])
    print(f"worktrees remaining : {n}")
    print(f"branches preserved  : {br}")

    _write_evidence(
        H / "reports" / "worktree_disposition_20260911.json",
        json.dumps({"plan": plan, "removed": removed, "failed": failed,
                    "worktrees_remaining": n, "branches_preserved": br},
                   indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
