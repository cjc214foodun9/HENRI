#!/usr/bin/env python3
"""CORRECTED worktree triage: expand untracked DIRECTORIES (-uall).

Defect in pass 1 (OBSERVED): `git status --porcelain` collapses an untracked
directory into ONE entry `?? dir/`. The pass-1 salvager then skipped it because
`Path.is_file()` is False for a directory. So any worktree holding a large
untracked directory was misclassified as "no unique content".

This pass lists every untracked file individually via `-uall`, so the unique
set is complete before anything is removed.

Reports per worktree:
  changed_all   files differing from HEAD or untracked (individually listed)
  unique        content hashes ABSENT from the git object database
  head          worktree HEAD
  branch        checked-out branch (if any)
  unmerged      commits reachable from HEAD but from no remote ref

READ-ONLY. Writes JSON + a text summary outside the repo.
"""
from __future__ import annotations

import json
import datetime
import subprocess
from pathlib import Path

def _write_evidence(path: Path, payload: str, force: bool = False) -> Path:
    """Write an evidence artifact WITHOUT destroying a prior one.

    A triage reflects a specific set of worktrees at a specific time. Re-running
    it after those worktrees are gone must not erase the record of the earlier
    set (observed incident 2026-09-11: an unguarded re-run reduced a 44-worktree
    record to 0 rows). Non-trivial existing content is never silently replaced.
    """
    if path.exists() and not force and path.stat().st_size > 512:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
        alt.write_text(payload, encoding="utf-8")
        print(f"REFUSED to overwrite evidence ({path.stat().st_size:,} B): "
              f"{path.name}")
        print(f"  wrote {alt.name} instead (pass --force to replace)")
        return alt
    path.write_text(payload, encoding="utf-8")
    return path


REPO = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM")
OUT = Path(r"C:\Users\chan\AppData\Local\hermes\reports")


def git(args, cwd, stdin_text=None):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          input=stdin_text)


def worktrees():
    r = git(["worktree", "list", "--porcelain"], REPO)
    out, cur = [], {}
    for line in r.stdout.splitlines():
        if line.startswith("worktree "):
            if cur:
                out.append(cur)
            cur = {"path": line[9:].strip()}
        elif line.startswith("HEAD "):
            cur["head"] = line[5:].strip()
        elif line.startswith("branch "):
            cur["branch"] = line[7:].strip()
        elif line.strip() == "detached":
            cur["branch"] = "(detached)"
    if cur:
        out.append(cur)
    return out


def changed_all(wt: Path) -> list[str]:
    """Every changed/untracked file, with untracked dirs EXPANDED (-uall)."""
    r = git(["status", "--porcelain", "-z", "-uall"], wt)
    if r.returncode != 0:
        return []
    toks = [t for t in r.stdout.split("\0") if t]
    paths, i = [], 0
    while i < len(toks):
        e = toks[i]
        i += 1
        if len(e) < 3:
            continue
        xy, path = e[:2], e[3:]
        if "R" in xy or "C" in xy:
            i += 1                      # next token is the rename source
        if "D" in xy:
            continue                    # deleted in worktree: nothing to hash
        if path:
            paths.append(path)
    return paths


def main() -> int:
    primary = REPO.resolve()
    rows = []
    for w in worktrees():
        p = Path(w["path"])
        try:
            if p.resolve() == primary:
                continue
        except OSError:
            continue

        paths = changed_all(p)
        existing = [x for x in paths if (p / x).is_file()]
        uniq, nbytes = [], 0
        if existing:
            h = git(["hash-object", "--stdin-paths"], p,
                    stdin_text="\n".join(existing))
            hs = h.stdout.split()
            if len(hs) != len(existing):
                uniq = ["<HASH_ALIGNMENT_FAILED>"]
            else:
                c = git(["cat-file", "--batch-check"], p,
                        stdin_text="\n".join(hs))
                for f, line in zip(existing, c.stdout.splitlines()):
                    if line.endswith("missing"):
                        uniq.append(f)
                        try:
                            nbytes += (p / f).stat().st_size
                        except OSError:
                            pass

        unmerged = 0
        if w.get("head"):
            rc = git(["rev-list", "--count", w["head"], "--not", "--remotes"], p)
            try:
                unmerged = int(rc.stdout.strip() or 0)
            except ValueError:
                unmerged = -1

        rows.append({"path": w["path"], "head": (w.get("head") or "")[:12],
                     "branch": w.get("branch", "?"),
                     "changed_all": len(paths), "unique": len(uniq),
                     "unique_bytes": nbytes,
                     "unique_files": uniq,
                     "unmerged_commits": unmerged})

    rows.sort(key=lambda r: (-r["unique"], -r["changed_all"]))
    tot_u = sum(r["unique"] for r in rows if r["unique"] > 0)
    tot_b = sum(r["unique_bytes"] for r in rows)
    tarball = [r for r in rows if r["changed_all"] > 0]

    print(f"worktrees examined          = {len(rows)}")
    print(f"with any change             = {len(tarball)}")
    print(f"WITH UNIQUE CONTENT         = {sum(1 for r in rows if r['unique'] > 0)}")
    print(f"total unique files          = {tot_u}")
    print(f"total unique bytes          = {tot_b:,}")
    print(f"max changed_all in one tree = {max((r['changed_all'] for r in rows), default=0)}")
    print()
    print(f"{'unique':>7} {'changed':>8} {'unmerged':>9}  worktree")
    print("-" * 78)
    for r in rows:
        if r["unique"] or r["changed_all"]:
            print(f"{r['unique']:>7} {r['changed_all']:>8} "
                  f"{r['unmerged_commits']:>9}  {r['path']}")
    print()
    for r in rows:
        if r["unique"]:
            print(f"--- {r['path']}  ({r['unique']} unique, {r['unique_bytes']:,} B) ---")
            for f in r["unique_files"][:40]:
                print(f"      {f}")

    _write_evidence(OUT / "worktree_triage_uall_20260911.json",
        json.dumps({"worktrees": rows, "total_unique_files": tot_u,
                    "total_unique_bytes": tot_b}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'worktree_triage_uall_20260911.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
