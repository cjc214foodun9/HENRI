#!/usr/bin/env python3
"""Read-only worktree inventory: find content that exists NOWHERE ELSE.

For every dirty linked worktree, hash each changed/untracked file's WORKING-TREE
content (git hash-object, no -w: non-destructive) and test whether that blob
exists in the object database. A file whose blob is absent exists only in that
worktree — it is unrecoverable if the worktree is removed. Those files are the
salvage set.

This is the only safe classifier before any worktree removal.

Writes JSON + a readable list to the reports dir. Never writes to the repo.
"""
from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path


def _write_evidence(path: Path, payload: str, force: bool = False,
                    encoding: str = "utf-8") -> Path:
    """Write an evidence artifact WITHOUT destroying a prior one.

    The salvage inventory is the record of which bytes were unique BEFORE the
    redundant worktrees were removed. Re-running this tool afterwards computes
    against the surviving state, so an unguarded write would replace that proof
    with a near-empty file. Non-trivial existing content is never silently
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
OUTDIR = Path(r"C:\Users\chan\AppData\Local\hermes\reports")


def git(args, cwd, text_input=None):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          input=text_input)


def wt_list():
    r = git(["worktree", "list", "--porcelain"], REPO)
    out = []
    for line in r.stdout.splitlines():
        if line.startswith("worktree "):
            out.append(Path(line[len("worktree "):].strip()))
    return out


def changed_paths(wt: Path):
    """Return existing files that differ from HEAD or are untracked."""
    r = git(["status", "--porcelain", "-z"], wt)
    if r.returncode != 0:
        return []
    toks = [t for t in r.stdout.split("\0") if t]
    paths, i = [], 0
    while i < len(toks):
        entry = toks[i]
        i += 1
        if len(entry) < 3:
            continue
        xy, path = entry[:2], entry[3:]
        if "R" in xy or "C" in xy:      # rename/copy: next token is the source
            i += 1
        if "D" in xy:                   # deleted in worktree -> nothing to hash
            continue
        if path:
            paths.append(path)
    return paths


def classify(wt: Path):
    primary = REPO.resolve()
    try:
        if wt.resolve() == primary:
            return None
    except OSError:
        return None

    paths = changed_paths(wt)
    if not paths:
        return None

    existing = [p for p in paths if (wt / p).is_file()]
    if not existing:
        return {"worktree": str(wt), "changed": len(paths), "unique": 0,
                "unique_files": [], "unique_bytes": 0}

    h = git(["hash-object", "--stdin-paths"], wt, text_input="\n".join(existing))
    hashes = h.stdout.split()
    if len(hashes) != len(existing):
        # alignment broken (path with newline, or unreadable file) -> fail closed
        return {"worktree": str(wt), "changed": len(paths), "unique": -1,
                "unique_files": ["<HASH_ALIGNMENT_FAILED>"], "unique_bytes": 0}

    c = git(["cat-file", "--batch-check"], wt, text_input="\n".join(hashes))
    checks = c.stdout.splitlines()
    if len(checks) != len(hashes):
        return {"worktree": str(wt), "changed": len(paths), "unique": -1,
                "unique_files": ["<CATFILE_ALIGNMENT_FAILED>"], "unique_bytes": 0}

    uniq = [f for f, line in zip(existing, checks) if line.endswith("missing")]
    nbytes = 0
    for f in uniq:
        try:
            nbytes += (wt / f).stat().st_size
        except OSError:
            pass
    return {"worktree": str(wt), "changed": len(paths), "unique": len(uniq),
            "unique_files": uniq[:60], "unique_bytes": nbytes}


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    trees = wt_list()
    results, skipped = [], 0
    for wt in trees:
        try:
            rec = classify(wt)
        except Exception as exc:                      # fail closed, never crash
            rec = {"worktree": str(wt), "changed": -1, "unique": -1,
                   "unique_files": [f"<ERROR {type(exc).__name__}>"],
                   "unique_bytes": 0}
        if rec is None:
            skipped += 1
            continue
        results.append(rec)

    dirty = [r for r in results if r["unique"] != 0 or r["changed"] > 0]
    withuniq = [r for r in results if r["unique"] > 0]
    errs = [r for r in results if r["unique"] == -1]

    plan = {
        "generated": "2026-09-11",
        "worktrees_registered": len(trees),
        "clean_or_primary_skipped": skipped,
        "dirty_count": len(dirty),
        "with_unique_content": len(withuniq),
        "classifier_errors": len(errs),
        "total_unique_files": sum(r["unique"] for r in withuniq),
        "total_unique_bytes": sum(r["unique_bytes"] for r in withuniq),
        "worktrees": sorted(results, key=lambda r: -r["unique"]),
    }
    _write_evidence(OUTDIR / "worktree_inventory_20260911.json", json.dumps(plan, indent=2))

    lines = ["WORKTREE UNIQUE-CONTENT INVENTORY (read-only)", "=" * 62,
             f"registered={len(trees)}  clean/skipped={skipped}  dirty={len(dirty)}",
             f"WITH UNIQUE CONTENT = {len(withuniq)} worktrees, "
             f"{plan['total_unique_files']} files, "
             f"{plan['total_unique_bytes'] / 1e6:.2f} MB",
             f"classifier errors = {len(errs)}", "",
             f"{'unique':>7} {'changed':>8}  worktree"]
    for r in plan["worktrees"]:
        if r["unique"] or r["changed"]:
            lines.append(f"{r['unique']:>7} {r['changed']:>8}  {r['worktree']}")
    lines.append("")
    for r in withuniq:
        lines.append(f"--- {r['worktree']}  ({r['unique']} unique) ---")
        for f in r["unique_files"][:25]:
            lines.append(f"      {f}")
    _write_evidence(OUTDIR / "worktree_inventory_20260911.txt", "\n".join(lines))
    print("\n".join(lines[:40]))
    print(f"\nfull: {OUTDIR / 'worktree_inventory_20260911.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
