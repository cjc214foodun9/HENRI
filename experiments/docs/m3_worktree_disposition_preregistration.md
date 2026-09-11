# M-3 — Worktree Disposition Preregistration

**Date:** 2026-09-11 · **Status:** `REGISTERED` (triage run, no removal yet)
**Risk class:** irrecoverable data loss. This is the only irreversible step in
the consolidation program.

## 1. Why pass 1 was insufficient (OBSERVED defect)

Pass 1 used `git status --porcelain` **without `-uall`**. Git collapses an
untracked *directory* into one entry: `?? somedir/`. The pass-1 code then tested
`Path.is_file()`, which is `False` for a directory, and skipped it.

Consequence: any worktree holding a large untracked directory was classified
"no unique content" and would have been removable — destroying files that exist
nowhere else. That is precisely the risk the user named: *"some of these files
or branches may contain the necessary code we need to progress."*

Pass 2 (`scripts/henri_worktree_triage.py`) lists every untracked file
individually via `-uall`.

## 2. Pass 1 result, for comparison (OBSERVED)

| metric | pass 1 |
|---|---:|
| worktrees registered | 45 |
| dirty | 36 |
| with unique content | 32 |
| unique files | 50 |
| unique bytes | 79,004 |
| classifier errors | 0 |

Committed salvage: `_archive/worktree_salvage_20260911/`, manifest with per-file
sha256, 18 substantive tools promoted to `HENRI V2/tools/` (commit `5ef91f1`).

## 3. Classifier definition (authoritative)

A file is **irreplaceable** when its working-tree content hash, computed with
`git hash-object`, returns `missing` from `git cat-file --batch-check`. Then the
bytes exist only in that worktree.

| class | test | action |
|---|---|---|
| `clean-merged` | no diff, no unmerged commits | removable after archive |
| `dirty-all-recoverable` | all changed blobs present in object DB | removable after archive |
| `dirty-unique` | ≥1 blob absent from the object DB | **preserve** until salvaged |
| `unmerged-branch` | commits unreachable from any remote ref | **preserve** the branch |

`git worktree remove` never deletes branches. `git branch -D` is out of scope
and must not appear in any disposition script.

## 4. Pre-registered acceptance

| outcome | criterion |
|---|---|
| **ACCEPT** | every worktree classified; unique set fully salvaged and committed; archive verified; then removal |
| **REJECT** | any worktree removed while holding a blob absent from the object DB |
| **BLOCKED** | classifier error on any worktree — that worktree is neither archived nor removed |

## 5. Mandatory ordering

```text
triage (-uall, read-only)
  → salvage unique blobs (copy only)
  → commit salvage + verify manifest against source hashes
  → archive remaining diff as a bundle
  → ONLY THEN git worktree remove <path>
  → git worktree prune
  → verify: registered count drops, no unique blob lost
```

Any step that removes a path before its unique blobs are committed violates this
preregistration. A dirty worktree is **never** removed merely to reclaim disk.

## 6. Explicit non-goals

- No `git branch -D`.
- No force push.
- No removal of a worktree whose HEAD has unmerged commits.
- No assumption that a large `changed` count implies a large unique set. Pass 1
  measured 36 dirty worktrees with only 50 unique files total: most "changed"
  content already exists in the object database.
