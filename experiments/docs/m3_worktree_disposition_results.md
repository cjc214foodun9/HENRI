# M-3 — Worktree Disposition: RESULTS

**Date:** 2026-09-11 · **Prereg:** `m3_worktree_disposition_preregistration.md`
**Verdict:** `ACCEPT` — every worktree classified, unique content salvaged and
committed, ignored data archived, then removal. Zero data loss.

## Outcome

| metric | before | after |
|---|---:|---:|
| worktrees registered | 45 | **1** (primary only) |
| local branches | 130 | **130** (none deleted) |
| worktrees holding unsalvaged unique content | 32 | **0** |

## Salvage (irreplaceable content)

Classifier: `git hash-object` on the working-tree content, then
`git cat-file --batch-check`; a `missing` answer means the bytes exist only in
that worktree.

| metric | pass 1 | pass 2 (corrected) |
|---|---:|---:|
| worktrees triaged | 45 | 44 |
| with unique content | 32 | 32 |
| unique files | 50 | **59** |
| unique bytes | 79,004 | **194,118** |
| classifier errors | 0 | 0 |

**Defect found and fixed (`OBSERVED`).** Pass 1 used `git status --porcelain`
without `-uall`. Git collapses an untracked *directory* into one `?? dir/`
entry, and the pass-1 code skipped it because `Path.is_file()` is `False` for a
directory. Nine files were therefore invisible — including 25 ARC
`environment_files/*/metadata.json`. Pass 2 expanded them via `-uall`.

Had the disposition run on pass-1 data, those bytes would have been destroyed
with their worktrees. This is the exact failure the preregistration was written
to prevent.

Salvage archive: `_archive/worktree_salvage_20260911/`, 59 files, 194,118 B,
per-file sha256 in `MANIFEST.json`, 0 integrity failures. 18 substantive tools
promoted to `HENRI V2/tools/` with `PROVENANCE.md` (commit `5ef91f1`).

## Ignored data (the silent-loss vector)

`git status` does NOT list gitignored files, so `git worktree remove` destroys
them without any warning. Measured across the worktrees:

| class | files | bytes |
|---|---:|---:|
| substantive, archived | 30 tars | 25,243,510 uncompressed → 9,576,891 packed |
| regenerable (`__pycache__`, `.pyc`) | counted, not archived | rebuildable |

Archived to `%LOCALAPPDATA%\hermes\archive\worktree_ignored_20260911\` —
deliberately **outside** the repository, because the payload includes binary
`*.pt` checkpoints and the standing rule is that `*.pt` are external overlays
and the repository stays free of generated binaries.

Specifically preserved:

- `path_b2_codec_gate_a_ckpt.pt` (4,826 KB) and
  `path_b_codec_gate_a_ckpt.pt` (4,812 KB) — codec gate checkpoints
- `class48/class49` A/B arm telemetry (`arm_a_t*`, `arm_b_t*`, 1.6–1.7 MB each)
- `mbpp.jsonl` canonical + `HENRI V2/data/mbpp.jsonl`
- ARC `environment_files/*/*.py` (ka59 0.87 MB, lp85 0.48 MB, sc25, g50t, bp35 …)

## Fail-closed behaviour observed

Two defects in the disposition tooling **stopped** removal rather than
proceeding:

1. A `Path.relative_to(REPO)` call raised once the archive directory moved
   outside the repo. The tar had already been written and verified, but the
   run aborted with `SKIP_TAR_FAILED` for 30 worktrees — 14 were removed, 30
   were held back. Fixed by recording an absolute path when the relative form
   is impossible.
2. The second run archived them and removed the remaining 30.

No worktree was removed while its unique blobs or substantive ignored data were
unaccounted for. That is the prereg acceptance criterion, met.

## Verification

`%LOCALAPPDATA%\hermes\scripts\henri_post_disposition_verify.py`:
archive member counts reconciled against the run record, salvage sha256
re-checked, removed-worktree set cross-checked against the triage, and both
codec `.pt` checkpoints confirmed present.

## Non-goals honoured

No `git branch -D`. No force push. No removal of a worktree whose HEAD carried
unmerged commits (all 44 reported 0 unmerged commits). 130 branches preserved.

## Residual risk

The archive is a local, single-copy store on one disk. It is not replicated.
If that path is lost, the codec checkpoints and A/B telemetry are gone. A
follow-up may copy it to Drive or object storage; that is not claimed here.

## Incident: evidence artifact clobbered, then recovered

**Found by** post-hoc probing after a stale background-completion notification.

**What happened.** `henri_worktree_triage.py` wrote a FIXED path with no
overwrite guard. A verification loop re-ran it **after** the 44 worktrees had
been removed, so it computed an empty set and replaced the record of the real
set: `44 worktrees -> 0 worktrees (79 B)`.

**Why it matters.** That JSON was the machine-readable record underlying an
irreversible action. Destroying it is the same evidence-preservation failure the
M-3 preregistration exists to prevent. Nothing was lost from disk — only the
*record* was overwritten.

**What survived (`OBSERVED`).** The disposition report (30 plan entries,
25,243,510 B recorded), the run log with the full table and per-file listings,
the git-committed salvage `MANIFEST.json`, and both codec `.pt` checkpoints in
the 30 tars.

**Recovery.** `worktree_triage_20260911_recovered.json` is rebuilt by parsing the
surviving run log and is labelled `RECONSTRUCTED` with its provenance.

| check | manifest (committed) | reconstruction | agree |
|---|---:|---:|---|
| unique files | 59 | 59 | yes |
| unique bytes | 194,118 | 194,118 | yes |
| worktrees with unique content | 32 | 32 | yes |

**Hardening.** The triage tool now refuses to overwrite non-trivial evidence
(`_write_evidence`), writing a timestamped sibling instead unless `--force` is
passed. Regression guard: run triage twice; the first artifact must survive.

**Lesson.** A tool that measures a disappearing resource must never write its
record to a fixed, unguarded path. Structural fix: evidence artifacts are
write-once, and verification loops must not re-run stateful collectors.
