# DISCLOSURE — I destroyed 32 evidence files while trying to commit them

**Date:** 2026-09-16 | **Tree:** `carrier/aaii-v43`
**Severity:** self-inflicted, recoverable, disclosed before the commit that hides it.

---

## 1. What happened

Commit `c573175` reported success. Its `git add` had printed this warning and
continued:

```
warning: could not open directory '.../m2-agent-loop-gate-v3__<sha>__<ts>/
  m2-part-i__<sha>__<ts>/automationbench-aa-scaffold__<sha>__<ts>/tasks/':
  Filename too long
```

So 34 of the 35 files in the authoritative M2 run directory were never staged.
The committed M2 gate clause `file_output_written: True` cited **32
`export.json`/`summary.json` files that existed on disk and nowhere in the
repository**.

I then wrote `fixpath.sh` to repair it. That script was **destructive**:

```bash
for inner in */; do
  inner="${inner%/}"
  case "$inner" in tasks|*.jsonl|*.json) continue ;; esac
  if [ -d "$inner" ]; then
    if [ -d "t" ]; then rm -rf "t"; fi     # <-- unconditional delete of the target
    mv "$inner" "t"
  fi
done
```

On iteration 1 it moved `m2-part-i__…` to `t`. On iteration 2 the `*/` glob
re-expanded to the `t` it had just created, matched it, and ran `rm -rf t` —
deleting the directory holding the 32 output files — then `mv t t` no-op'd.
`git commit` found nothing staged and exited 1.

**Net effect:** the evidence for a committed claim was deleted from disk by the
script written to preserve it. The `t` directory survives, empty (0 files).

## 2. Why this is the same defect class as the rest of the session

| Instance | Tool said | Reality |
|---|---|---|
| detached HEAD | `git push` → `Everything up-to-date` | 2 commits unpushed |
| `git add` warning | exit 0, "success" | 34 files unstaged |
| my `fixpath.sh` | `flattened: … -> t` | `rm -rf` on the output directory |

**A tool's success signal is not the work.** In each case I acted on the status
line instead of reading the artifact. Here the artifact was the file count, and
step 2 of `fixpath.sh` printed `files after flatten : 1` — down from 35 — and I
let the script proceed to `git commit`. I should have aborted on that line.

## 3. Recovery — the evidence was never actually lost

| Run dir | Status | Files | Outputs |
|---|---|---|---|
| `m2-agent-loop-gate-v3__…d5875c5f` | **DAMAGED** by me | 1 | 0 |
| `m2-agent-loop-scaffold-v2__…dd2d728e` | **INTACT** (untouched) | 35 | **32** |
| `runs/automationbench-aa-scaffold__c5731755a377__…31b049d1` | **NEW** re-run | 34 | 32 |

The untouched `m2-agent-loop-scaffold-v2` run is the *original* part-(i) run: it
produced the same numbers (16 attempted / 16 passed / 0 failed / 0
execution_errors / 48 turn rows / 16 outcome rows / 32 tool calls / accounting
valid). So the claim's evidence existed all along in a second run directory; my
`git add` had simply never reached it either.

The recovery probe also **re-ran** the scaffold (deterministic — outputs come
from the built-in toolbox, not an external service) into a short root so every
path fits, and emitted `EVIDENCE_MANIFEST.json` with `sha256` + `bytes` for all
34 artifacts. `MAX_PATH` is no longer a factor: longest path **182** chars.

## 4. What this changes about prior claims

- The M2 part-(i) **verdict is unchanged** — `attempted 16, passed 16, failed 0,
  execution_errors 0`, `accounting valid`, reproduced three times independently.
- What changes is that the evidence is now **committed**, not asserted. Before
  this commit, `file_output_written: True` pointed at files absent from the
  repository.
- The destroyed run dir keeps its `receipt.json` (1910 B), so the run is still
  identifiable; its artifacts are gone and must not be cited.

## 5. Rules adopted

1. **Never `rm -rf` a glob-named target inside a loop that also creates that
   name.** Move with a fixed short name outside the glob, or copy-then-verify.
2. **Assert the file count before and after any path-rewriting step**, and abort
   when the count drops.
3. **`git add` warnings are failures.** Treat `Filename too long`, `could not
   open directory`, and `did not match any files` as fatal, and verify
   `disk_count == HEAD_count` for every run directory cited in a receipt.
