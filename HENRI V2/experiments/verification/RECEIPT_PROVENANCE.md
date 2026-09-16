# Receipt provenance: which tree each number describes

Two suite receipts are committed together. They describe **different trees**.
Neither is wrong. Quoting one without its tree is.

## 1. `suite_receipt_238408c_post_class49.txt` -- DEVELOPMENT tree (not reproducible from the commit)

```text
1568 passed, 20 skipped, 0 failed, EXIT=0, 149.01s (CPU)
source : $LOCALAPPDATA/Temp/suiteC.txt
sha256 : 2babcdc56c61a1a057cb2dd653f24de28993370d82ce957cab6a46148bc52233
```

Measured on the operator's development worktree, which carries **196 porcelain
lines of pre-existing state**, including untracked test files and the untracked
`basal_*` module family. Those untracked files contribute roughly 172 collected
tests that are in **no commit**. This receipt therefore does **not** describe the
committed tree, and must not be quoted as if it did.

## 2. `suite_receipt_clean_worktree.txt` -- COMMITTED tree (AUTHORITATIVE for this carrier)

```text
1395 passed, 21 skipped, 0 failed, EXIT=0, 124.15s (CPU)
collected: 1416 tests, 0 collection errors
source : $LOCALAPPDATA/Temp/suite_wt_fixed.txt
sha256 : ca241c81e80881c13b5640930a4a467f11f749c78eb068bb632959ad56408385
```

Measured in an **external clean worktree**: a detached checkout of the carrier
commit with only committed files present. This is what a fresh clone sees.
**Quote this number for the carrier.**

## The defect this file exists to record

The first attempt (`b67af79` -> `c1ddea6`) added
`tests/unit/test_basal_boundary_bundle.py`, an **untracked** test whose four
imports are untracked and absent from the repository:

```text
basal_boundary_engine, epsilon_band_gate,
koopman_action_ledger, unified_henri_vla_engine
```

In the clean worktree that tree aborted at collection:

```text
ERROR tests/unit/test_basal_boundary_bundle.py
ModuleNotFoundError: No module named 'basal_boundary_engine'
Interrupted: 1 error during collection          (exit code 2)
```

The development suite passed only because those four modules happen to exist on
local disk as untracked files. A commit was therefore pushed whose own test
suite could not be collected from a fresh clone. Commit `663d6f2` removes that
one path; the clean-tree receipt above is the verification that removal restored
collection.

**Lesson.** Editing a file does not authorize committing it, and a suite count is
evidence only when the tree it was measured on is reconstructible from the
commit.

## Line endings (disclosed, NOT normalized)

Five committed files carry a CRLF-to-LF flip versus `238408c`, so their raw diffs
show whole-file rewrites while the normalized diff is small:

| file | raw | normalized |
|---|---|---|
| `production_arc_run.py` | 3266+/3250- | 22+/6- |
| `darwinian_phase_swarm.py` | 718+/702- | 21+/5- |
| `tests/contract/test_f6_adaptive_functor.py` | 284+/208- | 81+/5- |
| `tests/contract/test_f7_affine_egress.py` | 281+/222- | 60+/1- |
| `tests/unit/test_henri_calibrator_ingest.py` | 151+/138- | 15+/2- |

The CRLF state is **pre-existing in the development worktree**, not introduced by
these edits: `test_f6`, `test_f7` and `pre-commit-seal-gate.sh` were not edited
this session. The working tree was preserved byte-for-byte rather than
normalized, because normalizing it would modify the operator's uncommitted state.
Content is unchanged -- the CRLF-normalized and whitespace-normalized diffs agree
exactly.

A `--renormalize` pass plus a `.gitattributes` policy are **recommended for the
separate main-promotion step**. They are not applied here: rewriting a pushed
branch requires approval, and force-push is prohibited without it.

## Scope of every claim above

CPU only. No CUDA verification was possible: Vast `50797414` is EXITED and
unfunded, and the production Zone C endpoint `:10100` is closed. All numbers are
`OBSERVED` on CPU and the live **dev** Zone C database. Nothing here is a
capability, score, or benchmark claim.
