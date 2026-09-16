# Receipt provenance: which tree each number describes

**Three** suite receipts are committed together. They describe **different trees**.
None is wrong. Quoting one without its tree is.

## 1. POST-MERGE tree at `3660e8ef` -- AUTHORITATIVE for `main`

```text
========== 1587 passed, 21 skipped, 22 warnings in 129.90s (0:02:09) ========== (CPU)
collected : 1608 tests, 0 collection errors
commit    : 3660e8efb38322d923ba83701fed90e6bf2da009
source    : $LOCALAPPDATA/Temp/tip_suite.txt
sha256    : 5603c3daa533aadedbba6428151b562f523105facaacc918f476777b59efe7e8
artifact  : HENRI V2/experiments/verification/suite_receipt_merge_3660e8ef.txt
```

> **Hash provenance (corrected 2026-09-16).** An earlier revision of this file
> published `sha256: 3b0d145b…` for this receipt. That is the hash of the **scratch**
> file `$LOCALAPPDATA/Temp/tip_suite.txt`, NOT of the artifact a clone receives. The
> receipt is tracked `i/lf`, so `core.autocrlf=true` stripped 252 CR bytes at commit
> time: scratch is 20,696 B, the committed blob is 20,444 B, and
> `blob == scratch` with every CR byte removed holds exactly. A verifier hashing the
> committed receipt got `5603c3da…` and read the published hash as corruption.
>
> **The hash above is the BLOB hash and is the one to verify.** Reproduce it with:
>
> ```bash
> git cat-file -s main:HENRI\ V2/experiments/verification/suite_receipt_merge_3660e8ef.txt   # 20444
> git show    main:HENRI\ V2/experiments/verification/suite_receipt_merge_3660e8ef.txt | sha256sum
> # 5603c3daa533aadedbba6428151b562f523105facaacc918f476777b59efe7e8
> ```
>
> Receipts 2 and 3 below are unaffected: both are tracked `i/crlf`, so their blobs
> kept CRLF and their published hashes already match their blobs (`MATCH: True`,
> verified byte-for-byte). Only the newly added receipt was normalized.

Measured at the **committed tip** of the promotion branch, in an external worktree
whose porcelain was 0 lines before the run. The carrier branch tip `3ae27c7` and
`main` `85149cf` are both ancestors of this commit.

**Quote this number for `main`.** The two receipts below are historical.

## 2. Carrier tree at `3ae27c7` -- pre-merge (historical)

```text
1395 passed, 21 skipped, 0 failed, EXIT=0, 124.15s (CPU)
collected: 1416 tests, 0 collection errors
source : $LOCALAPPDATA/Temp/suite_wt_fixed.txt
sha256 : ca241c81e80881c13b5640930a4a467f11f749c78eb068bb632959ad56408385
```

Measured in an external clean worktree of the carrier commit, before `main` was
merged in. Superseded by receipt 1, which contains this tree plus `main` plus the
resonator and DAG carriers.

## 3. DEVELOPMENT tree -- not reproducible from any commit (historical)

```text
1568 passed, 20 skipped, 0 failed, EXIT=0, 149.01s (CPU)
source : $LOCALAPPDATA/Temp/suiteC.txt
sha256 : 2babcdc56c61a1a057cb2dd653f24de28993370d82ce957cab6a46148bc52233
```

Measured on the operator's development worktree, which carries **196 porcelain lines
of pre-existing state**, including untracked test files and the untracked `basal_*`
module family. Those files contribute roughly **172 collected tests that are in no
commit**. This receipt does **not** describe a committed tree and must not be quoted
as if it did.

---

## The promotion itself

`main` was advanced by a merge of `main` (`85149cf`) INTO the carrier line
(`3ae27c7`), because `main` could not be fast-forwarded: the two had diverged
(`main..carrier` = 40, `carrier..main` = 9, merge-base `10f5f23`). Force-push was
prohibited. The merge result is a **fast-forward** for `origin/main`, so the push
itself needed no rewrite of published history.

Commits on the promotion branch, in order:

| commit | content |
|---|---|
| `1e8dabe` | merge `main` into the carrier line; 3 conflicts resolved |
| `eb53532` | K1-K4 kill-condition pre-registration (landed BEFORE the mechanism) |
| `a355c36` | tripartite VSA resonator carrier + kill-gate contract tests |
| `b7090cc` | Zone C artifact DAG migration + contract tests (ledger v4) |
| `3660e8e` | post-merge semantic regression fix (see below) |

### The three conflicts, and why each resolution is safe

A first pass reported **9** conflicts. That was wrong: the count came from
`grep -cv` on `merge-tree` output, which counts its informational lines too. The
authoritative count from `git merge` is **3**.

| path | kind | resolution | evidence it loses nothing |
|---|---|---|---|
| `.gitignore` | content | **union** | both policy blocks kept; `main`'s basal-archive block was 10 lines, appended verbatim |
| `HENRI V2/basal_triton_kernel.py` | add/add | carrier | diff `main`->carrier = 2 removed / 41 added; both sides define the same 16 functions; the carrier replaces the literal `block_span: int = 64` with the measured `BLOCK_SPAN_DEFAULT = 128` |
| `HENRI V2/zone_bc_engram_sync.py` | add/add | carrier | diff = 5 removed / 36 added; the removed lines are the superseded 3-argument contract text, replaced by the CLASS49 7-argument form |

Neither add/add resolution drops a unique function: both files define 16 functions on
both sides.

### The post-merge regression, and the fix

The full merged-tree suite measured **1 failed, 1586 passed, 21 skipped**. The
failure was **caused by the conflict resolution**, not by either side alone:

```text
FAILED tests/unit/test_basal_boundary_bundle.py::TestDecoupledEngramSync
       ::test_digest_round_trips_through_a_live_store_contract
       assert committed["digest"] == env.digest
       E   KeyError: 'digest'
```

`main` documents a **dual** store contract -- a minimal
`write_engram(wave, domain, sagnac_stress)` **or** a `commit(envelope)` callable. The
carrier side of the merge hard-coded the CLASS49 7-argument call, making one of the
two forms mandatory. `main`'s test uses the 3-argument stub, so `drain()` raised
`TypeError`; `drain()` swallows that in its `except Exception` handler, so the fault
surfaced far from its cause as a missing dict key.

This is the exact class the merge could not catch: **both files changed, git merged
them with no conflict markers, and the behaviour still broke.** A clean textual merge
says nothing about whether behaviour survives.

`3660e8e` replaces the hard-coded call with contract-width discovery from the
callable's real signature (`_accepts_provenance`). A blind `except TypeError` retry
was deliberately rejected -- it would also swallow a genuine `TypeError` raised
*inside* a correct 7-argument store. When the minimal form is used the event is
**counted**, not hidden: `provenance_undeliverable` is exposed in `telemetry()`,
because a silent 3-argument commit is precisely the "unattributed row wearing a
provenance costume" pattern CLASS49 exists to prevent.

Verified both directions: 3-arg -> `_accepts_provenance` False, commit succeeds,
digest reproduced, counter 1. 7-arg -> True, `run_id`/`arm_id`/`commit_sha`/
`domain_family` all delivered, counter 0.

## The push gates actually run (before the push, not after)

Run by `HENRI V2/scripts/maintenance/check_test_collection.py`, added in this step so
the check is repeatable rather than ad hoc.

| gate | result |
|---|---|
| suite at the committed tip | 1587 passed, 21 skipped, 0 failed, EXIT=0 |
| `--collect-only` | 1608 tests collected, 0 collection errors |
| collection coverage (scope = `python_files = test_*.py`) | 144 / 144 collectible modules collect |
| import closure | 0 edges to untracked local modules |
| new carrier files tracked | 6 / 6 |

`tests/contract/check_static_partition_wiring.py` is tracked but collects nothing. It
has no `test_` prefix, defines **zero** `test_*` functions, and carries a `__main__`
block: it is a standalone verifier excluded by the project's own `python_files`
setting. An earlier revision of the gate demanded its collection, which was an
over-broad rule rather than a defect in the file. The gate now scopes to the
configured pattern and reports such files instead of failing on them.

## Line endings (disclosed, NOT normalized)

`core.autocrlf = true` on this host. Measured with `git ls-files --eol` at the
promotion tip: 1096 tracked files, **1049 CRLF in the working tree**, **10 CRLF in the
INDEX** (which is the real canonical-LF violation, since the index is what a clone
receives):

```text
HENRI V2/darwinian_phase_swarm.py
HENRI V2/production_arc_run.py
HENRI V2/tests/contract/test_f6_adaptive_functor.py
HENRI V2/tests/contract/test_f7_affine_egress.py
HENRI V2/tests/unit/test_henri_calibrator_ingest.py
HENRI V2/experiments/verification/collection_receipt_clean_worktree.txt
HENRI V2/experiments/verification/field_channel_manifest.json
HENRI V2/experiments/verification/suite_receipt_238408c_post_class49.txt
HENRI V2/experiments/verification/suite_receipt_clean_worktree.txt
HENRI V2/experiments/verification/zone_abc_cpu_e2e_observed.json
```

`.gitattributes` currently pins exactly two paths to `eol=lf`, both added for a
byte-pinned SHA-256 assertion in `test_arc_k3_koopman.py`.

**A `--renormalize` pass is NOT applied here, deliberately.** It rewrites file
content, several tests assert byte-level SHA-256 of files read with `read_bytes()`,
and rewriting published history requires its own approval. It remains an open,
bounded item with the exact file list above. The CRLF state is pre-existing: it is
carried over from the development worktree and none of these ten files was edited to
introduce it.

## Scope of every claim above

CPU only. No CUDA verification was possible: Vast `50797414` is EXITED and unfunded,
and the production Zone C endpoint `:10100` is closed. The resonator kill gates and
the DAG contract tests are `ENGAGED_WIRING_ONLY`: they prove wiring and invariants,
never capability or task score. The DAG tests ran against the live **dev** Zone C
database at `:5434`; the production endpoint was not touched.

Nothing here is a capability, score, or benchmark claim. The two composite indices
`AAII_Composite_Intelligence_Index = 0.7262` and `ARC_AGI_3_RHAE_Score = 0.7347` were
withdrawn as score literals and must not be cited.
