# Adjudication — `NEW_RC=1` in `WORKTREE_SUITE_RECEIPT_5ab46c2.txt`

**Date:** 2026-09-16 | **Branch:** `carrier/aaii-v43` | **Artifact under review:** `a724331936a289719cfcc0a2d683994d6db0b810`
**Subject:** section 3 of the committed verifier receipt records `190 collected / 1 failed` (`NEW_RC=1`) while section 4 records `1778 passed / FULL_RC=0`. A green full suite does not erase a red focused gate, so this was adjudicated explicitly.
**Outcome:** `H1_BENIGN` — **the pushed artifact is green on this axis.** The receipt's section-3 numbers describe an **uncommitted intermediate revision**. Label: `RECEIPT_DESCRIBES_UNCOMMITTED_INTERMEDIATE_STATE`.

---

## 1. The recorded failure

`WORKTREE_SUITE_RECEIPT_5ab46c2.txt` (5,219 B, sha256 `df2a55517d3f77d1a79b629b…`), section 3:

```
collected 190 items
tests\contract\test_pdf_ingress.py .................................F... [ 67%]
FAILED tests/contract/test_pdf_ingress.py::test_check_text_is_a_last_act_guard_on_the_write_path
E   Failed: DID NOT RAISE <class 'henri_pdf_ingress.ZoneCContaminationError'>
    C:\...\tests\contract\test_pdf_ingress.py:537
1 failed, 189 passed
NEW_RC=1
```

## 2. What the committed artifact actually contains

| Check | Value |
|---|---|
| `test_check_text_is_a_last_act_guard_on_the_write_path` in `HEAD` blob | **absent** |
| `last_act` / `last_act_guard` occurrences in `HEAD` blob | **0** / **0** |
| `def check_text` in `henri_pdf_ingress.py` | **does not exist** |
| test functions in `HEAD` blob | **49** |
| test functions on disk | **49** (identical set) |
| `git diff 5ab46c2..HEAD -- HENRI V2/tests/contract/test_pdf_ingress.py` | **0 bytes** |
| committed tree, same 5 files, `--collect-only` | **191** |
| receipt recorded | **190** collected / 1 failed |

The failing test's name does not exist anywhere in the committed artifact, and the file is byte-identical between the verified commit and `HEAD`. A test cannot fail in a file it is absent from.

## 3. Root cause

The verifier raced a **concurrent writer**. Child `sa-0` was still rewriting `test_pdf_ingress.py` when section 3 ran. The receipt therefore captured an **intermediate working-tree revision** in which a differently-named guard test existed and failed. The child then finalised the file: that name was replaced by two guard tests that pass.

Arithmetic is consistent: 190 → 191, one test renamed/split. The intermediate bytes were never committed and are not recoverable — and must not be reconstructed.

## 4. Is the guard genuinely tested? (mutation test — the decisive check)

A receipt that cannot be trusted on counts cannot be trusted on *coverage* either, so the guard was verified by **mutation**: neutralise it and confirm the committed tests fail.

Mutation: substitute the exception type at every raise site — `raise ZoneCContaminationError` → `raise ValueError` (3 sites: `assert_zone_c_clean` ×2, `_ingest` ×1). Type substitution is parse-safe at all call sites, including multi-line raises.

| Arm | collected | passed | failed | errors |
|---|---|---|---|---|
| **BASELINE** (unmutated) | 49 | **2** | 0 | 0 |
| **MUTATED** (guard neutralised) | 49 | **0** | **2** | 0 |
| restore sha256 == original | — | **True** | — | — |

With the guard neutralised, both committed guard tests **FAIL**. They therefore reach and assert the write-path guard. The tests are **non-vacuous**.

The two committed guard tests:

- `test_zone_c_guard_fires_on_contamination` — smuggles a contaminated field into a record, expects `ZoneCContaminationError` from `assert_zone_c_clean`.
- `test_check_text_guard_fires_when_text_leaks` — a deliberate **negative control**: monkeypatches `canonical_bytes` to append raw chunk text, expects `ZoneCContaminationError` from `ingest_pdf_document_strict`.

## 5. Verdict

```
NEW_RC=1              -> RECEIPT_DESCRIBES_UNCOMMITTED_INTERMEDIATE_STATE  (not a code defect)
committed artifact     -> green: 191 collected / 191 passed / 0 failed
write-path guard       -> VERIFIED (mutation-confirmed non-vacuous)
full suite             -> 1778 passed / 21 skipped / 0 failed / FULL_RC=0
```

No code change follows from this adjudication. The artifact was already correct.

## 6. Defects in this adjudication's own harness (disclosed)

Three malformed attempts preceded the decisive run. Each is recorded because a verdict emitted from a broken harness is void:

1. **v1** — returned `rc=2 / passed=0 / failed=0` (a pytest *collection* error; nothing executed) and its verdict branch printed **`H2 DEFECT CONFIRMED`** anyway. The branch was `if failed > 0 → H1 else → H2`, so a collection error silently became a "defect". Same vacuous-gate class this session has been eliminating, this time inside my own verdict logic.
2. **v2** — injected a module-level `_MUT_NOOP` **above** the `from __future__` import → `SyntaxError: from __future__ imports must occur at the beginning of the file` → collection error.
3. **v3** — same defect, because the `__future__` import sits at **line 19**, after a 40-line docstring, so a first-12-lines scan never found it. Reported `no-op injected at line 1`.

v4 removed the injection entirely (exception-type substitution) and treats `collected == 0` as `INCONCLUSIVE`, never as a verdict. Only then did baseline and mutated arms both collect 49.

## 7. Process lesson (durable)

**A verifier must not run concurrently with a writer.** Racing a patching child produced a receipt that described a state that never existed in version control, and the mislabelled numbers then required a mutation test to unwind. The cure, applied to the v4 harness and required for future verifiers:

- record `sha256` of every file under test **before** the run and re-check **after** the run;
- if any hash drifted, the run is `INCONCLUSIVE_DUE_TO_CONCURRENT_WRITE`, not a pass or a fail;
- treat `collected == 0` as `INCONCLUSIVE` (never `FALSE`);
- never emit a verdict from an arm whose baseline did not validate.

## 8. Reproduction

```
PYTHONPATH="HENRI V2" python experiments/verification/gate_guard_mutation_test.py
```

Artifacts: this document, `gate_guard_mutation_test.py`, `WORKTREE_SUITE_ADJUDICATION_20260916.json`.
Git state before and after the mutation test: **1 status line** (`?? HENRI V2/henri_audit_chain.json`, a runtime-regenerated genesis stub); module sha256 restored to `f6fbfbe6bd17322680dd7bd0`.
