# M3 RESOLVED — the SciCode harness runs on the sandbox contract

**Date:** 2026-09-16 | **Tree:** `carrier/aaii-v43` @ `f3fecfe44c5b87b2b610e06b0ead172982578f2d`
**Device:** CPU (`cuda_available=False`) | **Class:** `INSTRUMENT_CONTROL` — **not a capability claim**

---

## 0. The gap this closes

The audit's M3 read:

> **M3 — Isolated code-execution harness (unlocks 20%).** Measured state: the
> `--sandbox-mode` two-mode contract exists (`namespace` probe gate +
> `container-rlimit` surrogate), but **no SciCode/Terminal-Bench harness runs on
> it.**

The second half of that sentence is now false, by execution.

## 1. Verdict

```
INSTRUMENT_VALIDATED
positive control            2/2 PASSED
unattributed rows           0      (was 16 before the classifier fix)
infrastructure errors       0      (after the provisioning + resource fixes)
reconcile                   {'valid': True, 'problems': []}
mode                        container-rlimit   isolated=False  surrogate=True  rlimits=True
```

**The positive control:** SciCode's **own** `ground_truth_code` fed to SciCode's
**own** published `test_cases`, inside `SandboxHarness`. If the reference
solution failed its reference test, the instrument would be broken and every
later number void. It passed.

## 2. Window attribution (16 sub_steps × 2 protocols)

| Attributed cause | Count | Meaning |
|---|---|---|
| `BLOCKED_EXTERNAL_CONSTANT` | 14 | published test needs external `target` |
| `BLOCKED_INFRA_RESOURCE` | 12 | OpenBLAS allocation failure (see §3) |
| `ISOLATION_ARTIFACT(...)` | 4 | **my** protocol error (see §4) |
| `BLOCKED_DEPENDENCY(scicode)` | 2 | SciCode's own helper module, not in the HF split |
| `UNATTRIBUTED` | **0** | attribution complete |

## 3. Real infrastructure defect found and fixed

```
symptom : OpenBLAS error: Memory allocation still failed after 10 retries, giving up.
cause   : thread-pool over-allocation in the sandboxed child
fix     : OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1,
          OPENBLAS_DEFAULT_NUM_THREADS=1, mem_limit_mb=4096
result  : 8 of 12 resource-blocked items CLEARED
```

After the fix those items resolved to `BLOCKED_EXTERNAL_CONSTANT` — i.e. the
resource failure was **masking** the dataset limitation. This is a K-B finding:
an infrastructure defect, correctly not reported as a capability result.

## 4. My own defects, all disclosed

| # | Defect | Evidence | Fix |
|---|---|---|---|
| D16 | Ran sub_steps **in isolation**; SciCode sub_steps are **sequential** | `NameError: get_lattice_coords` in isolated arm only | added `accumulated` protocol |
| D17 | `BLOCKED_EXTERNAL_CONSTANT` classifier searched `stderr[:180]`; traceback header fills that window | classifier fired 0× in v1, 5× after fix | search full text |
| D18 | `target` scan covered `sub_steps` only, never `general_tests` | 0 → 54 entries scanned | scan both |
| D19 | `passed = getattr(res, "passed", False)` **unverified**; could have made every count 0 | `SandboxResult.passed` is a property; calibrated in-run | derive from `STATUS_PASSED` |
| D20 | Reported 11 `BLOCKED_DEPENDENCY` without cause | `-I` implies `-s`; scipy/sympy live in **user** site-packages | dedicated venv |
| D21 | Persisted 8 `FAILED` rows **without stderr** → unattributable | could not read back | persist full stderr before aggregation |
| D22 | Classifier lacked resource + isolation-artifact branches | 16 `UNATTRIBUTED` | added both, with a falsifiable test for the isolation case |

## 5. Dataset limitation (not a harness limitation)

```
self-contained assertions in problems_dev.jsonl:
    sub_steps     : 2 of 50   (78.1, 78.2)
    general_tests : 0 of 54
```

Every other published test references an external `target` constant that the dev
split **does not contain** — it is supplied by SciCode's official grader. Two
consequences recorded honestly:

1. The positive control is limited to **2 items**. It validates the instrument; it
   cannot exercise the other 14.
2. Injecting a local tolerance would **substitute my grader for the official
   one**. Refused.

Problem `10.7` additionally needs a module named `scicode`, which ships in
SciCode's GitHub repo, not the HF dataset.

## 6. Sandbox boundary (unchanged, stated plainly)

```
mode=container-rlimit  isolated=False  surrogate=True  rlimits=True
downgrade_reason: namespace isolation NOT PROVEN: unshare-direct rc=None:
                  cannot spawn '/usr/bin/unshare': [WinError 2]
```

The harness reports `isolated=False` itself. It is a **resource-limited
surrogate**, not a namespace sandbox. Any result produced under it carries that
qualification.

## 7. What this does and does not license

**Does:** M3's stated gap ("no SciCode harness runs on it") is closed. The
pipeline can execute SciCode items and attribute every outcome.

**Does not:** license any score. HENRI's egress cannot yet produce a candidate
program (A2: no id→string binding), so `pass@1` against these tests is not
currently measurable. This run measured the **instrument**, using dataset code.

## 8. Reproduce

```bash
V2="C:/Users/chan/henri-worktrees/aaii-v43/HENRI V2"
cd "$V2" && env -u VIRTUAL_ENV -u PYTHONHOME PYTHONPATH="$V2" \
  C:/Python314/python.exe "$LOCALAPPDATA/Temp/m3_close.py"
```

Artifacts: `experiments/verification/scicode-instrument-control-close__f3fecfe44c5b__<ts>/`
(`receipt.json`, `items.jsonl` — 42 persisted rows).
