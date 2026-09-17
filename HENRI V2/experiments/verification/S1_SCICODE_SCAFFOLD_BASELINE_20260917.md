# S1 — SciCode scaffold executed on the LOCAL CPU runner

**Date:** 2026-09-17 | **Tree:** `carrier/aaii-v43` @ `ece2710105ae13ed2ee2a9adc8eba8012ac22e31`
**Device:** CPU only (`cuda_available=False`, no network) | **Class:** `INSTRUMENT_CONTROL + BASELINE`
**Runner:** `experiments/verification/s1_scicode_scaffold_runner.py`
**Receipt:** `experiments/verification/s1-scicode-scaffold__ece2710105ae__20260917T213902Z-40d1ca46/receipt.json`
(`items.jsonl` sits beside it; raw `*.jsonl` is gitignored, so the receipt also carries a
per-item digest with taxonomy, evidence line and stderr sha256 for all 16 rows.)

## Verdict

```
SCORED          controls passed and the dataset pin matched
pass@1          0 / 2          (window 16, attemptable 2)
controls        7 / 7 behaved
guards          4 / 4 self-tested as designed
sandbox         mode=container-rlimit  isolated=False  surrogate=True
```

## The zero is a TRUE zero, not an instrument failure

Three independent facts establish this:

1. **The candidate path is capable of PASS.** `PC-U2` and `PC-U4` feed a *control-only*
   synthetic generator the reference solution for 78.1 / 78.2 **through the candidate-arm
   code builder**, and both return `PASSED`. The candidate path is therefore not
   structurally incapable of scoring.
2. **The reference path passes.** `PC-D1` / `PC-D3` run SciCode's own `ground_truth_code`
   against SciCode's own published tests through the same harness: 2/2 PASSED.
3. **The candidate arm emits nothing.** The only registered candidate source is
   `NullCandidateSource` (`produces_code=False`), which returns `""` for every subproblem.
   All 16 candidate rows fail with `CANDIDATE_UNDEFINED_SYMBOL:<fn>` — the payload defines
   no function at all.

The denominator is 2 because **2 of 50** dev sub_steps ship self-contained published tests
(`78.1`, `78.2`). 48 of 50 sub_steps reference an external `target` constant that the dev
split does not contain; SciCode's official grader supplies it. Items whose *own reference
solution* cannot pass their *own published test* are not attemptable and are excluded from
the score — not silently dropped, but reported as `BLOCKED_EXTERNAL_CONSTANT` (13) or
`BLOCKED_DEPENDENCY(scicode)` (1, item 10.7, whose helper module ships in SciCode's GitHub
repo and not in the HF split).

**No local grader, tolerance, or re-derived expected value was substituted anywhere.**

## Failure taxonomy (16 rows, full text search over stderr)

| Arm | Class | n |
|---|---|---|
| candidate | `ERROR_OTHER` / `CANDIDATE_UNDEFINED_SYMBOL` | 15 |
| candidate | `BLOCKED_DEPENDENCY(scicode)` | 1 |
| reference | `STATUS_PASSED` | 2 |
| reference | `BLOCKED_EXTERNAL_CONSTANT` | 13 |
| reference | `BLOCKED_DEPENDENCY(scicode)` | 1 |

`unattributed_rows = 0`. Every failure row persisted **full** stderr before aggregation
(no truncation).

## Dataset pin

| file | sha256 | pin |
|---|---|---|
| `problems_dev.jsonl` | `193968aff23b7ed931c8f6d196b611e2f6694adb49e6cc085444a60ad2fc4f7b` | MATCH |
| `problems_test.jsonl` | `38797fef78f434720be6d053b4f3a86839d6f8ea5fb9115450677cd3a6edf81d` | MATCH |

Revision `4510f6a6aa27c43fad7b43da2c59602a86e88480`, license Apache-2.0, gated false.

## Interpreter / sandbox imports (D20)

```
chosen : C:\Users\chan\AppData\Local\Temp\m3_bench_venv\Scripts\python.exe
  numpy   2.5.3   m3_bench_venv\Lib\site-packages\numpy\__init__.py
  scipy   1.18.1  m3_bench_venv\Lib\site-packages\scipy\__init__.py
  sympy   MISSING (ModuleNotFoundError)
rejected: C:\Python314\python.exe
  numpy   2.4.6   C:\Python314\Lib\site-packages\numpy\__init__.py
  scipy   MISSING
```

Importability is proven by a probe that runs **inside the sandbox child** before any item
executes; the harness forces `PYTHONNOUSERSITE=1` and `-I` implies `-s`, so user
site-packages are invisible to the child. `sympy` is genuinely absent from the sandbox and
is reported as such; a declared-dependency scan shows the dev split needs only
`numpy`, `scipy`, `math`, `cmath`, `time`, so no dev-split item is blocked by it.

## Defects avoided (from the M3 control) and found

| # | Avoided / found | How |
|---|---|---|
| D16 | avoided | reference arm uses the **accumulated** protocol (SciCode sub_steps are sequential); no isolated arm is run, so the isolation artifact cannot recur |
| D17 | avoided | every classifier branch searches the **full** stderr |
| D19 | avoided | counts derived from `henri_sandbox_harness.STATUS_PASSED` vs `result.status`; the `passed` **property** is recorded for calibration only and agreed on 16/16 rows |
| D20 | avoided | dedicated venv + in-sandbox import probe; interpreter and resolved module paths recorded |
| D21 | avoided | full stderr written and fsynced **before** aggregation; aggregation reads the ledger back; 0 truncated, 0 unattributed |
| **new** | found | the first draft of the guard self-test itself read reference text outside the reference arm. **Guard G1 raised and killed the run.** Fixed, and a second draft bug (a `prior_defs` helper reading reference text outside the reference arm) was caught by the same guard. The guards demonstrably fire when violated. |
| **new** | found | a `NameError` in the candidate arm was being attributed to the environment (`UNDEFINED_SYMBOL`) when the symbol was in fact missing from the candidate payload. Now attributed from the payload's own bindings as `CANDIDATE_UNDEFINED_SYMBOL`. |

## Non-claims

1. This is an **instrument validation and a baseline**, NOT evidence of HENRI capability.
2. No code generator is wired to SciCode (text egress under repair; A2 egress unbinder).
   The `0/2` is the floor for an **absent** generator.
3. **No claim about the AAII index**, its weights, or any constituent.
4. Not a namespace-isolated run: `container-rlimit`, `isolated=False`, `surrogate=True`.

## Reproduce

```bash
V2="C:/Users/chan/henri-worktrees/aaii-v43/HENRI V2"
cd "$V2" && env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME \
  PYTHONPATH="$V2" PYTHONDONTWRITEBYTECODE=1 \
  C:/Python314/python.exe "$V2/experiments/verification/s1_scicode_scaffold_runner.py"
```

Deterministic: two consecutive runs produced identical verdicts, taxonomy counts and score.
