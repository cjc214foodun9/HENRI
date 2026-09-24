# UHR-05 — SciCode on the OFFICIAL grader: 48-item window, pass@1 5/48

**Date:** 2026-09-24 · **Author:** HENRI arbiter, from own tool calls
**Run:** `s1-scicode-scaffold__16d573ccf1c3__20260924T215943Z-59e7c591`
**Commit:** `87efc9707067` · **Evidence classes:** `OBSERVED` / `DERIVED` / `FALSIFIED`

## What changed and why

The scaffold could reach a denominator of only **2** because `target` — the value every
published test asserts — is assigned **zero times** in either corpus split
(dev: 219 uses / 0 assignments; test: 1172 / 0). Excluding those items was correct
protocol (`s1_scicode_scaffold_runner.py:18-22`: *"No local tolerance, no re-derived
expected value, no local grader injection"*).

But "unavailable" meant *not shipped with the corpus*, not *does not exist*. The
benchmark's OWN grader ships it:

| Piece | Value (`OBSERVED`) |
|---|---|
| Grader repo | `github.com/scicode-bench/SciCode`, clone `rc=0`, `pip install -e .` → `import scicode` OK |
| Numeric results | `eval/data/test_data.h5`, **1 049 345 865 B**, 338 top-level groups |
| Accessor | `src/scicode/parse/parse.py:126 process_hdf5_to_tuple(step_id, test_num, h5py_file)` — unmodified |
| h5_sha256 | `48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890` |
| revision | `4510f6a6aa27c43fad7b43da2c59602a86e88480` (matches the committed dataset pin) |

The data file is far larger than GitHub's 100 MB limit, so it is fetched from the
project's published Google Drive and its digest is pinned in the receipt.

## Result (flag `HENRI_SCICODE_OFFICIAL_TARGETS=1`, window `--window 48`)

```
verdict     SCORED     reason "controls passed and the pin matched"
score       5 / 48     pass@1 = 0.104167          window_size 48
reference   {"STATUS_PASSED": 48}                 <- SciCode's OWN solutions, OWN tests, 48/48
candidate   {"FAILED_ASSERT": 16, "ERROR_OTHER": 25,
             "BLOCKED_DEPENDENCY": 2, "STATUS_PASSED": 5}
official    items_considered 96 · items_targeted 96 · injections 98 · refusals 0 · resolver_error ""
controls    True · guards True · accounting.valid True · execution_error_count 0
```

**`NOT CLAIMED`:** this is **not** HENRI capability. The candidate arm is a local
pretrained backbone (`Qwen/Qwen2.5-1.5B-Instruct`), labelled
`BACKBONE_BASELINE_NOT_HENRI_CAPABILITY` in the receipt. No AAII v4.3 score is claimed
(75 % of that index's weight is externally graded).

**The reference arm is the instrument validation.** 48/48 `STATUS_PASSED` means the
sandbox, the published tests, and the injected `target` all resolve on the SAME items
the candidate is scored on. Without it, a candidate zero would be attributable to the
harness rather than to the generator.

## Counter derivation: 96 and 98 are NOT a double-count (settled)

An earlier read flagged `injections 98` vs `items_targeted 96` as possibly double-scoring
an item, which would have invalidated the denominator. Measured, both numbers are
exactly explained by the code and the ledger:

```
dev corpus           50 sub-steps · 48 carry `target` · 2 do NOT (SciCode-78-78.1, -78.2)
window               48 items · 47 carry target · 1 does not · refusals 0
items.jsonl          48 rows    · 48 unique item_ids   · 0 duplicate (arm,item_id)
double-wrap check    lines where `_official_tests(` appears twice = 0

MEASURED by replicating both counting phases against the live module
(import the runner, call select_window(dev, n=48), then replay the :778-779
reference loop):

  window select (:524)     delta considered +48  targeted +48  injections  0  refusals 0
  reference pairs (:778)   50 pairs · _official_tests calls 50 · injections +50
  TOTALS after both         considered 96  targeted 96  refusals 0
  RECEIPT                   items_considered 96  items_targeted 96  refusals 0   -> MATCH

items_targeted 96 = 48 (window select) + 48 (reference scan)          [MEASURED]
injections     98 = 50 (reference arm: every dev sub-step is h5-resolvable)
                     + 48 (candidate arm: one _official_tests call per window item)
                     -> the 48 term is DERIVED as 98 - 50, and equals the 48 ledger rows
```

`_targets_satisfiable` counts only items whose published tests reference `target`
(the early return at `:473-474` counts nothing): 50 calls, 48 counted, 2 uncounted
(`78.1`, `78.2`) — measured exactly. `_official_can_target` succeeds on all 50 because
the h5 covers every dev sub-step, which is why the reference arm injects 50, not 48.
**The denominator of 48 is a set of 48 unique items** (48 ledger rows, 48 unique
`item_id`s, 0 duplicate `(arm, item_id)` pairs).

`FALSIFIED`: a claim that the candidate call site double-wraps
(`_official_tests(_official_tests(...))`). Verified: **0** lines call it twice.

## Defects found and repaired (all by building a check, not by inspection)

| # | Defect | How it surfaced |
|---|---|---|
| 1 | `items.jsonl` was never validated; a uniform `test_num=4` read a **complete** corpus as `17/50` (33 sub-steps ship 3 tests, 17 ship 4) | graded the grader's own accessor over all 50 dev sub-steps |
| 2 | The scored window was **not controllable** — `add_argument` count 0, so a wider `--n-prompts` was silently ignored | compared the flag I passed against `[3] WINDOW n=16` in the run output |
| 3 | `non_claims[1]` asserted *"no code generator is wired"* on a run that emitted 46 items | read the receipt's provenance block against its own claim |
| 4 | My own first `non_claims` patch referenced undefined names — invisible to `py_compile`/`ast.parse` | the commit gate runs the code path, not a compile |

## Environment constraint (why this ran on CPU)

`transformers` imports and generates on this host (`5.14.1`, `torch 2.13.0+cpu`,
`cuda=False`); an earlier "DLL-blocked" note referred to a different interpreter. So the
candidate arm is CPU-bound at ~35-45 s/item for 48 items. Production runs belong on the
Vast CUDA target.

## Reproduction

```bash
export HENRI_SCICODE_OFFICIAL_TARGETS=1
export HENRI_SCICODE_CANDIDATE_SOURCE=backbone
python experiments/verification/s1_scicode_scaffold_runner.py --window 48
```

The flag is **default OFF**, so the committed 16-item default path is byte-identical
(verified: 50/50 predicate agreement, verbatim test text).

## Next falsification

Swap the backbone for HENRI's own wave→text egress on the same window, instrument, and
pin. A result at or below **5/48** places the egress no better than a 1.5 B backbone on
this channel. Prerequisite: the `fractional_shift` `(P3,P4)` pair is withdrawn as
unsound (N=480: 238/480 = 0.4958, exact p = 0.59; a same-construction control passes it),
so `phasor_bind` is the only arm with a defensible content claim.
