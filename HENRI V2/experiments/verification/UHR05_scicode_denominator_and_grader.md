# UHR-05 — Why SciCode pass@1 *had* a denominator of 2, and what legitimately widens it

> **ANSWERED AND SUPERSEDED IN PART — 2026-09-24, commit `d33ad10`.**
> The question posed by this title is now settled; the measurement below is the historical
> record of the **pre-grader** state and is left byte-intact.
>
> The blocker was real and correctly handled: `target` is assigned **zero** times in both
> corpus splits, so excluding those items was correct protocol (`target` is the value every
> published test asserts). But `target` was never *unavailable* — the benchmark supplies it
> at grade time through its **own** accessor over its **own** data file. Using those is not
> the local tolerance / re-derived expected value / grader injection that `runner:18-22`
> forbids.
>
> MEASURED with my own calls:
> * official grader obtained — repo clone `rc=0`, `pip install -e .` → `import scicode` OK
> * `test_data.h5` **1,049,345,865 B**, 338 groups; h5 ∩ dev **50/50** (test 288/291)
> * `process_hdf5_to_tuple` resolves **50/50** dev targets; mapping control — corpus
>   `test_cases` counts equal h5 test counts on **50/50** sub-steps, zero disagreement
> * `test_num` must be the **actual** per-item count `{3:33, 4:17}`; a uniform `4` raises
>   `KeyError` on the 33 three-test items (this is why a uniform 4 first read as 17/50)
> * flag `HENRI_SCICODE_OFFICIAL_TARGETS` (default **OFF**) — satisfiable pool **2 → 50**,
>   attemptable **2 → 16** in a 16-item window, reference arm **2/2 → 16/16 STATUS_PASSED**
>
> Therefore the "denominator of 2" in the tables below is a **pre-grader artifact, not a
> property of the task**. Resolution and full safety argument: the `d33ad10` commit message.

**Date:** 2026-09-24 · **Author:** HENRI arbiter, from own tool calls
**Evidence classes:** `OBSERVED` · `DERIVED` · `FALSIFIED` · `BLOCKED`

## The measurement under audit

Newest SciCode scaffold run (`s1-scicode-scaffold__16d573ccf1c3__20260924T195937Z-82fe785b`,
chosen by **mtime**, not name-sort):

```
verdict SCORED · candidate_source BackboneCandidateSource (produces_code=true)
score: numerator 1 / denominator 2 = pass@1 0.5
denominator_definition: "sub_steps whose OWN SciCode reference solution passed its
                        OWN published tests through this harness inside the window"
```

Window = 16 items. Per-arm taxonomy:

| arm | STATUS_PASSED | ERROR_OTHER | BLOCKED_EXTERNAL_CONSTANT | BLOCKED_DEPENDENCY |
|---|---:|---:|---:|---:|
| candidate | 1 | 5 | **9** | 1 |
| reference | 2 | 0 | **13** | 1 |

So the denominator is 2 because the **reference arm** cannot execute on 13 of 16 items.

## Why: `target` is assigned nowhere in the shipped corpus

The classifier maps a `NameError` to `BLOCKED_EXTERNAL_CONSTANT` when the missing name is in
`EXTERNAL_CONSTANT_NAMES = ("target",)` (`s1_scicode_scaffold_runner.py:530-533`). Where does
`target` come from?

| corpus file | rows | rows using `target` | total uses | **assignments** |
|---|---:|---:|---:|---:|
| `problems_dev.jsonl` | 15 | 15 | 219 | **0** |
| `problems_test.jsonl` | 65 | 65 | 1172 | **0** |

It appears only in `test_cases` and `general_tests` — and is **assigned in neither split**.
Examples (raw strings, `\n` shown escaped):

```
assert np.allclose(normalize(v), target)
assert np.allclose(cross(a,b), target)
assert np.allclose(GS(A, b, eps, x_true, x0), target)
```

`target` is the **expected return value** of the step's own function. So it must be supplied
by the grader, not by the dataset.

## The harness is correct, and this is NOT an eighth harness defect

Every defect found earlier this session was an instrument that could not return success
(relocated import, unsatisfiable predicate, stub `numpy`, off-by-one, unsatisfiable G2). This
one is different, and the distinction is load-bearing. The runner declares the decision in its
own header (`:18-22`):

> 3. NOT a substitute grader. The published tests are used verbatim. Where an item's
> published test references a constant that the dev split does not ship (`target`), the item
> is BLOCKED_EXTERNAL_CONSTANT and is excluded from the score numerator and denominator.
> No local tolerance, no re-derived expected value, **no local grader injection**.

Binding `target` from the reference arm would reduce every published test to
`candidate == reference` — that is grader reimplementation, and it would convert a real
external dependency into a self-confirming loop. `FALSIFIED` as a fix path.

The harness *can* read the field (`READABLE_FIELDS` at `:147` includes `general_tests`), so
the exclusion is a deliberate protocol choice, not an access limitation.

## The legitimate unblock path

| candidate source | result |
|---|---|
| PyPI `scicode` / `scicode-eval` / `sci-code` | **HTTP 404** — not published under those names |
| vendored in the repo tree | none (only the corpus, its pin, and unrelated `evaluate_*` scripts) |
| hermes venv | `scicode` not importable |

The official mechanism, read from the project's own README:

```bash
git clone git@github.com:scicode-bench/SciCode.git
cd SciCode && pip install -e .          # the `scicode` package (not on PyPI)
# then download the numeric test results -> eval/data/test_data.h5
```

`test_data.h5` is a **separate Google Drive download**, and it is the source of the `target`
values. Installing the package alone does not supply them.

## Consequence for the score — the honest framing

```
dev corpus: 15 problems, 50 sub_steps   |   window used: 16 items
SciCode leaderboard (official, main problem / subproblem resolve rate):
  o3-mini-low 10.8 / 33.3 ·  o1-preview 7.7 / 28.5 ·  DeepSeek-R1 4.6 / 28.5 ·  GPT-4o 1.5 / 25.0
```

`pass@1 = 1/2 = 0.5` is **not comparable** to any leaderboard figure. With `n = 2` the 95%
interval spans most of `[0, 1]`, and the denominator is a property of the harness's reference
arm, not of HENRI. Labelled in its own receipt `BACKBONE_BASELINE_NOT_HENRI_CAPABILITY`.

## Verdicts

| Claim | Class | Basis |
|---|---|---|
| `target` is a dev-split gap | **`FALSIFIED`** | assigned 0 times in dev **and** test |
| The 9 exclusions are a harness defect | **`FALSIFIED`** | declared protocol `:18-22`; the harness reads the field |
| The numerator/denominator are arithmetically correct | `OBSERVED` | 1 passed of 2 reference-verified items |
| Re-deriving `target` locally is a valid unblock | **`FALSIFIED`** | collapses the test to `candidate == reference` |
| Widening the denominator needs the official grader | `BLOCKED` | PyPI 404; needs repo clone + `test_data.h5` download |
| `pass@1 = 0.5` is a benchmark number | **`FALSIFIED`** | `n = 2`; denominator is a harness property |

## Owner defect disclosed

My first pass at the `target` question used `str()` on `test_cases`, which is a **list of
strings**. `str()` produced a repr with escaped newlines, so my line-anchored `re.M` search
for `target = ` could never match inside it — a **false negative on exactly the field under
test**. Corrected by walking the raw strings; the assignment count is 0 in both splits.

## Next falsification

1. Install the official `scicode` package (repo clone + `pip install -e .`) on the Vast CUDA
   target, fetch `test_data.h5`, and re-run the scaffold against the official grader. Only
   then does a SciCode number become citable.
2. Until then, report SciCode as `BLOCKED` (grader absent) rather than as a low score.
3. A full 50-sub-step dev sweep is an honest **coverage** increase, but it does not remove the
   `target` dependency and so cannot by itself produce a benchmark number.
