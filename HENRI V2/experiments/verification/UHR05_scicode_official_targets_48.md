# UHR-05 DECISIVE: SciCode now runs on the OFFICIAL grader at 48 items (5/48)

**Date:** 2026-09-24 · **Author:** HENRI arbiter, from own tool calls
**Run:** `s1-scicode-scaffold__16d573ccf1c3__20260924T215943Z-59e7c591`
**Evidence classes:** `OBSERVED` (own receipt) · `DERIVED` (arithmetic on it) · `BLOCKED`

## What changed

For every prior SciCode run the denominator was **2**. `target` -- the value each
published test asserts -- is assigned **zero times** in both corpus splits (dev: 219
uses / 0 assignments; test: 1172 / 0), so N items were excluded as
`BLOCKED_EXTERNAL_CONSTANT`. That exclusion was CORRECT: the scaffold's own header
forbids substituting a local tolerance or a re-derived expected value.

It was also one step away from becoming permanent. `target` is supplied by the
benchmark's **own grader**, from its **own** data file:

```
grader   = github.com/scicode-bench/SciCode   (clone rc=0, `pip install -e .`, `import scicode` OK)
accessor = src/scicode/parse/parse.py:126  process_hdf5_to_tuple(step_id, test_num, h5py_file)
data     = eval/data/test_data.h5   1,049,345,865 B   338 groups   (h5py only; `datasets` unused)
```

Using that accessor over that file is not the substitution the header forbids. The
mapping is proven, not assumed: corpus `test_cases` counts vs h5 test counts agree on
**50/50** dev sub-steps with the identical distribution (`{3: 33, 4: 17}`), which is
what justifies `test_cases[i] -> targets[i]`.

## The result (`OBSERVED`)

```
verdict              SCORED          reason "controls passed and the pin matched"
score                numerator 5 / denominator 48   pass@1 = 0.104167   window_size = 48
passed_item_ids      SciCode-29-29.1, -29-29.2, -38-38.2, -47-47.1, -78-78.1
failure_taxonomy     candidate_arm: FAILED_ASSERT 16, ERROR_OTHER 25, BLOCKED_DEPENDENCY 2, STATUS_PASSED 5
                     reference_arm: STATUS_PASSED 48        <-- 48/48, SciCode's own code vs its own tests
official_targets     items_considered 96, items_targeted 96, injections 98, refusals 0, enabled true
dataset_pin          all_match True   revision 4510f6a6aa27c43fad7b43da2c59602a86e88480
accounting.valid     True             execution_error_count 0        ledger_rows_written 48
wall_seconds         2535.56
```

**The reference control is the instrument validation, and it is now 48/48** -- up from
2/2. SciCode's own `ground_truth_code` passes its own published tests on every single
item through this harness, with the official `target` injected. Without that, a 0 from
the candidate arm would be unattributable: it could be the harness, the sandbox, or the
target plumbing rather than the generator.

## Candidate provenance (`OBSERVED`)

```
name        BackboneCandidateSource      produces_code true
model_id    Qwen/Qwen2.5-1.5B-Instruct   revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
device      cpu                          max_new_tokens 1024     deterministic true
calls 159   cache_hits 111   generations 48   emitted 46   truncated 1   no_code_emitted 2
defines_expected 46
```

The candidate class changed **meaning**, not just count. Previously every item died at
the `target` wall and was excluded. Now the code **executes** and fails on its own merit:
`FAILED_ASSERT` (16) means the model's function ran and its output disagreed with the
published assertion; `CANDIDATE_UNDEFINED_SYMBOL` appears only 3 times. That is
attribution instead of exclusion.

## Non-claims (do not weaken these)

1. **This is NOT a HENRI capability claim.** The candidate arm is a **local pretrained
   backbone** (Qwen2.5-1.5B) plus this harness. HENRI's own wave->text egress is not in
   the loop. `pass@1 = 0.104167` measures that backbone.
2. **No AAII claim.** Nothing about the Artificial Analysis index, its weights, or any
   constituent follows from this number.
3. **Not namespace-isolated.** `mode=container-rlimit`, `isolated=False`,
   `surrogate=True`; the downgrade is recorded in the receipt.
4. **The reference arm is excluded** from the candidate numerator and denominator; it
   exists to validate the instrument.
5. **n = 48 is a window**, not the benchmark: the dev split holds 15 problems / 50
   sub-steps, and 96 sub-steps exist in the corpus overall.

## Disclosed defect in the committed receipt

The receipt's `non_claims[1]` still reads *"No code generator is wired to SciCode; the
only registered candidate source emits the empty string. Any pass@1 here is the floor
for an ABSENT generator."* That clause became **FALSE** the moment the flag-gated
`BackboneCandidateSource` existed -- this very run emitted 46 items. It is the
absent-generator text, emitted because the clause was hard-coded.

The source is **now fixed** to derive the clause from the run's own `candidate_source`
block, so future receipts cannot carry a claim that contradicts their own provenance.
**This receipt is NOT retro-edited.** Editing a stored receipt to match a later
understanding would destroy its value as evidence, so it is committed as-is with this
disclosure, and a re-run at the same window supersedes it.

## Cheapest next falsification

**Swap the backbone for HENRI's own wave->text egress** and re-run the same 48-item
window. Same instrument, same pin, same controls; only the candidate arm changes. A
result at or below 5/48 would place the egress no better than a 1.5B backbone baseline
on this channel -- a governance win that redirects effort. Note the prerequisite: the
`fractional_shift` `(P3,P4)` pair is withdrawn as unsound (N=480: 238/480 = 0.4958,
p = 0.59; a same-construction control PASSES it), so `phasor_bind` is the only arm with
a defensible content claim.

## Reproduce

```bash
export PY="C:/Users/chan/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe"
cd "C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2"
HENRI_SCICODE_OFFICIAL_TARGETS=1 HENRI_SCICODE_CANDIDATE_SOURCE=backbone \
HENRI_SCICODE_WINDOW=48 "$PY" experiments/verification/s1_scicode_scaffold_runner.py
```

Requires the staged grader root (`.../Temp/scicode_official`, 1.05 GB h5) and the
Qwen2.5-1.5B snapshot; the pin verifies both on every run.
