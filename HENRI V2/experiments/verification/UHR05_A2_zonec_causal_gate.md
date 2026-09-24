# UHR-05 A2 — The Pearl intervention gate: validated, then given a production consumer

**Date:** 2026-09-24 · **Author:** HENRI arbiter, from own tool calls
**Evidence classes:** `OBSERVED` (own execution) · `DERIVED` · `HYPOTHESIS` · `FALSIFIED` · `BLOCKED`

---

## 1. The amendment, and what "wire it" actually required

Amendment 2: *ratify a causal link in Zone C if and only if the external environment
registers irreversible change (`ΔS_ext != 0`).*

The gate for this **already existed and was already correct**:
`henri_causal_contingency.ratify_causal_link`, whose own docstring reads
*"This is the function a Zone C writer should call."*

What was missing was a consumer. Measured module-level consumer map over the whole tree:

| Location | Imports of the gate |
|---|---:|
| `tests/contract/test_uhr04_amendments.py` | 4 |
| `experiments/verification/uhr05_a2_ledger_wiring.py` | 1 |
| `zone_c_causal_engram_dag.py` | 0 (1 incidental ref) |
| **`production_arc_run.py`** | **0** |

Production contained **zero** references to `henri_causal_contingency`, `PearlGate`,
`forge_edge`, or `ZoneCCausalEngramDAG`. So every Zone C engram write on the
production path was ungated. That gap is what this change closes.

---

## 2. Two premises I had to test before trusting the gate

### 2a. Is `ΔS_ext` read from a constant? (the mock question)

The record carried a note: *"skip = EXTERNAL_OUTCOME_EFE DEAD FLAG, 0 writers."* If
that were true, the gate would ratify on a self-computed quantity — the
self-confirming-detector defect class.

**Measured — the note is wrong, on both halves:**

```
production_arc_run.py:177  EXTERNAL_OUTCOME_EFE = os.environ.get("EXTERNAL_OUTCOME_EFE","0") == "1"   <- HAS a writer
                      :780  external_outcome_efe=EXTERNAL_OUTCOME_EFE          <- forwarded
                     :1063  if EXTERNAL_OUTCOME_EFE: orch.planner.reset_external_outcomes()
                     :2923  if EXTERNAL_OUTCOME_EFE:  ... frame-change comparison
```

And the gate does not read a constant. When `delta_s_ext` is not supplied it
**computes** it from real frames:

```
henri_causal_contingency.py:586-591
    if delta_s_ext is None:
        delta_s_ext, sig = _delta_and_signature(before, after)
_delta_and_signature:632-658
    mag = float(diff.mean()); sig = frozenset of differing flat indices
```

Own execution on real grid pairs:

| input | status | ΔS_ext |
|---|---|---|
| `[[0,0,0],[0,0,0]]` → `[[1,0,0],[0,2,0]]` | `RATIFIED` | 0.333333 |
| `before` → `before` (unchanged) | `REFUSED_STATIC` | 0 |

**Verdict:** the flag is live and the gate is externally grounded. `OBSERVED`.

### 2b. Can the gate RATIFY, or does it only ever veto?

A gate that only vetoes is not a gate. The record had proven only refusals
(ft09 → `CONFOUNDED`, static → `SOLIPSISM`). A ratify control **already existed**:
`test_c11b_gate_ratifies_the_reactive_environment` (`14 passed` for `-k "c11 or c12 or c14"`).

My first three-way discrimination attempt **failed on my own fixture**, not on the gate:
I gave each action one constant signature, which drives `mi_step` to 0 and makes the
permutation null degenerate. Same defect class as the `phasor_bind` fixture earlier.
The gate was fine; my fixture was not. Recorded, not buried.

---

## 3. A real defect found by running the wiring: the signature FORM

My first bridge used `frozenset(changed_indices)` as the change signature — matching
`henri_causal_contingency._delta_and_signature`. That form is right for the
**per-observation** `pearl_intervention_gate`, which only asks *"did the world move?"*
It is **wrong** for the **episode-level contrast** this bridge feeds.

Measured, twice:

```
R1 step = unique loop counter            -> n_step_matched_pairs = 0  -> REFUSED_INSUFFICIENT
R2 step keys pooled, INDEX-SET signature -> REFUSED_CONFOUNDED: mi_action_given_step = 1.0 == null_q99 = 1.0
```

The validated control documents the mechanism in its own comment
(`tests/contract/test_uhr04_amendments.py:71-75`): an index-set signature is
*"near-unique per record, which makes the permutation null DEGENERATE and the
conditional-MI comparison vacuous — measured: observed == null q99 == 1.58408 exactly."*

The measurable, discriminating form is the **low-cardinality count** — the form the real
ka59 harvest records:

```python
sig = ("count", max(1, base + jitter))
```

After the fix, through the bridge's own record builder on real grid pairs:

| regime | status | signature cardinality | ΔS_ext | mi_action_given_step vs null_q99 |
|---|---|---|---:|---|
| reactive | **`RATIFIED`** | 9 | 0.2383 | 1.4414 > 1.3164 |
| step-driven cursor | `REFUSED_CONFOUNDED` | 1 | 0.25 | — |
| static world | `REFUSED_STATIC` | 1 | 0.0 | — |

Three-way discrimination holds. The count form still separates action-contingency
(different actions change different *numbers* of cells) from a step-driven cursor
(the band moves the same number every step).

---

## 4. The precondition that makes strict `enforce` unsatisfiable today

The contrast statistic is `I(change; action | step)`. It needs at least one step index
observed under **two or more** distinct actions.

Production's loop structure:

```
:1042      for env_name in env_ids:
:1641          for step in range(args.steps):
```

**One episode per environment.** Every step index therefore carries exactly one action,
`n_step_matched_pairs = 0`, and the verdict is `REFUSED_INSUFFICIENT` — so strict
`enforce` would refuse **every** Zone C write. Shipping that would repeat the session's
central defect: an instrument that cannot return success.

Resolution uses the gate's **own** distinction. `REFUSED_INSUFFICIENT` means *"the
contrast is UNCOMPUTABLE, not absent"* — so a single-pass run cannot claim to have
established either a pass or a refutation. The bridge adds four modes:

| Mode | Behaviour |
|---|---|
| `"0"` (default) | gate absent; `allow_zonec_write` → `True`, byte-identical default path |
| `advisory` | evaluate + emit telemetry; write always proceeds |
| `enforce` | refuse anything not `RATIFIED` — requires the replay precondition |
| `enforce_single_pass` | refuse only what a single-pass run can **establish** (`REFUSED_STATIC`, `REFUSED_CONFOUNDED`); allow `REFUSED_INSUFFICIENT` labelled `contrast_uncomputable` |

To satisfy strict `enforce`, replay the same environment with different action
sequences. **Pooling step indices across different environments would fabricate the
contrast** (the signature differs because the env differs), which is why the bridge
clears its buffer whenever the environment changes.

Fail-closed rules: an unknown mode string is treated as `"0"` for the write decision and
reported as `mode_invalid`, so a typo can neither silently enable nor disable
enforcement. In `enforce`, an evaluation defect refuses.

---

## 5. Where the gate is wired

Both Zone C write sites consult it:

- `:3580` scheduled checkpoint (`step % CHECKPOINT_EVERY == 0`)
- `:3649` episode-end consolidation marker (EDMD L3)

Producer: `observe_zonec_causal_gate(step, macro_actions[0], grid, post_frame)` at the
frame-change site (`:2918`), fed by the **environment's own return** — pre-action `grid`
and post-action `obs_next.frame[0]`. The planner's prediction is never used. An absent
post-frame is **not** recorded, because a missing observation must not count as "the
world moved".

A refusal emits `zonec_causal_gate_refused_write` to telemetry. It is never silent.

---

## 6. Pre-registration note for the A4 arm (not yet run)

A4 appends four `Translate(...)` ops:

```
d4_group_actions() -> 6 names          (IDENTICAL to the OFF literals)
ADDED by the flag  -> 4 Translate ops  ON = 13 ops vs OFF = 9 ops
REMOVED by the flag -> none
```

So an ON/OFF A/B measures **"13 actions with group structure" vs "9 actions"** — two
variables at once. To isolate the Lie-generator claim, hold the candidate set at 13 ops
in **both** arms and vary only the generation strategy (exhaustive search vs
`d4_group_actions()`). Also note `sagnac_mcts_planner.py:122` executes a translation as
`np.roll(np.roll(res, dx, axis=0), dy, axis=1)`; the generator bank **verifies** equality
to `np.roll` (8.9e-16) — it does not compute the operation. Calling the bank
"computational" would be proof by naming.

---

## 7. Claims and classes

| Claim | Class | Evidence |
|---|---|---|
| The gate computes ΔS_ext from real frames | `OBSERVED` | `:586-591`; `RATIFIED @ 0.333333` on real pairs |
| `EXTERNAL_OUTCOME_EFE` is live | `OBSERVED` | `:177` env writer → `:780` → `:1063`, `:2923` |
| The gate can ratify AND refuse | `OBSERVED` | three-way table, §3 |
| Index-set signatures make the null degenerate | `OBSERVED` | `REFUSED_CONFOUNDED`, mi == null_q99 == 1.0 |
| A2 had zero production consumers | `OBSERVED` | consumer map, §1 |
| Strict `enforce` is unsatisfiable single-pass | `OBSERVED` | production loop shape + `n_step_matched_pairs = 0` |
| The wiring is exercised on a real CUDA run | `BLOCKED` | not run; validation is CPU-level |
| Any AAII v4.3 score | `BLOCKED` | 75% of weight is externally graded |

**Not claimed:** that the gate improves any score. It gates writes; it does not raise
them. Its value is that an ungated Zone C write is no longer possible to make silently.
