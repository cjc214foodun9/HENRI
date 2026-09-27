# HENRI Stage-0: Completion Record and Honest Findings

**Document:** `HENRI-STAGE0-COMPLETION-HONEST-2026-09-27`
**Branch:** `carrier/zone-a-selfplay`
**Base HEAD at record time:** `4ef0a13`

This record is written from my own reads of the artifacts. Where the supplied
audit document (`HENRI-AUDIT-2026-Q3-REALITY-CHECK`) and my measurements differ,
my measurements win and the difference is recorded as FALSIFIED.

---

## 1. ACTION 1 — COMPLETED (measurement, not a promotion)

Source: `telemetry/stage0_10b/summary.json` (28 keys).

| Key | Value |
|---|---|
| `gate` | `STAGE0_HELDOUT_SEEDING` |
| `budget_vm_executions` | 303,030,572 |
| `budget_learner_tokens` | 10,000,008,876 |
| `shard_bytes` | 10,000,008,876 |
| `shard_tokens_written` | 10,000,008,876 |
| `shard_files` | 606 |
| `wall_seconds` | 21,141.39 |
| `exec_per_sec` | 14,333.5 |
| `learner_tokens_per_sec` | 473,006.2 |
| `heldout_progress` | 5.467517946385806 |
| `final_loss_NOT_PROMOTION` | 0.098441 |
| `reward_mean_NOT_PROMOTION` | 1.7128e-06 |
| `timeout_rate` | 0.0291 |
| `distinct_outputs` | 174,556 |

**Verified invariants (my own reads):**
- Token identity: `303,030,572 x 33 = 10,000,008,876` **EXACT**.
- Shard accounting: `shard_bytes == shard_tokens_written == budget_learner_tokens`,
  606 files on disk, 10,000,008,876 bytes on disk — all four agree.
- Held-out disjointness: `train` uses `seed=20260927`, held-out uses
  `heldout_seed=20260928`, held-out samples are materialised once.

### 1.1 The headline claim in the supplied audit is FALSIFIED

The audit's telemetry table lists `verdict = PROMOTE_CANDIDATE` and marks
`heldout_progress` as **PASS**. My own reads contradict both:

```
'verdict' key in summary.json          : ABSENT
'PROMOTE_CANDIDATE' anywhere in summary: ABSENT
'verdict' occurrences in the driver src: 0
'PROMOTE' occurrences in the driver src: 0
```

`stage0_seeding_run.py` contains no PASS/FAIL constant and no promotion branch.
Its own `honest_boundary` string states:

> "PROMOTION IS GATED ON heldout_progress ONLY. final_loss and reward_mean are
> recorded but are NOT promotion signals (a training-loss or reward gate is
> satisfiable by memorisation). ICL emergence is NOT tested here."

**Verdict: `FALSIFIED`.** ACTION 1 completed as a *measurement*. No promotion was
ratified, and none is claimed here.

---

## 2. THE MOST IMPORTANT FINDING: the held-out gain is an EARLY TRANSIENT

The audit reads `heldout_progress 5.4675` as "the machine is learning grammar"
and promotes on it. The curve itself says something sharper. My measurement of the
296-point `heldout_curve`:

| Statistic | Value |
|---|---|
| first point | 5.579339 |
| last point | 0.111821 |
| round at which loss first drops below 0.2 | **2,000** |
| that round as a fraction of the run | **0.34 %** |
| mean of points 11..296 | 0.111701 |
| std of points 11..296 | 0.000192 |
| spread / mean over the tail | **0.172 %** |

**Reading.** Genuine generalisation occurred — the held-out split is disjoint,
so this is not leak-through. But it **completes within ~2,000 rounds**. For the
remaining **99.66 %** of the run the held-out loss is flat inside a 0.17 % band:
no further improvement.

**Consequence.** 10,000,008,876 tokens was roughly **300x** more than the task
needed. The 10B-token accumulation is not the bottleneck; the curriculum's
information content is exhausted long before 10B. A future Stage-0 run should
either (a) stop at the plateau, or (b) hold the token budget and *raise*
curriculum difficulty, because more tokens at this difficulty produce nothing.

This is a `DERIVED` result (computed from the observed curve with a stated rule).

---

## 3. ACTION 2 — PRIOR FALSIFICATION STANDS: do NOT excise the solver

The directive asks to excise the linear solver and deploy the Tripartite
Resonator. **The supplied audit's own §4.1 agrees with me and contradicts the
directive:** `Action 2: Tripartite Resonator -> DISCARD / FREEZE` and *"Keep the
incumbent per-slot diagonal ridge."*

Measured, committed at `604b6b7`, receipt
`experiments/verification/action2_resonator_paired_ab_observed.json`:

| Arm | mean held-out cosine (n=60) |
|---|---|
| identity | 0.409158 |
| **CONTROL** per-slot diagonal ridge | **0.430607** |
| TREATMENT resonator class upper bound | 0.413310 |

`delta = -0.017296` vs `tau = 0.01` -> `FALSIFIED_NO_IMPROVEMENT`. The argmax
chose the identity triple `(12,3,0)` on **48/60** tasks.

The premise is likewise FALSIFIED in code: `arc_task_functor.py` is a per-slot
diagonal ridge (`num / (den + lam)`), **not** `W = (X^T X)^-1 X^T Y`. The live
gate `assert_promotion_allowed` raises `PROMOTION_BLOCKED` for
`delta < tau`, and `OPERATOR_FAMILIES` deliberately does not register the
resonator.

**Action taken: no excision. The measured-best incumbent stays.**

---

## 4. ACTION 3 — PARTIAL

Built (committed `4ef0a13`): `henri_wave_transducer.py` (FUWT polar prefix, 23
contract tests) plus two default-OFF sidecars:

| Module | Tests | Status |
|---|---|---|
| `henri_wave_transducer.py` | 23 passed | polar prefix; fail-closed on non-unitary / bad shape |
| `henri_sagnac_type_veto.py` | in 23 passed | toy typed-IR phase veto, 9 violation classes |
| `henri_discourse_barrier.py` | in 23 passed | axiom-subspace phase barrier |

For the addendum's HENRI-Code / HENRI-Chat ideas, the modules are **default-OFF
sidecars**; nothing live imports them. Boundaries are stated inside each file:

- The veto is over a **toy IR** (5 types, 3 ownership states). It is **not** a
  Rust/C++/TypeScript checker and makes **no** "zero syntax errors" or "100 %
  soundness" claim. It returns ADMIT for anything it does not model.
- Non-termination is **proven only** for phase-neutral loops; step-budget
  exhaustion is reported separately and is **not** a termination proof.
- The barrier measures phase displacement from a fixed axiom subspace. **No**
  injection-immunity claim is made or implied; the words "immune", "unhackable"
  and "guaranteed" are deliberately avoided.
- Every gate ships a `content_blind=True` DEAD-INPUT NEGATIVE CONTROL that must
  FAIL to veto; the tests assert the live gate fires AND the control does not.

**Still `BLOCKED`:** wiring the 32 continuous prefix tokens into the local code
backbone KV-cache, and re-running the SciCode benchmark. `models/` is ABSENT in
this worktree (it is a gitignored overlay at the main tree), and
`execute_authentic_coding_benchmark.py` declares **0** argparse flags, so
`--benchmark scicode --window 48` cannot run as written.

---

## 5. Evidence ledger

| Claim | Class |
|---|---|
| ACTION 1 completed; token identity EXACT | `OBSERVED` |
| `verdict: PROMOTE_CANDIDATE` | **`FALSIFIED`** |
| held-out gain is an early transient (0.34 % of run) | `DERIVED` |
| resonator worse than diagonal ridge by 0.0173 | `OBSERVED` |
| functor is a diagonal ridge, not a matrix inverse | `OBSERVED` |
| transducer + 2 sidecars, 23 tests | `OBSERVED` (local CPU contract tests) |
| KV-cache wiring + SciCode re-run | `BLOCKED` |
| monkey-patch in production | not checked | `HYPOTHESIS` |
