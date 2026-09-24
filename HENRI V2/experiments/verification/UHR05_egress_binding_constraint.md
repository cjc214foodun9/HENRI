# UHR-05 — The Egress Binding Constraint: measured, diagnosed, and repaired (flag-gated)

**Date:** 2026-09-24 · **Author:** HENRI arbiter, from own tool calls
**Evidence classes:** `OBSERVED` (own execution) / `DERIVED` (shown arithmetic) / `HYPOTHESIS`
**Scope of the claim:** ONE representation defect in the text-egress channel. Not a score.

---

## 1. The benchmark structure that bounds any answer (OBSERVED, pinned audit)

AAII Intelligence Index **v4.3** is a **10-member** composition. GPQA Diamond, MMLU-Pro,
MMMU-Pro, τ²/τ³-Banking, Terminal-Bench 2.1/Hard, MATH-500, AIME 2025 and LiveCodeBench are
**legacy — present in methodology, NOT in v4.3**. Planning against the legacy list targets a
suite the operator does not score.

| Access | Weight | Members |
|---|---:|---|
| **Locally gradeable** | **25%** | SciCode (10%), Terminal-Bench 4.0 (10%), AutomationBench-AA (5%) |
| **Externally graded / private** | **75%** | AA-Briefcase, GDPval-AA v2, AA-Omniscience, GDP.pdf, AA-LCR v1.1, HLE, CritPt |

By response channel: **tools 40% · open-answer 40% · code execution 20%**.

**Consequence (`DERIVED`):** three quarters of the index cannot be measured offline. Two fifths of
the remaining weight is the open-answer channel, and one fifth needs a code-execution harness.
Any non-zero score plan must name which channel it repairs.

## 2. HENRI's recorded external-outcome ledger (OBSERVED, prior artifacts)

| Benchmark | Recorded outcome | Class |
|---|---|---|
| MBPP | **0/452** | `OBSERVED` negative |
| ARC-AGI-3 | `EVALUATION_BLOCKED` by design | `BLOCKED` |
| AAII v4.3 (all 10 members) | `NOT_EVALUATED` | `BLOCKED` |
| SciCode scaffold | dataset absent at the runner's pinned path | `BLOCKED_INFRASTRUCTURE` |
| Egress discrimination gate (M1) | falsifier FIRED — see §3 | `OBSERVED` negative |

## 3. The measured defect (my own execution, this session)

Recomputed from the committed `m1_open_answer_gate_receipt.json`:

> **READ THIS BEFORE QUOTING THE TABLE BELOW — 2026-09-24 (UHR-05).** The receipt has
> since been regenerated from a strictly narrower instrument, so the `distinct top-1`
> figures below remain valid as history but the receipt's **verdicts changed**:
> `fractional_shift` is now `M1_GATE_FAIL:P3_NOT_DECISIVE` (the P2 rationale given here
> was `FALSIFIED` at `N=480`; P3 as written admitted a coin flip), while `phasor_bind`
> remains `M1_GATE_PASS`. The `VACUOUS_DISTINCT_COUNT_NOT_INFORMATIVE` inference below
> still holds — the armed falsifier did fire — but the ROOT CAUSE is a scale defect in
> the statistic, not the control winning the axis. See
> `UHR05_m1_control_hardening.md` for the measured replacement.

| arm | distinct top-1 (floor 0.50) | RANDOM control |
|---|---:|---:|
| `fractional_shift` | **0.233** (below floor) | **0.592** (above floor) |
| `phasor_bind` | **0.117** (below floor) | **0.700** (above floor) |

The treatment scored **below** the floor while the random control scored **above** it. The gate's
own falsifier fired: `VACUOUS_DISTINCT_COUNT_NOT_INFORMATIVE`. The inference is stronger than
"the statistic is uninformative": on this substrate top-1 distinctness does not track content.

**Diagnosis (`OBSERVED`, own probe, 40 varied prompts):**

| arm | mean \|cos(w, mean_w)\| | shared energy |
|---|---:|---:|
| `fractional_shift` | **0.8421** | ~71% of every prompt's wave lies in **ONE shared direction** |
| `phasor_bind` | 0.1038 | ~1% |

So the text-egress collapse is a **common-mode (DC) defect** in one position-binding arm, not an
absence of content. Removing the query's batch mean raised `fractional_shift` argmax uniqueness
17/40 → 30/40.

## 4. The repair, and its pre-registered verdict

One bounded, flag-gated change in `henri_vla_tokenizer.py`:

- `remove_common_mode(wave)` — pure batch-mean subtraction. **Not** renormalized: per-row
  rescaling reintroduces a nonzero batch mean (measured residual 1.25e-3 vs ~1e-8). Scale
  invariance verified — pure subtraction and renormalized subtraction give the **same argmax**,
  because `_embed` normalizes downstream and dividing by a positive scalar cannot change it.
- `logits(wave, *, center=None)` — resolution order explicit-argument → `HENRI_EGRESS_CENTER`
  (default `"0"`) → OFF. **The default path is byte-identical.**

Pre-registered paired test (criteria fixed before the numbers), with a random control that must
not be boosted:

| arm | verdict | equivalence | order sensitivity | random-control delta |
|---|---|---:|---:|---:|
| `fractional_shift` | **ACCEPT** | 0.425 → **0.775** | 0.550 → **0.750** | +0.04999 |
| `phasor_bind` | **REJECT** | 0.075 → 0.100 | 1.000 → 0.975 | +0.00000 |

`phasor_bind` is a documented REJECT: it has no shared component to remove, so centering is inert
there. A test asserts that stays true.

**Honest weakness (stated, not hidden):** on the `fractional_shift` fixture the random control
gains **+0.04999** in top-1 diversity — sitting exactly on the pre-registered +0.05 bound. So
centering is **not** a pure content operator on this substrate; it also marginally increases
readout diversity. The licensed claim is therefore narrow: *centering repairs one measured
shared-component defect and improves semantic invariance on that arm.* It is **not** established
as a general fix, and it ships **default OFF**.

## 5. Two defects in my own work, found by running it

1. **Implementation contract defect** — the first `remove_common_mode` subtracted the mean and
   renormalized per row, which reintroduces the mean, making the stated contract false. Caught by
   asserting on the batch-mean **norm** (a cosine is 0/0 when the mean is exactly zero and would
   have passed vacuously).
2. **Self-confirming fixture defect** — my first test built 40 prompts differing by one integer,
   planting the property under test in the data: `phasor_bind` read 0.687 instead of 0.087.
   Rewritten with varied prompts.

## 6. What this does NOT claim

- **No score.** A score needs the gradeable subset to be executed and graded; the local SciCode
  dataset is absent at the runner's pinned path, so the honest status is
  `BLOCKED_INFRASTRUCTURE`, not a number.
- **Not the 75%.** Agentic and privately graded members stay blocked.
- **Not a production wiring of A2.** The Zone C causal DAG's only consumers are the ledger smoke,
  the wiring control script, and the contract tests; it has **no production caller**. Wiring the
  contingency gate into production requires first making the DAG a production component — a
  larger architectural change, recorded here rather than faked.
- **A3 stays OFF.** The observational readout is deliberately fail-open and **never sets
  `is_pruned` and never admits a candidate**; it tracks `best_readout_node` separately as a
  diagnostic. It changes no returned program.

## 7. Structural insight for project HENRI

The wave core is not the binding constraint. **The egress channel is.** Every externally scored
member requires emitting something a checker can parse — text (open-answer 40%), tool calls
(40%), or code (20%). HENRI's measured failures are all on that boundary: MBPP 0/452, the M1
distinctness inversion, and the common-mode collapse. Ingress and the physical core are
separately instrumented and pass their gates.

Therefore the correct ordering of work is **egress-first**: one channel at a time, each with a
content-vs-random-control paired test, before any benchmark submission. Selection statistics must
be validated against a random control *before* they are used to rank anything — the retired
`top1_token_unique` gate is preserved as a negative control for exactly this reason.

## 8. Next falsification (pre-registered)

1. Stage the SciCode subset at the runner's pinned path and run the ≤16-item scaffold: does any
   item pass against the published checker? If zero, the coding channel is
   `FALSIFIED_AT_SCAFFOLD`, not "in progress".
2. Run the content selector's margin on real items **against the random control**. If the margin
   does not separate them, the open-answer channel is `BLOCKED_ON_SELECTION` and no threshold
   change is permitted in response.
3. Re-derive the `+0.04999` random-control gain on a second, independent prompt fixture. If it
   exceeds +0.05, centering must be reclassified as a diversity dial and withdrawn.
