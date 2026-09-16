# M1 Part 2 — Independent Equivalence Checker: the clause is SATISFIABLE but VACUOUS

**Date:** 2026-09-16
**Script:** `HENRI V2/experiments/verification/m1_equivalence_checker.py`
**Receipt:** `m1_equivalence_receipt.json`
**Command:**
```
cd 'C:/Users/chan/henri-worktrees/aaii-v43/HENRI V2' && \
env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME \
    PYTHONPATH='C:/Users/chan/henri-worktrees/aaii-v43/HENRI V2' \
    PYTHONDONTWRITEBYTECODE=1 C:/Python314/python.exe \
    experiments/verification/m1_equivalence_checker.py
```
RC=0. CPU only, deterministic seed 20260916.

## The clause being tested

> "…**and an independent equivalence checker must agree above chance.**"

The M1 sequence so far: Part 1 (distinct top-1 count) was executed and **FAILED** —
the random-wave control out-scored the treatment (71/120, 84/120 vs 28/120, 14/120),
so the metric measured dispersion, not semantics (`1024e74`). Part 1′ (IPWR) **PASSED
on `phasor_bind`** — in-prompt word recovery 0.9833 vs control 0.0333 (`06b885b`).
This file supplies the second clause.

## Result

| arm | E1 agreement | E2 wave bal. acc | E3 κ | **E5 byte bal. acc** | **wave − byte** | verdict |
|---|---|---|---|---|---|---|
| **`phasor_bind`** | 0.9575 | 0.8273 (tpr 0.988 / tnr 0.667) | 0.6898 | **0.8273 (tpr 0.988 / tnr 0.667)** | **+0.0000** | `PASS_LITERAL_ONLY_WAVE_DOES_NOT_EXCEED_BYTE_BASELINE` |
| `fractional_shift` | 0.6321 | 0.5804 (tpr 0.644 / tnr 0.517) | 0.2117 | 0.8273 | **−0.2469** | `FAIL:E1,E3` |

Calibrated thresholds, `phasor_bind`: τ_wave **0.8828** vs τ_byte **0.8846** — also
nearly identical, because the wave is a deterministic function of the bytes.

## The finding

**The wave's balanced accuracy is bit-identical to the byte-Hamming baseline** —
0.8273 vs 0.8273, with identical tpr and tnr. The holographic tokenizer adds
**exactly zero** discriminative information beyond raw byte similarity on this task.

`wave − byte = +0.0000` is not "slightly less than"; it is the same number. The
mechanism is direct: the tokenizer is an **untrained byte+position transducer**, so
its only input is bytes and every byte substitution is a content change to it.
Where semantic equivalence coincides with a small length-preserving byte change, the
wave sees it for the same reason a byte comparator does — and nowhere else.

**Therefore the audit's clause has two readings, and they disagree:**

- **Reading 1 (literal — "agree above chance", chance = 0.50):** `phasor_bind`
  **PASSES** (E1 0.9575, E2 0.8273 > 0.50).
- **Reading 2 (controlled — must add signal beyond byte similarity):** **FAILS**
  by exactly 0.0000.

Reporting Reading 1 alone would be **the same vacuity class this session keeps
catching**: a gate that can be passed by something with no relevant capability — here
by a byte comparator. The clause is satisfiable, and satisfying it proves nothing.

## Why this does not contradict Part 1′

Part 1′ (IPWR 0.9833) used a **156-word manifest the codebook was built from**, so
recovering a manifest word is not purely byte matching. Part 2 compares
*arbitrary string pairs* with no shared manifest, which is where the byte transducer
runs out. Both results are correct and they are measuring different things: **the
codebook supplies discrimination; the encoder does not.**

## Consequence — the design requirement this pins down

A representation that must judge semantic equivalence needs an **equivalence source
that is not the codec**. Two candidates exist and neither is built:

1. **Test-time compilation from demonstration pairs** `(X_i, Y_i)` — the path the
   zero-pretraining invariant already specifies. Equivalence is *compiled online*
   from examples rather than assumed of the encoder.
2. **A learned equivalence head** — explicitly out of scope under the
   MINIMAL-TRAINING mandate without approval (M5 territory).

This is a **constructive** result: it identifies precisely which component the
zero-pretraining invariant requires, and shows the encoder cannot substitute for it.

## Third independent confirmation of the `position_binding` decision

| instrument | `fractional_shift` | `phasor_bind` |
|---|---|---|
| order sensitivity (`1024e74`) | 0.5083 | **0.9917** |
| IPWR content recovery (`06b885b`) | 0.1583 | **0.9833** |
| equivalence agreement (this file) | 0.6321 | **0.9575** |
| vs byte baseline (this file) | **−0.2469** | **+0.0000** |

Three independent instruments agree. `fractional_shift` is *worse than a byte
comparator* (−0.2469); `phasor_bind` matches it exactly (+0.0000). Recommend
promoting `phasor_bind` as the tokenizer default.

## Disclosure — FOUR void constructions, all mine, each with its measured signature

This clause took four attempts. The first three are **void**, not results:

| # | construction | defect | measured signature |
|---|---|---|---|
| A1 | `pos = s.replace(" ","  ")+"   "` | changed string **length** → all `j/L` angles move, later absolute positions displaced; the *positive* arm carried the larger perturbation | `E1 = 0.0000` exactly, κ = −1.0000 |
| A2 | `pos = space→tab` | length-preserving but **byte-distance unmatched**: pos had 5 scattered byte subs, neg ~4 contiguous → the "equivalent" arm held the larger byte distance | pos cosine 0.8380 **<** neg 0.8883; byte bal. = 0.5000 |
| A3 | synonym table of 2 pairs | positive arm **unconstructible** to N | 2 positives, needed 100 |
| A4 | byte-matched synonyms (616 constructible) | **UNFAIR CONTROL**: wave used a *calibrated* threshold, byte baseline used a *fixed* 0.5 → everything scored "equivalent" | byte tpr 1.000, tnr 0.000 → bal. 0.5000 |

A4 is the instructive one: **the comparator was calibrated and the baseline was
not**, so the baseline was crippled by my threshold choice. Fixing the baseline to
use the *identical* rule is what produced the real answer — and the answer is that
the wave and the baseline are the same measurement.

**A same-origin test whose arms are not matched on the superficial distance the
representation could exploit measures the confound, not the model.** Every void
attempt above violated that rule in a different way.

## Evidence class

`OBSERVED` for every number (parent-run, this commit). Controls: a byte-Hamming
baseline **calibrated with the identical rule**, plus a declared synonym table with
an antonym deliberately included and excluded (`large`/`small`) to prove the filter
does work. `DERIVED` for the transducer bound. **INSTRUMENT — not a capability
claim, not a benchmark score.** No training, no corpus read, no network, no model
inference, CPU only (`cuda_available = false`), deterministic seeds.

## Boundary

**M1 remains open.** Part 1 retired (metric unsound); Part 1′ passes on
`phasor_bind` (content recovery, codebook-mediated); Part 2 satisfies the clause
literally but adds nothing beyond bytes, so the *independent equivalence* requirement
is **not met in substance**. The honest status is:

> M1 — `BLOCKED_SEMANTIC_CAPACITY`, partially unblocked: egress recovers content via
> the tokenizer-derived codebook (IPWR 0.9833), but the encoder carries no
> equivalence signal beyond byte similarity, and the compensating mechanism
> (test-time compilation from demonstration pairs) is not built.

A2 `RETRAIN_REQUIRED`. M5 `REQUIRES_APPROVAL`. M6 `BLOCKED` (Vast credit 0).
NotebookLM `BLOCKED_AUTH_EXPIRED` — no corpus citation in this work.
**0 of 10 AAII v4.3 members scored: `NOT_EVALUATED`.** `main` untouched.
