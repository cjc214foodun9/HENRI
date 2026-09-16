# M1 Pre-Registered Gate — Result: `M1_GATE_FAIL`

**Date:** 2026-09-16
**Script:** `HENRI V2/experiments/verification/m1_open_answer_gate.py`
**Receipt:** `m1_open_answer_gate_receipt.json`
**Command:**
```
cd 'C:/Users/chan/henri-worktrees/aaii-v43/HENRI V2' && \
env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME \
    PYTHONPATH='C:/Users/chan/henri-worktrees/aaii-v43/HENRI V2' \
    PYTHONDONTWRITEBYTECODE=1 C:/Python314/python.exe \
    experiments/verification/m1_open_answer_gate.py
```
**Exit code:** 1 (`M1_GATE_CLOSED = False`). CPU only, deterministic seed 20260916.

---

## Why this gate had to be run

The audit's M1 entry stated:

> **Gate:** on a held-out probe set of N>=100 distinct prompts, the count of
> distinct top-1 tokens must exceed a pre-registered floor, and an independent
> equivalence checker must agree above chance.

That gate was **never executed**. The prior verdict `top1_token_unique = 1` was
already known to be vacuous: the random control scored 37/128 distinct against a
treatment of 39/128 — indistinguishable. So a distinct-token COUNT alone cannot be
the acceptance criterion. This run therefore pre-registers five criteria and
**builds in a falsifier**: if the random-wave arm also reaches the distinctness
floor, the gate is declared VACUOUS regardless of how the treatment looks.

## Pre-registered criteria (declared before measurement)

| ID | Criterion | Floor |
|---|---|---|
| P1 | determinism: same prompt twice -> identical top-1 | 1.00 |
| P2 | distinct top-1 / N | >= 0.50 |
| P3 | order sensitivity: shuffled chars (same multiset) -> different top-1 | >= 0.50 |
| P4 | equivalence: same-origin re-rendered view -> same top-1 | >= 0.50 |
| P5 | random waves through the SAME codebook must NOT reach the P2 floor | required |

N = 120 distinct prompts built from 12 templates over a 156-word manifest.

## Measured results

| arm | det. | distinct top-1 | P3 order | P4 equiv. | **random-wave control** | entropy / ln(V) | verdict |
|---|---|---|---|---|---|---|---|
| `fractional_shift` | 1.00 | 28/120 = **0.2333** | 0.5083 | 0.6667 | **71/120 = 0.5917** | 4.5861 / 5.0499 | `VACUOUS_DISTINCT_COUNT_NOT_INFORMATIVE` |
| `phasor_bind` | 1.00 | 14/120 = **0.1167** | **0.9917** | 0.8167 | **84/120 = 0.7000** | 4.7838 / 5.0499 | `VACUOUS_DISTINCT_COUNT_NOT_INFORMATIVE` |

**The random-wave control out-scores the treatment in both arms** (71 > 28 and
84 > 14). P5 fires. The gate FAILS, and it fails for the informative reason.

## Interpretation — the metric measures spread, not semantics

This is an `OBSERVED` negative result with a clear mechanism:

- **Random waves are mutually near-orthogonal** in 2048 real dimensions. Each
  snaps to its own nearest manifest row, so the top-1 distribution spreads widely
  (84/120 distinct).
- **Real prompts are not mutually near-orthogonal.** They are built from 12
  templates, so many share structure and collapse onto fewer, template-indicative
  rows (14/120 distinct).

Distinctness of the top-1 token is therefore **anti-correlated with content** on
this substrate. It does not measure semantic capacity; it measures dispersion.
A gate built on this statistic cannot distinguish a working egress from noise, and
in the measured case would have *rewarded* noise.

**Consequence for the audit's M1 line:** “calibrated open-answer egress” remains
unmet, and the specific acceptance statistic named in the audit is itself
unsound. Any future M1 gate must key on a metric that a near-orthogonal control
cannot win — for example target-token recovery against a shuffled-label control,
or mutual-information between prompt content and emitted token above a
label-permutation null. Distinct-count must not be used again.

## Decision evidence produced for `position_binding`

`phasor_bind` (absolute integer position shifts) vs the document's
`fractional_shift` (`j/L`), same prompts, same manifest, same codebook:

| quantity | `fractional_shift` | `phasor_bind` | direction |
|---|---|---|---|
| P3 order sensitivity | 0.5083 | **0.9917** | phasor_bind ~2x |
| P4 equivalence | 0.6667 | **0.8167** | phasor_bind higher |
| P2 distinct top-1 | 0.2333 | 0.1167 | — |

On the two criteria that measure *structure preservation*, `phasor_bind` is
stronger, consistent with the earlier order-blindness finding (shuffle
cos +0.916569 vs one-char edit +0.987634). It is **not** a clean win: it emits
fewer distinct tokens, consistent with the dispersion metric being the wrong
instrument rather than with the representation being worse. Recorded as measured,
not as a recommendation on its own.

## Disclosure — a vacuous arm in my own first version, caught and repaired

The first run reported `equivalence = 1.0` (perfect). That was **vacuous**: my
`near_view()` returned `" ".join(s.split())`, which is the **identity** on these
already-single-spaced prompts. P4 compared every string with itself.

Repaired to produce a genuinely different byte string with identical semantics
(`"  ".join` spacing + trailing spaces, asserted different AND asserted to
normalize back to the original). Post-repair P4 = 0.6667 / 0.8167 — a real
measurement that can fail. **A same-origin test whose two arms are the same bytes
tests nothing**, and it would have reported a perfect equivalence score for any
representation, including a broken one.

## Evidence class

`OBSERVED` for every number (parent-run, this commit). `DERIVED` for the
spread-vs-semantics interpretation. Control type: INSTRUMENT_CONTROL — this is a
measurement apparatus result, **not** a capability claim and **not** a benchmark
score. No training, no corpus read, no network, no model inference, CPU only
(`cuda_available = false`), deterministic seeds.

## Boundary

M1 remains **`BLOCKED_SEMANTIC_CAPACITY`** — now for a sharper reason than before:
the previously named acceptance statistic is unsound, and the replacement metric
has not yet been built. A2 `RETRAIN_REQUIRED`. M5 `REQUIRES_APPROVAL`. M6
`BLOCKED` (Vast credit 0). NotebookLM `BLOCKED_AUTH_EXPIRED` (no corpus citation
appears in this work). **0 of 10 AAII v4.3 members scored: `NOT_EVALUATED`.**
