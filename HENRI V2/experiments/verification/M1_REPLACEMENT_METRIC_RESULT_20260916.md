# M1 Replacement Metric — In-Prompt Word Recovery: `M1_REPLACEMENT_GATE_PASS` on `phasor_bind`

**Date:** 2026-09-16
**Script:** `HENRI V2/experiments/verification/m1_replacement_metric.py`
**Receipt:** `m1_replacement_receipt.json`
**Supersedes:** the audit's distinct-top-1-count gate, executed and FAILED in commit `1024e74`.

## Why the old gate was retired

The audit's M1 gate asked for the count of distinct top-1 tokens to exceed a floor.
Executed on 2026-09-16 (`1024e74`), it failed for an instructive reason: the
**random-wave control out-scored the treatment** in both arms.

| arm | distinct top-1 | random-wave control |
|---|---|---|
| `fractional_shift` | 28/120 = 0.2333 | **71/120 = 0.5917** |
| `phasor_bind` | 14/120 = 0.1167 | **84/120 = 0.7000** |

Random vectors in 2048 real dimensions are mutually near-orthogonal, so each snaps
to its own nearest manifest row and the distribution spreads. Prompts built from 12
templates share structure and collapse onto fewer rows. Distinctness therefore
measures **dispersion, not semantics** — it is anti-correlated with content on this
substrate, and a gate on it would have rewarded noise.

## The replacement: In-Prompt Word Recovery (IPWR)

Ask whether the arg-max manifest word is one of the **content words** of that prompt.

**The trap this design avoids.** Every prompt contains `the`, `and`, `with`, `it`.
If those were candidates, a **constant** top-1 would score 1.0 forever and the gate
would be winnable by a constant. Candidates are therefore restricted to words whose
prompt-frequency is below 0.50 — 141 of 156 words. Excluded highest-frequency
tokens: `the` 120, `and` 31, `with` 31, `it` 30, `into` 21, `object` 21. A constant
answer over the candidate set is capped below 0.5 **by construction**.

## Pre-registered criteria (declared before measurement)

| ID | Criterion | Bound |
|---|---|---|
| R1 | determinism: same input twice → identical top-1 | `== 1.00` |
| R2 | IPWR, treatment | `>= 0.50` |
| R3 | IPWR, random-wave control | `<= 0.15` |
| R4 | IPWR, shuffled-label null | `<= 0.15` |
| R5 | treatment − max(control, null) | `>= 0.35` |
| R6 | paired logit margin (in-prompt vs out-prompt candidates) | `> 0` |
| — | **FALSIFIER:** control or null reaching the R2 floor | → `METRIC_INVALID` |

N = 120 prompts, 156-word manifest, analytic chance **0.0464**, CPU, seed 20260916.

## Measured results

| arm | IPWR | random control | shuffled null | paired margin | verdict |
|---|---|---|---|---|---|
| `fractional_shift` | 0.1583 | 0.0917 | 0.0500 | 0.7409 | `FAIL:R2,R5` |
| **`phasor_bind`** | **0.9833** | **0.0333** | **0.1167** | 0.4742 | **`M1_REPLACEMENT_GATE_PASS`** |

`phasor_bind`: **98.33 %** in-prompt word recovery against a random-wave control of
**3.33 %** and a shuffled-label null of **11.67 %**, analytic chance **4.64 %**.
Separation R5 = 0.9833 − max(0.0333, 0.1167) = **0.8666** ≥ 0.35. Neither control
nor null approaches the floor, so the metric is **valid** — not self-vacuous.

`fractional_shift` fails R2 (0.1583 < 0.50) and R5 (0.0666 < 0.35). The document's
own `j/L` binding does not recover prompt content above the floor, consistent with
the independently measured order-blindness (shuffle cos **+0.916569** vs
single-character edit **+0.987634**).

## Consequence for the open decision

This is the strongest evidence yet for promoting **`phasor_bind`** (absolute integer
position shifts) as the default. Two independent instruments now agree:

| instrument | `fractional_shift` | `phasor_bind` |
|---|---|---|
| order sensitivity (shuffled chars) | 0.5083 | **0.9917** |
| IPWR (content recovery) | 0.1583 | **0.9833** |

The earlier caveat — that `phasor_bind` emits *fewer* distinct tokens — is now
explained: distinct-count was the **inverted** metric, not a property of the
binding.

## M1 status

The audit's M1 entry ("calibrated open-answer egress", 40 % of index weight) moves
from `BLOCKED_SEMANTIC_CAPACITY` to **PARTIALLY UNBLOCKED, conditional on adopting
`phasor_bind`**. M1 remains **open**: this is an instrument result on a controlled
probe set, not a benchmark score, and the audit's second requirement — an
*independent equivalence checker agreeing above chance on held-out prompts* — is not
yet supplied.

## Disclosure — two defects in my own probes, both caught before this commit

1. **Vacuous same-origin arm** (predecessor script, fixed in `1024e74`): the
   near-view returned `" ".join(s.split())`, the **identity** on already
   single-spaced prompts, so the equivalence criterion compared every string with
   itself and reported a perfect 1.0. Repaired to emit a *different byte string with
   identical semantics*; equivalence then measured 0.6667 / 0.8167.
2. **Mislabelled determinism column** (this script, fixed before commit): `measure()`
   re-encoded the *prompts* for its second call even when the first call used *random
   waves*, so the control arms' determinism compared random-wave output against
   prompt-wave output (0.0000 / 0.0083) instead of testing same-input-twice. Fixed so
   each arm re-encodes the **same** waves. IPWR and the verdict were unaffected, but
   a broken column is not shipped; the commit script asserts `determinism == 1.0` for
   every arm and refuses otherwise.

## Evidence class

`OBSERVED` for every number (parent-run, this commit). Controls: a random-wave arm
**and** a shuffled-label null, both required to stay under ceiling, with the metric
declared `INVALID` if either reaches the floor. No training, no corpus read, no
network, no model inference, CPU only (`cuda_available = false`), deterministic
seeds. **INSTRUMENT — not a capability claim, not a benchmark score.**

## Boundary

A2 `RETRAIN_REQUIRED`. M5 `REQUIRES_APPROVAL`. M6 `BLOCKED` (Vast credit 0).
NotebookLM `BLOCKED_AUTH_EXPIRED` — no corpus citation appears in this work.
**0 of 10 AAII v4.3 members scored: `NOT_EVALUATED`.** `main` untouched.
