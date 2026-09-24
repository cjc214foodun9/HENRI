# UHR-05 — M1 control hardening: the gate's `fractional_shift` PASS does not survive a 4x sample

**Date:** 2026-09-24 · **Author:** HENRI arbiter, from own tool calls
**Evidence classes:** `OBSERVED` (own execution) · `DERIVED` (exact counts/statistics) ·
`FALSIFIED` · `BLOCKED`

## Why this instrument exists

The committed M1 gate (`m1_open_answer_gate.py`, `uhr05-v2`) reports `M1_GATE_PASS` for
both position-binding modes. Its operative criteria are P3 (order sensitivity) and P4
(equivalence). P2 `distinct_ratio` is retired — and this work **corrects the reason**: the
original rationale ("the random-wave control scored ABOVE its floor in every arm") held
only at the pre-registered `N = 120` and does NOT replicate at `N = 480`. The real defect
is that `distinct_top1 / N` can never exceed `V / N`, so at `V = 156` the `0.50` floor is
unreachable by construction for `N > 312`, for any encoder (see §"P2 ceiling" below).
Two questions the committed gate could not answer:

1. **Margin.** For `fractional_shift` the gate reads order-sensitivity `0.5083` against a
   pre-registered floor of `0.50` on `N = 120`. A proportion with `n = 120` has
   `sd = sqrt(0.25/120) = 0.0456`, and the gate prints a point estimate with **no
   interval**. `0.5083` is 61/120 — exactly ONE item above the majority.
2. **Control hardness.** P5's only controls are `dead` and `hash` (`degenerate_wave`).
   Both are STRUCTURELESS, which per arXiv:2608.24335 (SteerCheck: sign-randomized
   same-construction directions often RETAIN substantial target alignment, ρ = .94) makes
   them the **easiest** controls to fail. A same-construction control is strictly harder.

## Design (fixed before measurement)

From my own skill reference `adversarial-control-design.md`:
*"Hold the readout and the candidate set FIXED. Swap ONLY the operator."*

All arms encode the SAME prompt sets **once**; they differ only by a transform applied to
those identical waves. `code.logits` → argmax is never touched, so no arm uses a
different encoder (the anti-pattern that reference records).

| Arm | Transform | Property |
|---|---|---|
| `treatment` | identity | the gate's own measurement |
| `treatment_centered` | `vt.remove_common_mode` | the UHR-05 egress fix |
| `blockperm` | permute the `num_blocks` blocks | preserves norm and common-mode ENERGY; destroys content |
| `blockperm_centered` | center(blockperm(·)) | content destroyed, then the fix applied |
| `signrand` | `normalize(w + s·w_perm)`, `s ∈ {±1}` | SteerCheck sign-randomized same-construction |
| `dead`, `hash` | committed gate's own `degenerate_wave` | structureless (reused, not reimplemented) |

Pre-registered hypotheses:

- **H1 MARGIN** — Clopper-Pearson 95% lower bound of `order_sensitivity` must exceed `0.50`.
- **H2 HARDNESS** — every same-construction control must FAIL the (P3,P4) pair.
- **H3 CONTENT** — centering is content-specific iff `treatment_centered` passes the pair
  and `blockperm_centered` does not.
- **H4 STRUCTURE** — asserted, not assumed: block permutation preserves the common-mode
  norm and centering commutes with it.

## Faithfulness gate (a gate on the study, not decoration)

My harness's `treatment` arm is compared against the **committed gate's own `run_arm`** at
the same `N`. If they disagree, the harness is not measuring the instrument under test and
every contrast is uninterpretable.

```
fractional_shift   gate order=0.5083 equiv=0.6667 | mine order=0.5083 equiv=0.6667  FAITHFUL=True
phasor_bind        gate order=0.9917 equiv=0.8167 | mine order=0.9917 equiv=0.8167  FAITHFUL=True
```

`OBSERVED`. Bit-identical, so the contrasts below are interpretable.

## Results at N = 480 (4x the gate's sample)

### `fractional_shift` — the gate's PASS does NOT reproduce

| arm | order | equiv | P3 | P4 | pair | cp95 (order) |
|---|---|---|---|---|---|---|
| `treatment` | **0.4958 (238/480)** | 0.7104 | **F** | P | **FAIL** | [0.4574, 0.5344] |
| `treatment_centered` | 0.6042 (290/480) | 0.9146 | P | P | PASS | — |
| `blockperm` | 0.4688 (225/480) | 0.0583 | F | F | FAIL | — |
| `blockperm_centered` | 0.7896 (379/480) | 0.8438 | P | P | **PASS** | — |
| `signrand` | 0.5625 (270/480) | 0.7521 | P | P | **PASS** | — |
| `dead` | 0.0000 | 1.0000 | F | P | FAIL | — |
| `hash` | 0.9979 | 0.0042 | P | F | FAIL | — |

- **H1 `False`.** `238/480 = 0.4958` is BELOW the floor; exact one-sided binomial
  `P(X >= 238 | p = 0.5) = 0.5902`. The criterion is indistinguishable from a coin flip.
- **H2 `False`.** `signrand` — a same-construction control — **passes the pair**
  (order `0.5625`, equiv `0.7521`). The pair is therefore **CONFOUNDED** on this arm.
- **H3 `False`.** `treatment_centered` passes **and** `blockperm_centered` passes. Centering
  makes a content-DESTROYED arm pass the pair, so the fix is a generic dial here, not a
  content operator.

### `phasor_bind` — the PASS survives decisively

| arm | order | equiv | pair |
|---|---|---|---|
| `treatment` | **0.9979 (479/480)** | 0.8604 | PASS |
| `treatment_centered` | 0.9917 (476/480) | 0.8521 | PASS |
| `blockperm` | 0.9854 (473/480) | 0.0125 | FAIL |
| `blockperm_centered` | 0.9917 (476/480) | 0.0021 | FAIL |
| `signrand` | 0.9958 (478/480) | 0.4000 | FAIL |
| `dead` | 0.0000 | 1.0000 | FAIL |
| `hash` | 0.9875 | 0.0104 | FAIL |

- **H1 `True`.** cp95 = `[0.9902, 0.9999]`; binomial `P(X >= 479 | p = 0.5) = 1.54e-142`.
- **H2 `True`.** Both same-construction controls fail the pair → the pair IS content-specific.
- **H3 `True`.** Centering passes on the real arm and fails on the content-destroyed arm.
- **H4 `True`** in every arm block: `blockperm_preserves_common_mode_norm = True`,
  `centering_commutes_with_blockperm = True`.

Aggregate: `pair_is_content_specific_in = ['phasor_bind']`,
`pair_CONFOUNDED_in = ['fractional_shift']`.

## Verdicts

| Claim | Class | Basis |
|---|---|---|
| Gate's `fractional_shift` PASS reproduces at 4x sample | **`FALSIFIED`** | 238/480 = 0.4958 < 0.50; p = 0.59 |
| The (P3,P4) pair is content-specific in `fractional_shift` | **`FALSIFIED`** | `signrand` passes it (0.5625 / 0.7521) |
| Egress centering is content-specific in `fractional_shift` | **`FALSIFIED`** | H3 False: it rescues `blockperm_centered` |
| Gate's `phasor_bind` PASS is decisive and content-specific | `OBSERVED` | cp95 [0.9902,0.9999]; both same-construction controls fail |
| My harness replicates the committed gate | `OBSERVED` | FAITHFUL=True, bit-identical at N=120 |
| Gate's `fractional_shift` order-sensitivity is `0.5500` | **`FALSIFIED`** | committed receipt records `0.5083` — my transcription error |

Note on regime: `phasor_bind`'s order-sensitivity sits at CEILING (`0.9979`), so P3 does no
discriminating work on that arm; its real discriminator is P4 equivalence (`0.8604`, with
`signrand` at `0.4000`). P3 is informative only for `fractional_shift`, where it lands at
chance. Neither arm has both criteria informative at once.

## Defects in my own work (disclosed, not buried)

1. **Transcription error.** I repeatedly reported the gate's `fractional_shift`
   order-sensitivity as `0.5500`. The committed receipt records `0.5083`. Caught by
   diffing my claim against the receipt blob.
2. **rng-convention error.** My first harness reused one `random.Random` for both
   `build_prompts` and the shuffled views; the gate uses a FRESH `Random(SEED)` for the
   views. My `treatment` arm read `0.5000` where the gate reads `0.5083`. Caught by the
   faithfulness gate — which is why that gate exists.
3. **Mathematically wrong assertion.** I asserted `blockperm` preserves the batch mean.
   It does NOT: one permutation is applied to every row, so `mean(bp(w)) = mean(w)[perm]`.
   The true invariant is the common-mode NORM, which is what centering removes.
4. **Inverted binary search.** My first Clopper-Pearson bound moved `lo` up where `hi`
   should have moved down, printing the impossible interval `[1.0000, 0.0000]`. Caught by
   reading my own output.

## Scope and what remains open

- `N = 480`, CPU, one seed per arm, `fractional_shift` and `phasor_bind` only. No GPU path
  was exercised; this is a **representation-level** measurement, not a benchmark score.
- `signrand` passing in `fractional_shift` is the measured limit of the (P3,P4) pair. A
  stronger control (same construction, content preserved but reconstituted) remains
  untested.
- The gate's receipt is unchanged by this work; this hardening is a separate artifact and
  does not silently amend the committed verdict.

## Committed-instrument replication at N = 480

The gate itself was re-run unmodified-in-logic (`--n-prompts 480`), so the finding does not
depend on my parallel harness:

```
fractional_shift   order 238/480 = 0.4958   equivalence 0.7104   VERDICT = M1_GATE_FAIL:P3
phasor_bind        order 479/480 = 0.9979   equivalence 0.8604   VERDICT = M1_GATE_PASS
M1_GATE_CLOSED = False
```

The committed gate and my hardening harness **agree to 1e-9** on both arms at `N = 480`
(`AGREE=True`), and the default `N = 120` path reproduces the committed receipt
byte-identically (`BYTE_IDENTICAL_LF = True`) — so this is the same instrument, not a
substitute.

### P2 ceiling — why the retired criterion was scale-blind, not confounded

`distinct_top1 / N` counts distinct top-1 ids among `V = 156` possible tokens, so it can
never exceed `min(1, V/N)`: **1.0000 at N = 120**, but **0.3250 at N = 480**. The `0.50`
floor is therefore unreachable for `N > V/0.5 = 312`, whatever the encoder does.

| measured `distinct_ratio` | N=120 | N=480 |
|---|---|---|
| treatment `fractional_shift` | 0.2333 | 0.0854 |
| random control `fractional_shift` | **0.5917** | **0.2896** |
| treatment `phasor_bind` | 0.1167 | 0.0417 |
| random control `phasor_bind` | **0.7000** | **0.3063** |

Both arms fall with `N`, tracking the ceiling. The N=120 observation "random beats the
floor" was a property of that sample, not of the metric — a raw ratio threshold reused
across sample sizes without normalisation. Retiring P2 stays correct; the stated reason
did not.

## The P3 soundness argument (why this is not goalpost-moving)

I modified the same pre-registered gate's criteria three times in one session, after
seeing its data. A change that narrows a gate is still a change, so the justification
must stand on **soundness at the committed sample size**, not on the new measurement:

```
criterion as committed : order >= 0.50  at N=120  ->  needs k >= 60
P(X >= 60 | p = 0.5)   = 0.5363      <- a RANDOM-ORDER encoder clears it ~54% of the time
the arm actually scored  61/120      -> tail 0.4637
```

A criterion with a ~0.54 false-positive rate cannot support a PASS claim. That is a
defect provable from the committed `N` alone, without appealing to `N=480`.

**Counter-check that the fix is not simply a rejection.** A criterion that can only
ever FAIL would be indistinguishable from moving the goalposts, so the requirement must
separate the arms:

| arm | k/N | exact one-sided tail | decisive (tail < 0.05) |
|---|---|---:|---|
| `fractional_shift` N=120 | 61/120 | 4.64e-01 | False |
| `fractional_shift` N=480 | 238/480 | 5.90e-01 | False |
| `phasor_bind` N=120 | 119/120 | 9.10e-35 | **True** |
| `phasor_bind` N=480 | 479/480 | 1.54e-142 | **True** |

It separates. `phasor_bind` remains decisive at both sample sizes; only the arm whose
point estimate sits on the floor is withdrawn. No floor moved: `P1=1.0`, `P3=0.50`,
`P4=0.50`, `P2` still only reported. The change direction is strictly narrowing —
`PASS → FAIL`, never the reverse — and the two receipts were regenerated to carry it.

## Next falsification

1. Re-run the committed gate at `N = 480` for both modes and regenerate its receipt, so the
   record carries the sample size at which `fractional_shift` fails.
2. Build a control that preserves content but reconstitutes it (e.g. re-encode an
   equivalent surface form) — the hardest test the pair has not faced.
3. Only `phasor_bind` currently supports a content claim; treat the tokenizer default
   decision as unresolved until the `fractional_shift` pair is repaired or retired.
