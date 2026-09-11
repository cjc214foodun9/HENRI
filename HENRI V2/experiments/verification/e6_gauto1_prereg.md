# G-AUTO-1 — Structural-Entropy Gate on Continuous Wave State (sealed prereg)

**Spec:** HENRI-SPEC-2026-09-12-E6-GAUTO1-STRUCTURAL-ENTROPY
**Carrier:** `carrier/e6-physical-verifier` @ `c14ac05`
**Status:** SPEC ONLY — `NOT_RUNNABLE` this session. No remote run. No code.
**Relationship to D3:** G-AUTO-1 is the acceptance gate for the autopoietic energy
carrier (D3b). D3b's kernel is ABSENT (receipt `e6_decision3_open.json` sha
`395fdeaf5944cac9`), so this gate cannot execute. This document exists to fix the
definitions BEFORE the carrier is authored, because the carrier's design depends on
which quantity it must move.

---

## 1. Why the gate was blocked, and what unblocks it

The paper defines its structural metric **on discrete byte tapes**. HENRI Zone A
carries continuous complex wave state on 𝕊^{D−1}, D = 65,536. Comparing a metric on
one to a state on the other is not a measurement; it is a category error. This was
recorded as the blocker in `e6_corpus_params.json` (sha `5d631823d486ad1e`):

> `blocking_bridge`: "paper defines it on DISCRETE BYTE TAPES; HENRI Zone A carries
> continuous wave states. A canonical discrete projection of the agent state is
> REQUIRED and is a new design decision with its own carrier."

**The projection is already in the project's own ingress spec.** `Project HENRI
Architecture.md` (sha `10ad480e0b62ab78`) states that text is transduced by a qFHRR
character-engram phase codebook with **quantisation mod K = 256**:

    z_d = exp(j·2π·q_d/K),   q_d ∈ Z_256,   K = 256

That quantisation **is** a canonical discrete projection. It is not invented here;
it is the project's own ingress operator, already instantiated in the tree as
`PhaseRingCodebookDecoder(k_bins=256)` and `HENRINeuralEgressUnbinder`. Using the
ingress quantiser as the projection keeps the metric commensurable with the state
that actually exists, and avoids inventing a bespoke readout whose only purpose
would be to pass a gate.

**Provenance split (recorded, not blurred):** the *formalism* is the paper's
(`H_shannon − K/n`, LZ-approximated Kolmogorov complexity, verbatim-cited in
`e6_corpus_params.json`). The *projection* is the architecture document's qFHRR
codebook. The paper does **not** state that projection; this prereg introduces it as
an engineering bridge and labels it as such.

---

## 2. The metric, fully specified

### 2.1 Projection (the bridge)

    Π : C^D → Z_256^N

1. Apply the qFHRR phase quantiser to the wave state per component:
   `q_i = floor((clamp(x_i, -1, 1) + 1)/2 · (K − 1))`, `K = 256`.
   This is exactly `PhaseRingCodebookDecoder.quantize_phase_ring`.
2. Emit one symbol per quantised component of the **measured** subspace.
   `N` = number of components carried into the metric (see §2.3 for the window).

**Falsifiable requirement:** Π must be deterministic and state-injective enough that
two materially different wave states do not quantise to the same symbol stream.
Check: `|Π(a) − Π(b)|` must separate genuine state changes from the quantisation
floor. If the quantiser collapses the state (too few distinct symbols), the metric is
measuring the quantiser, not the agent. **This check is a precondition of the gate,
not part of it.**

### 2.2 The two terms

    G(seq) = H_shannon(seq) − K_LZ(seq)/n

- `H_shannon(seq)`: Shannon entropy of the **empirical symbol distribution** over the
  window, in bits. Estimator: plug-in (maximum-likelihood) over the observed symbol
  histogram. Plug-in entropy is **biased downward** at small n; the minimum-n gate in
  §2.3 exists to bound that bias.
- `K_LZ(seq)`: LZ compressed size **in bits** of the quantised symbol stream, produced
  by a pinned compressor. `n` = sequence length in **symbols** (not bytes).
  The compressor, its version, and its settings are part of the specification; an
  unpinned compressor makes the metric irreproducible.
- Both terms require `n` and the unit of `K_LZ/n` stated in **bits per symbol**.

**Why this pair is not gameable by a constant stream.** For an i.i.d. stream the
metric converges to 0 as n grows (paper-verbatim). For a stream of concatenated copies
of one shorter string, `H_shannon` is small but `K_LZ/n` is far smaller, so `G` is
**substantially non-zero**. Therefore:
- a **null / frozen** state scores ≈ 0 (both terms collapse),
- a **repetitive-but-structured** state scores > 0,
- a **high-entropy noise** state scores ≈ 0 (H large, K/n ≈ 1 bit/symbol, difference
  near zero).
A dead system and a noise generator both score ≈ 0. That is the property that makes
the metric meaningful for a solipsism gate.

### 2.3 Windows, stride, and the minimum-n gate

- Window length `W` and stride `S` must be fixed before execution and reported.
- **Minimum-n gate:** `n ≥ n_min` where `n_min` is the smallest n at which the LZ
  estimator's bias is bounded. LZ over-estimates `K` at small n (the dictionary
  warm-up region), which inflates `K_LZ/n` and drives `G` **negative**. The prereg
  requires a **bias curve**: run the estimator on synthetic streams of known
  complexity across a range of n, and set `n_min` at the knee. If the curve is not
  measured, `n_min` is a guess and the gate is not runnable.

**Rejection rule:** if `G < 0` at any window, that window is **not** passed to the
verdict. A negative `G` means the estimator, not the agent, is being measured.

---

## 3. Limit-cycle detection (the solipsism condition)

The directive's gate is: *"break internal solipsistic limit cycles (Kuramoto r → high,
zero external reward) within N epochs, driving anisotropic Langevin variance to melt
non-productive attractors."* As written, `N` and the "variance threshold" have no
units. This prereg fixes them.

- **Observable:** Kuramoto order parameter `r(t)`, plus the egress-entropy `G(t)`.
- **Solipsism signature (must be detected, not assumed):** `r(t) → r_high` while
  `G(t) → 0` and external reward is identically zero. Both conditions are required:
  high phase-locking alone is not solipsism (a correct system may be phase-locked),
  and low entropy alone is not solipsism (a dead system is also low-entropy).
- **Break criterion:** the epoch `N*` at which `G(t)` exits the near-zero band while
  external reward is still zero. `N` is bounded at **512**, the paper's
  maximum step count (verifiable operand, not a guess).
- **Estimator for "variance melting":** the anisotropic Langevin variance is reported
  as the trace of the injected covariance restricted to the **misaligned orthants**,
  with the orthant basis stated. A raw scalar variance without a basis is not a
  measurement.
- **Baseline:** `r(t)` and `G(t)` measured with the thermostat **disabled**. Without a
  paired baseline, an observed break cannot be attributed to the thermostat.

---

## 4. Pre-registered acceptance and rejection criteria

| ID | Criterion | Pass condition | Failure meaning |
|---|---|---|---|
| **P0** | Projection separates states | ≥ 2 distinct symbols per window on real run data | metric measures the quantiser |
| **P1** | Estimator bias bounded | `n ≥ n_min` from a measured bias curve | metric sign is an artefact |
| **P2** | Solipsism detected | `r → r_high` AND `G → 0` AND reward ≡ 0 in some window | gate has no trigger; vacuous |
| **P3** | Break inside budget | `N* ≤ 512` with thermostat ON | carrier fails G-AUTO-1 |
| **P4** | Attribution | `N*_thermostat < N*_baseline` (paired) | break not attributable to the thermostat |

**Rejection rule (pre-registered):** if P2 fails — if the system never enters the
solipsistic signature — then **the gate is not passed and not failed; it is
VACUOUS**, and must be re-specified. A gate that cannot trigger is the defect already
found once in this project (`Sagnacfunctor.txt` metric C, range 3e-05). It must not be
repeated.

---

## 5. Resource limits and kill criteria

- Cost ceiling: one GPU-exclusive session on `vast-5090`; no training, evaluation only.
- **Cheapest kill experiment:** run P0 and P1 **alone** — projection separation and the
  LZ bias curve — on synthetic streams of known complexity. Both are CPU-only, take
  seconds, and each can independently invalidate the whole metric. Kill the metric
  before spending GPU time on the carrier.
- Stop condition: if P0 fails, do not proceed. If P1 yields no usable `n_min`, the
  metric is unusable and the gate must be re-specified rather than relaxed.

---

## 6. Explicit blockers (do not read this prereg as progress)

| Item | Status | Evidence |
|---|---|---|
| Gate execution | **BLOCKED** | `_fused_autopoietic_kuramoto_kernel` ABSENT from 363 files |
| Carrier this gate accepts | **NOT AUTHORED** | D3b is `BLOCKED_MISSING_PREMISE` |
| `n_min` | **NOT MEASURED** | bias curve is a prerequisite, not yet run |
| Compressor identity | **UNPINNED** | must be fixed before any number is reported |
| Projection validity (P0) | **NOT MEASURED** | CPU-only, seconds, not yet run |

**No E6 remote run is admissible.** This document fixes definitions; it produces no
result. Any later claim that G-AUTO-1 "passed" requires P0–P4 with the artefacts
named above.

---

## 7. Evidence labels

- Paper formalism (`H_shannon − K/n`, LZ approximation, i.i.d. limit): `OBSERVED`
  (verbatim in `e6_corpus_params.json`, sha `5d631823d486ad1e`).
- qFHRR K=256 quantisation as the projection: `DERIVED` (architecture doc sha
  `10ad480e0b62ab78`; instantiated in-tree as `k_bins=256`).
- Paper's max step count 512 as the N bound: `OBSERVED` (same receipt).
- Solipsism signature definition, `n_min` requirement, P0–P4: `HYPOTHESIS` — proposed
  here, not yet tested.
- Any gate outcome: `NOT_EVALUATED`.
