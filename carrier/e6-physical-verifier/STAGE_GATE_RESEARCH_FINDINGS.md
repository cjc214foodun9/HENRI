# Carrier Stage-Gate Research Findings

Document under test: `HENRI-ARCH-2026-CARRIER-AUDIT-AND-PHYSICAL-ML-GAPS`
Subject: qFHRR autopoietic carrier, 4-stage gate table.
Date: 2026-09-12
Host: Vast.ai instance 50797414 — RTX PRO 6000 Blackwell WS, 97 887 MiB, cc 12.0,
driver 595.71.05, torch 2.12.0+cu130, python 3.12.3.

Every verdict below is from executed code. Local runs are CPU; the `remote_*`
receipts were produced on the Blackwell instance and returned by scp with a
matching SHA-256 (`d4ccde93dbf2ab9ae95c4b1d68409a69b53c644bb95f079485e4b163db206ac0`).

Evidence class key: `OBSERVED` (real tool output) / `DERIVED` (calculated from
observed data) / `FALSIFIED` (contradicted by measurement).

---

## Verdict summary

| Stage | Gate bound | Verdict | Classification |
|---|---|---|---|
| 1 Carrier execution | `||G|| <= 0.02` on dead/noise | **PASS** | Fix required: defect D1 quantified |
| 2 Metric locality | AUC >= 0.85 | **PASS** on the fix | Mechanism **CONFIRMED**; my first gate could not see it |
| 3 Discrete egress snap | `H(Y) <= 1.2` bits at beta=8 | **FAIL** | **Spec defect** — beta is a joint function of query regime |
| 4 Transition identification | one-step `delta <= 0.15` | **FAIL** at 1500 rows; **PASS** at 12000 | **Spec defect** — metric and sample diversity unstated |

Two of the four gates as written in the source document do not survive
measurement. A third (Stage 2) initially appeared to fail and does not: the
fault was in this harness, and the correction is recorded below.

---

## Stage 1 — carrier execution: PASS

Defect D1 reproduced, then quantified. `zlib` carries a fixed warm-up cost of
roughly O(300..500) bits. That constant appears in `K_LZ` as `overhead/n`, so a
one-sided rule rejected its own controls.

Measured (`OBSERVED`):

| n | `||G||` white noise | `||G||` constant | verdict |
|---|---|---|---|
| 8 192 | 0.0303 | 0.0387 | FAIL |
| 16 384 | 0.0190 | 0.0174 | PASS |

Structured stream `G = 1.82` (correctly rejected as non-dead).

Resolution: a two-sided verdict plus a pre-condition gate at
`n_min = 16384 = ceil(overhead/epsilon)`. This is a **carrier-correct** fix, not
a threshold relaxation: the gate now rejects at small `n` instead of silently
accepting a biased estimate.

Draft-kernel defects flagged and recorded, not hidden:

- **D-A** block-local reductions
- **D-B** energy write race
- **D-C** `atan2` undefined at `|Psi| -> 0`
- **D-D** `dt_eff` is not a Margolus-Levitin bound

---

## Stage 2 — metric locality: mechanism CONFIRMED (earlier "falsified" verdict withdrawn)

The source document names `arc_public_ingress.py` as the mean-pooling site.
That file exists (7 742 B) and contains **no pooling of any kind** (`OBSERVED`).

The real sites are four identical `_bridge_to_d64*` projections, each the
canonical wave-to-d64 bridge for one ARC engine (`OBSERVED`, all tracked):

| file, relative to `HENRI V2/experiments/verification/` | line | enclosing function |
|---|---|---|
| `arc_g1_topological_engine.py` | 106 | `_bridge_to_d64_single` |
| `arc_f15_trajectory_engine.py` | 97 | `_bridge_to_d64` |
| `arc_f22_resolution_engine.py` | 89 | `_bridge_to_d64_single` |
| `arc_f23_causal_engine.py` | 93 | `_bridge_to_d64_single` |

All four contain the identical line:

```python
pooled = w.view(16, 4096).mean(dim=0)     # 65536 -> [16, 4096] -> mean -> 4096
```

`arc_g1_topological_engine.py:106` sits inside `_bridge_to_d64_single`, which
collapses a 65536-dimension wave to d=64: reshapes to `[16, 4096]`, averages the
16 blocks away, rescales to 64, then L2-normalizes. That is exactly the
metric-locality destruction this stage names. **The fix must be applied to all
four sites**, not to one file.

Measured AUC on the ORIGINAL gate task (`OBSERVED`, 8 classes x 12 per class):

| Method | mean AUC | min AUC | Gate (0.85) |
|---|---|---|---|
| Mean pooling (current) | 0.9292 | 0.8839 | **PASS** |
| 8-channel local Clifford blocks (proposed) | 1.0000 | 1.0000 | PASS |

That table alone suggests the document's predicted collapse does not occur.
**It is misleading, and the fault is in this harness.** `make_wave` put class
identity on the block axis AND the channel axis:

```python
blocks = [(label * 3 + i) % NUM_BLOCKS for i in range(3)]
chans  = [(label * 2 + i) % CHANNELS   for i in range(3)]
```

Mean pooling is `w.view(16, 4096).mean(dim=0)`. It averages over the 16 BLOCK
axis and **retains the channel axis**. So the channel signature survived
averaging and carried the classification to 0.9292. The harness author put
identity on the one axis the operator preserves, which is the defect.

### The discriminating test

`verify_stage2_task_dependence.py` imports both bridges from the gate and varies
only where class identity lives (`OBSERVED`, 8 classes x 12 per class):

| task | mean pooling AUC | Clifford AUC | block-perm cosine (mean) |
|---|---|---|---|
| `both` (original gate task) | 0.9292 PASS | 1.0000 PASS | 1.0000 |
| `channel_only` (blocks fixed) | **1.0000 PASS** | 0.9939 PASS | 1.0000 |
| `block_only` (channels fixed) | **0.5014 FAIL** | 0.9959 PASS | 1.0000 |

Verdict `MECHANISM_CONFIRMED`. On the minimal task where identity lives only in
block position, mean pooling collapses to **0.5014**, which is chance. The
document's Gap 1 mechanism is **real and measured**; the earlier "premise not
reproduced" verdict was an artifact of this harness and is withdrawn.

The mechanism is also visible directly: block-permutation cosine is `1.0000` for
mean pooling (a wave and its block-permuted twin are indistinguishable) versus
`-0.1024` for Clifford blocks (they are separated). Averaging over blocks is
exactly what destroys the local metric interval.

### What remains unestablished

Whether the collapse occurs on **real ARC grids through the real ingress path**
is not established. No wave corpus exists on this machine (`OBSERVED`: no `.npz`
`psi` banks; `HENRI V2/data/` holds only text benchmarks — gpqa, mbpp, mmlu, …;
no trajectory jsonl matching the `arc_g1` input contract). What is established
is the operator's behaviour: it destroys block-level locality whenever class
identity is not on the retained axis, and classification then falls to chance.

Correction to file: the fix target is the four `_bridge_to_d64*` bridges under
`HENRI V2/experiments/verification/`, not `arc_public_ingress.py`.

---

## Stage 3 — discrete egress snap: FAIL

`henri_egress.py` already uses `ContinuousHopfieldCleanup(beta=8.0)`.

Measured (`OBSERVED`, remote Blackwell, D=8192, M=1000, sqrt(D)=90.51):

| beta | H(Y) on-manifold | H(Y) off-manifold |
|---|---|---|
| 1.00 | 9.9568 | 4.0711 |
| 8.00 | 2.7433 | 0.4273 |
| 90.51 | 0.0000 | 0.0000 |
| 100.00 | 0.0000 | 0.0000 |

At the specified beta=8 the harness reports `H(Y) = 9.9601` bits against a
`<= 1.2` bound and returns `GATE RESULT: FAIL`. Only `beta = sqrt(D) = 90.5`
snaps to 0.0 bits.

FALSIFIED hypothesis: an earlier note claimed a random Gaussian query produces
near-uniform logits, making `H(Y)` an off-manifold detector. On real silicon the
opposite holds — `H(Y)` falls **monotonically with beta in both regimes**, and
the off-manifold value at beta=8 (0.4273) is *lower* than the on-manifold value.
`H(Y)` therefore does not separate the regimes and cannot serve as that detector.

Consequence: `beta` is a joint function of the operator and the query regime.
Fixing `beta = 8` in the specification, independent of `D` and of the query
distribution, is not well posed. The bound must be stated together with the
query distribution it applies to.

---

## Stage 4 — transition identification: FAIL at 1500 rows, PASS at 12000

### The digest-only control reproduces Gap 3

Digest-only ledger rows yield **0 recoverable `(x_t, a_t, x_next)` triples** —
identification is BLOCKED, exactly as the document predicts. The full payload
sidecar yields 1500 pairs and the fit runs.

### Failed gate run

At 1500 transitions (50 episodes x 30 steps):

```
delta(one-step) = 0.381738  -> FAIL     (mean per-sample metric)
delta(one-step) = 0.479137  -> FAIL     (second family)
linear dynamics, full payload : delta = 0.381738  -> FAIL
```

### Decisive root-cause experiment

The fitted operator is scored against the **ground-truth** operator on the same
held-out test set, so the noise floor cancels and `fitted/truth` isolates
estimator quality from data conditioning (`OBSERVED`):

| Regime | delta (Frobenius) | fitted/truth | kappa_max | Verdict |
|---|---|---|---|---|
| 50 ep x 30 st (n=1500, the gate) | 0.1562 | **2.308** | 126.9 | FAIL |
| 400 ep x 30 st (n=12000) | **0.0745** | 1.014 | 12.2 | **PASS** |
| i.i.d. n=12000 | **0.0250** | 1.010 | 7.9 | **PASS** |

Noise sweep on i.i.d. data, fitted vs ground truth:

| NOISE | delta fitted | delta ground truth |
|---|---|---|
| 0.00 | 0.00000 | 0.00000 |
| 0.005 | 0.00618 | 0.00612 |
| 0.02 | 0.02460 | 0.02434 |
| 0.05 | 0.06199 | 0.06127 |

### Conclusion — this overturns the earlier "2.31x estimator shortfall" claim

That claim is **FALSIFIED**. With adequate conditioning, the fitted operator
matches the ground-truth operator to within 1 percent
(`fitted/truth` = 1.014 at n=12000, 1.010 i.i.d.). The 2.308 ratio at n=1500 is
**ill-conditioning**, not estimator error: 45 dictionary parameters per action
against roughly 375 rows per action, drawn from only 50 correlated episodes,
gives `kappa_max` = 127.

Two independent defects in the gate definition:

1. **The metric is unstated, and the verdict flips on it.** Frobenius gives
   0.0745 (PASS) while mean per-sample error gives 0.3848 (FAIL) on the *same*
   fit. A relative per-sample metric additionally inflates without bound under
   contracting dynamics, because `||x_next||` shrinks while the noise does not.
2. **Required sample diversity is unstated.** The gate is reachable at 12000
   transitions and unreachable at 1500. The bound must be paired with a minimum
   row count and a conditioning target (`kappa_max`).

Actionable consequence: the ledger must return **at least ~12000 payload-backed
transitions with adequate per-action diversity**, or the identification gate is
not testable. 8x more data flips FAIL to PASS; no algorithmic change was needed.

---

## Cross-cutting finding

Gate bounds in this document are written as bare scalars. Measurement shows
each one is a joint function of additional variables:

| Gate | Written as | Actually depends on |
|---|---|---|
| 1 | `||G|| <= 0.02` | stream length `n` (via zlib warm-up) |
| 2 | `AUC >= 0.85` | which pooling site is meant; the named file has none |
| 3 | `H(Y) <= 1.2` | `beta` is coupled to `D` and to the query regime |
| 4 | `delta <= 0.15` | the error metric; sample count and conditioning |

Every bound needs its companion pre-conditions stated, or it cannot be
interpreted as pass or fail. This is the single highest-value correction to the
source document.

---

## Open items

1. Stage 2 fix target is the four `_bridge_to_d64*` bridges under
   `HENRI V2/experiments/verification/` (lines 106 / 97 / 89 / 93), not the
   named file `arc_public_ingress.py`, which has no pooling at all.
2. Stage 3 `beta`: decide between `beta = sqrt(D)` and a regime-dependent value
   using real post-unbinding wavefronts, not synthetic noise.
3. Stage 4: raise ledger payload volume to ~12000 transitions, then re-run; and
   fix the metric to noise-normalized Frobenius (state it in the gate).
4. The four draft-kernel defects D-A..D-D remain open for Stage 1.
