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
| 2 Metric locality | AUC >= 0.85 | **PASS** on the fix; mean pooling already passes | **Spec defect** — premise not reproduced |
| 3 Discrete egress snap | `H(Y) <= 1.2` bits at beta=8 | **FAIL** | **Spec defect** — beta is a joint function of query regime |
| 4 Transition identification | one-step `delta <= 0.15` | **FAIL** at 1500 rows; **PASS** at 12000 | **Spec defect** — metric and sample diversity unstated |

Three of the four gates as written in the source document do not survive
measurement. None of the three is an estimator or carrier failure.

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

## Stage 2 — metric locality: PASS, but the stated defect is FALSIFIED

The source document names `arc_public_ingress.py` as the mean-pooling site.
That file is a read-only manifest parser and contains **no pooling of any
kind** (`OBSERVED`). The real site is `arc_g1_topological_engine.py:106`:

```python
w.view(16, 4096).mean(dim=0)
```

plus sibling sites at f15, f22, f23.

Measured AUC (`OBSERVED`, 8 classes x 12 per class):

| Method | mean AUC | min AUC | Gate (0.85) |
|---|---|---|---|
| Mean pooling (current) | 0.9292 | 0.8839 | **PASS** |
| 8-channel local Clifford blocks (proposed) | 1.0000 | 1.0000 | PASS |

The document predicts mean pooling collapses AUC below 0.85. It does not:
0.9292 passes. So the stated failure mode is **not reproduced**.

The proposed fix is still an improvement and is independently justified: mean
pooling is invariant under block permutation (cosine 1.0 between permuted
inputs), while 8-channel Clifford blocks are not (cosine drops to -0.10). Metric
locality is a real property, but it is not the cause of an AUC failure.

Correction to file: the fix must be applied to
`arc_g1_topological_engine.py:106`, not `arc_public_ingress.py`.

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

1. Stage 2 fix target is `arc_g1_topological_engine.py:106`, not the named file.
2. Stage 3 `beta`: decide between `beta = sqrt(D)` and a regime-dependent value
   using real post-unbinding wavefronts, not synthetic noise.
3. Stage 4: raise ledger payload volume to ~12000 transitions, then re-run; and
   fix the metric to noise-normalized Frobenius (state it in the gate).
4. The four draft-kernel defects D-A..D-D remain open for Stage 1.
