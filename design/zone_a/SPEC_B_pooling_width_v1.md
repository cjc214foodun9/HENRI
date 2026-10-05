# SpecContract B — Pooling Width (DRAFT v1, NOT IMPLEMENTED)

**Status:** DRAFT. **Awaiting operator approval.** Nothing on the default path
changes until approval.

**Supersedes:** `SPEC_A_trainable_ingress_v1.md` item 1, which was **REJECTED on
its own pre-registered kill test** (`ROUTER_MOVES_NOTHING`, Δ G-U4 −0.008927,
commit `fbcc631`, recorded in SPEC_A section 10).

---

## 1. Why this contract exists

Gap #2 was attacked directly this turn. The measurement located the
compositional bottleneck, and it is **not** the missing gradient.

| probe | measured | reading |
|---|---|---|
| D1 wave pair \|cos\| | **0.148936** (random ≈ 0.011) | the **wave separates** specs |
| D2 pooled feature pair \|cos\| | **0.955315** | the **feature collapses** them |
| D5 program identification | train 0.5833 → held **0.0000** | program identity is destroyed |
| D6 input identification | train 1.0000 → held **0.2842** | input survives far better |

`VERDICT: L2_POOLING_DESTROYS_SIGNAL`.

**The document's central claim is falsified.** The attached audit asserts
`I(X; Ψ_random) ≪ H(X)`. G-U4 = **0.819028** with positive control **0.999997**
shows the wave carries recoverable information. The bottleneck is compositional
structure in the feature extractor, one layer *below* the readout algebra that
the document's Directive 3 targets.

## 2. Mechanism

`HopfieldCrossPooling.forward` computes every macro-token's similarity in
`d_k = 4` dimensions:

```
kv   = wave_proj(pair).view(b, self.n_mem, self.d_k)   # d_k = 4
sl   = einsum("bmn,bnk->bmk", att, kv)                 # [B, M, d_k]
out  = einsum("bmk,mkj->bmj", sl, self.mod)            # [B, M, d_model]
```

`resolve_n_mem(d_model)` returns `min(256, d_model // 4)`, so:
- `d_model=128`  → `n_mem=32`  → `d_k = 128//32  = **4**`
- `d_model=1024` → `n_mem=256` → `d_k = 1024//256 = **4**`

`self.mod` is `[M, d_k, d_model]`, so each macro-token's output is a 4-vector
mapped through a `[4, d_model]` slice — a **rank-≤4** contribution per token.
That is the measured collapse.

## 3. The proposed change

Widen the memory bank so **d_k ≥ 32 at every scale**:

```
resolve_n_mem(d_model, n_mem) -> min(256, max(1, d_model // 32))
```

| config | n_mem | d_k | total params | G-D2 envelope [400M, 500M] |
|---|---|---|---|---|
| **current default** | 256 | 4 | 439,798,031 | ✓ |
| **proposed** | **32** | **32** | **447,145,231** | **✓** |
| intermediate | 8 | 128 | 472,335,631 | ✓ |
| rejected | 4 | 256 | 505,922,831 | **✗ breaches by 5,922,831** |

The proposed point is **in envelope**. The `n_mem=4` variant that produced the
largest raw gain **breaches G-D2** and must not be promoted.

## 4. Measured support (small scale, diagnostic only)

`exp_widepool_seeds.py`, 5 construction seeds, bound **0.95 unchanged**:

| arm | mean | min | max | pass vs 0.95 |
|---|---|---|---|---|
| default (n_mem 32, d_k 4) | 0.921786 | 0.867919 | 0.954843 | **3/5** |
| wide (n_mem 4, d_k 32) | **0.977920** | **0.952423** | 0.991973 | **5/5** |

Mean gain **+0.056134**; `baseline_pre_reproduced: TRUE`.

## 5. Disclosed limits — these bound the claim

1. **Scale conflation.** The sweep ran at the SMALL config (`d_model=128`), where
   `n_mem=4` yields `d_k=32`. At the FULL config the matching `d_k=32` point is
   `n_mem=32`. **The gain has not been measured at full scale.**
2. **The gain is largely architectural, not learned.** Pre-training readings were
   already 0.968–0.991, so most of the movement precedes any gradient step.
3. **G-U4 is an R² proxy.** A more expressive pooling of the *same frozen wave*
   can raise held-out R² without adding one bit of information to the wave. The
   gain is real for the metric's stated purpose and its interpretation stops
   there.
4. **The wide arm used the fastest configuration, not the in-envelope one.** The
   contract specifies the in-envelope point, which is untested.

## 6. Pre-registered kill tests (frozen before the run)

| id | test | pass condition | kills the contract if |
|---|---|---|---|
| **K-B1** | baseline reproduction | pre-train G-U4 = 0.819028 ± 1e-3 at pin 20261004 | not reproduced → VACUOUS, no verdict |
| **K-B2** | in-envelope gain at small scale | d_k=32, n_mem=32: G-U4 ≥ 0.95 in ≥ 4/5 seeds | < 4/5 → REJECT |
| **K-B3** | envelope | formula total ∈ [400M, 500M] | outside → REJECT |
| **K-B4** | no readout regression | M4-G1 held-out EM must not fall below its frozen baseline + 0 | any fall → REJECT |
| **K-B5** | **full-scale transfer** | reproduce K-B2 at `dim=65536, d_model=1024`, ≥ 4/5 seeds | not reproduced → REJECT |
| **K-B6** | negative control | default d_k=4 must NOT clear 0.95 in ≥ 4/5 seeds | if it does, the gate does not discriminate → gate is broken |

**Signed criterion (D135).** The contract must improve the target metric and
degrade **nothing**. Movement alone, in either direction, is not support.

**Vacuity guards (D120/D121/D134/D137/D138).**
- Every arm reproduces the committed baseline before any delta is read.
- Any gradient-based arm asserts `grad ≠ 0` and a wave change on the **gate**
  corpus, not the 48 training rows.
- The frozen arm's wave is byte-identical before and after.

## 7. What this contract does NOT do

- It does **not** implement a straight-through estimator. Kill test #5 tested
  that path and rejected it: the router is dead because the write **address** is
  `tok % slot_dim` and the **phase** is a frozen buffer — neither depends on the
  router.
- It does **not** make the ingress trainable. The learnable-phase lever was
  measured and **hurts**: Δ G-U4 = **−0.040740**.
- It does **not** clear G-U4 by construction. The bound stays **0.95**.

## 8. Next gate

Operator decision. On approval: implement behind a default-OFF flag, run K-B1
through K-B6, and report the full-scale result of K-B5. If K-B5 fails, the
contract is REJECTED on record and the small-scale gain is recorded as a
scale-limited artifact.
