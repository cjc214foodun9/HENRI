# M1 RESOLVED — the egress defect is the CODEC, not the head

**Date:** 2026-09-16 | **Tree:** clean worktree `aaii-v43` @ `7750abba6f21803fe80355de3a390c0e80784d06`
**Device:** CPU (`cuda_available=false`) | **Checkpoint:** `LOADED`
`sha256 75572389083455a371546b40500b6614abfc3a245cfa0db9eba74c183a974060`, 799,034,119 B

**Verdict:** `INGRESS_LOSES_CONTENT` for the Z_256 epistemic codec;
`INGRESS_CARRIES_CONTENT` for the canonical real-wave ingress.
The inherited `BLOCKED_SEMANTIC_CAPACITY` label is **FALSIFIED as a head-capacity claim.**

---

## 1. The inherited claim

`aaii_v42_testtime_sufficiency_audit.md` recorded the text egress as
`DIAGNOSTIC_ONLY` because **16 distinct chunk waves produced the same top-1 token**
(id `29674`), `top1_token_unique = 1`. The audit I emitted carried that forward as
`BLOCKED_SEMANTIC_CAPACITY` and listed M1 as the largest single gap (40% of index
weight). Both readings blamed the **egress head**.

## 2. K-A (original) — gate was VACUOUS, withdrawn

Probe `experiments/verification/kaa_egress_probe.py`, 512 item rows.

| Arm | distinct top-1 |
|---|---|
| A `ring_uint8_scaled` | 39 / 128 |
| B `transduce_real` | 19 / 128 |
| **C `random_control`** | **37 / 128** |
| D `shuffle_control` | 19 / 128 |
| uniform reference `ln(32000)` | 10.3735 nats; measured ≈ 9.99 |

**Two design defects, both mine, both disclosed:**

1. **The gate was vacuous.** Pre-registered K-A read "distinct ≤ 1 →
   `FALSIFIED_NO_EGRESS`". Content scored 39, but **pure `torch.randn` scored 37**.
   A criterion its own negative control passes detects nothing. Verdict withdrawn
   as `GATE_VACUOUS`.
2. **`shuffle_control` was vacuous by construction.** Permuting the *same* prompt
   set cannot change the *set* of top-1 tokens; it reproduces arm B by identity.
   It tests nothing about content.

**Root cause of the vacuity:** `distinct_top1` counts *how many* tokens appear, not
*whether the token depends on the input*. With ~10 nats of entropy over 32,000
ways, a content-blind head still returns many distinct argmax values. Distinctness
is a **false-positive generator** for this mechanism.

## 3. K-A Amendment A1 — differential response at two boundaries

Pre-registered in `KAA_AMENDMENT_A1_nearfar_egress.md` **before** execution.
Measures near/far pair margins at the **wave** boundary (ingress) and the **logit**
boundary (egress), plus a **matched noise control whose margin is the floor any
real effect must beat**, and a determinism check.

Pairs: near = same topic, template differs by one word; far = different template
and different topic. n = 32 pairs, bootstrap CI (2000 resamples), seed 20260916.

### 3.1 Result — Z_256 epistemic codec ingress

Receipt `kaa-a1-nearfar__7750abba6f21__20260916T174543Z-3a39f0c7`

| Boundary | near | far | margin | CI95 | excludes 0 |
|---|---|---|---|---|---|
| W (wave) | +0.000561 | −0.000808 | **+0.001369** | [−0.00105, +0.003965] | **No** |
| L (logit) | +0.544385 | +0.544992 | **−0.000608** | [−0.006283, +0.004890] | **No** |
| noise W | +0.995039 | −0.000114 | +0.995153 | [+0.9938, +0.9966] | Yes |
| noise L | +0.997012 | +0.543976 | +0.453037 | [+0.4492, +0.4565] | Yes |

Determinism: **True** (same prompt twice → bit-identical logits).
Verdict: **`INGRESS_LOSES_CONTENT`**.

The instrument works — both noise controls separate decisively. Content does not:
near ≈ far at *both* boundaries.

### 3.2 Result — three ingresses, identical head, identical pairs

Receipt `kaa-a1-multi-ingress__7750abba6f21__*`

| Ingress | W margin | CI95 (W) | L margin | Verdict |
|---|---|---|---|---|
| `T_universal_transducer` | **+0.771927** | [+0.77181, +0.77206] | +0.330701 | **`INGRESS_CARRIES_CONTENT`** |
| `S_structured_charpos` | **+0.048210** | [+0.04779, +0.04866] | +0.025792 | **`INGRESS_CARRIES_CONTENT`** |
| `Z_epistemic_ring_uint8` | +0.001369 | [−0.00105, +0.00397] | −0.000608 | `INGRESS_LOSES_CONTENT` |

**Same head. Same pairs. Same seed. Only the ingress changed.** The canonical real
ingress separates near/far by **0.77 cosine**; the Z_256 ring codec separates by
**0.001**, with a CI that includes zero.

## 4. Root cause (mechanism, not label)

`qFHRREpistemicCodec.encode_text` produces a **Z_256 uint8 ring** built from
SHA-256-seeded `torch.randint`. Measured this session:
`encode_text('alpha')` and `encode_text('beta')` both → `[65536]` uint8,
`min=0 max=255 unique=256`. Every distinct string maps to an **independent random
ring**; similarity ≈ `1/√D` for ALL distinct pairs. A wave built this way carries
no metric structure, so no downstream head — trained or not — can recover a
near/far ordering from it. This matches the repository's own recorded property
(the structured-codec kill experiment, Run21).

The `(v/255)*2−1` adapter I used does not change that: it is a monotone rescale of
a random ring, so it preserves the absence of locality.

## 5. What this changes

| Item | Before | After |
|---|---|---|
| M1 blocker | `BLOCKED_SEMANTIC_CAPACITY` (head capacity) | **`INGRESS_CODEC_DEFECT`** — swap the ingress |
| Blamed component | `HENRINeuralEgressUnbinder` | `qFHRREpistemicCodec` |
| Sanctioned-gateway necessity | asserted by corpus (`INFERRED`) | **not required by this evidence** — the head discriminates once fed a structured wave |
| M1 status | largest single gap | **carries content on 2 of 3 live ingresses** |

This does **not** claim egress capability. It claims the *measured* failure is
upstream of the head, on a CPU probe, at 32 pairs. Semantic correctness of any
decoded token remains **`NOT_EVALUATED`**.

## 6. Honest limits

- CPU only; `cuda_available=false`. No CUDA verification exists or is claimed.
- n = 32 pairs, one seed. Margins are stable (tiny CIs) but this is one corpus.
- The verdict is a **mechanism** verdict on near/far geometry. It is **not** a
  score, not a capability claim, and does not enter the AAII composite.
- `S_structured_charpos` margin (+0.048) is real but small; `T_universal_transducer`
  (+0.772) is the strong arm.
- The Z_256 arm's `L` margin is *negative*, which is consistent with noise; it is
  not evidence of anti-correlation.

## 7. Required follow-up (pre-registered, not yet run)

**A2:** feed `T_universal_transducer` waves through the decoder and measure
whether an **open-answer** string is produced that an independent judge can score.
Until A2 runs, the open-answer members (AA-Omniscience, GDP.pdf, AA-LCR, HLE) stay
`BLOCKED_PENDING_A2` and no member is scored.

## 8. Session lesson

**A gate whose negative control can pass it measures nothing.** Both defective K-A
arms were caught by their own controls — the random arm (37 ≈ 39) and the shuffle
arm (identity by construction). Ask of every pre-registered gate: *can the negative
control pass this?* If yes, the gate is measuring the control, not the mechanism.
