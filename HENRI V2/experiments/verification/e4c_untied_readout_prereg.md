# E4c — Untied Readout Scaffold: 2×2 Factorial on Frozen-Backbone Features (sealed prereg)

**Spec:** HENRI-SPEC-2026-09-10-E4C-UNTIED-READOUT
**Carrier:** carrier/e4-construct (base `7df1ee1` = carrier/e3-egress-reform tip)
**Supersedes:** the E3 arm design. E3 changed tying AND trainability together and
reported one verdict, so it could not attribute its own result.

## Contract change being exercised (user-directed, 2026-09-10)

Token-level scoring is supplied by a **frozen, revision-pinned, contamination-reviewed
pretrained backbone**. This carrier does not train, fine-tune, or update the backbone.
Backbone is eval-only, `torch.no_grad()`, zero trainable parameters, fail-closed on
shard SHA mismatch.

## The correction that makes this carrier non-vacuous (R1/R2, adopted)

The readout consumes the **frozen backbone's final hidden state**, `h ∈ R^896`, NOT a
wave feature. Rationale (measured, E4b): the g7 wave codec cannot identify the
terminal word position (`last_word_to_front` cos 0.940 aggregate on the bag codec;
E4b's reserved position channel fixes the channel but is not this carrier's variable).
Feeding wave features into the 2×2 would reproduce the E3 representation failure and
all four arms would fail for a reason unrelated to tying/trainability.

`d_in = 896` is not guessed: it is the frozen table's `TEACHER_DIM` and the backbone's
`hidden_size` (both independently verified: `e1_egress_calibration.TEACHER_DIM = 896`;
`config.json` `hidden_size = 896`).

## Frozen artifacts (fail-closed on absence / SHA mismatch)

- Backbone `Qwen/Qwen2.5-0.5B`, rev `060db6499f32faf8b98477b0a26969ef7d8b9987`
  - shard `model.safetensors` = **988,097,824 B**, sha256 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`
  - `config.json` (hidden_size 896, vocab 151936, 24 layers, 14 heads)
  - `tokenizer.json` 7,031,645 B
- Teacher table `teacher_embeddings.pt` = **544,540,598 B** (the tied matrix / oracle readout)
- Corpus `wikitext2_train.parquet` sha256 prefix `e83889ba`, 36,718 rows

## Split (FRESH — never previously measured)

Identical construction to the sealed E4a audit:
`e2_calibrate` window rule (`sentence_split` → whole sentence → `prefix = words[:-1]`),
gold `y = tok(window)[len(tok(prefix))]` with the BPE prefix-stability guard.

- calibration windows **11,000 … 20,999** (10,000)
- evaluation windows **21,000 … 21,999** (1,000)
- Label: `CONDITIONAL_FRESH_SPLIT_SAME_CORPUS`
- The E2/E3/Gate-3.1 windows 0…10,999 are **consumed** (measured 3×) and are NOT used.

## Baselines (from sealed E4a, same construct, fresh split)

- context-free marginal: **P@1 = 0.440**, P@5 = 0.572
- strongest trivial (prefix-backoff): **P@1 = 0.483**, P@5 = 0.586
- uniform CE `ln 151936` = 11.9312 nats; marginal H(gold) ≈ 4.008 nats

## Four arms (2×2: tying × trainability) — all share input `h`

| Arm | Readout | Init | Trainable | Isolates |
|---|---|---|---|---|
| **a** | untied `[896→151936]` | N(0, 1/√896) | **yes** | the proposed treatment |
| **b** | tied `E` | teacher table `E` | **no** | exact E3 Arm-B control |
| **c** | tied `E` | teacher table `E` | **yes** | **decisive**: if c ≈ a, E3's failure was trainability, not tying |
| **d** | untied `[896→151936]` | N(0, 1/√896) | **no** | untied geometry with no learning |

**Arm (d) MUST be random-init.** A teacher-init untied-frozen arm is mathematically
identical to arm (b) and the arm would be vacuous (R1 correction, adopted).

## Oracle reference (diagnostic, non-gating)

Frozen backbone's **own** `lm_head` argmax on the same eval windows. This is the
ceiling for any linear readout on the same features: if the oracle is near the
marginal baseline, the construct is unwinnable and the carrier is `BLOCKED_CONSTRUCT`.

## Training

- Loss `CE(logits, y)`; AdamW lr 3e-4, wd 1e-4, batch 32, **1,000 steps**, seed 20260910
- Trainable arms (a, c) only; frozen arms (b, d) are evaluated without any step
- Features precomputed once by the frozen backbone and cached to disk (backbone
  forward is not in the training loop)
- fp32 for the readout; backbone bf16 under `no_grad`

## Pre-registered gates

- **G-A** `E4C_WIRING_OK`: all four arms produce finite logits of shape `[B, 151936]`.
- **G-B** `E4C_ORACLE_VIABLE`: oracle P@1 ≥ marginal baseline + 0.05. If this fails the
  construct is the blocker, not the readout → `BLOCKED_CONSTRUCT`.
- **G-C** `E4C_MOVEMENT`: at least one trainable arm exceeds the **per-construct**
  marginal baseline (0.440) by ≥ +0.05 within 1,000 steps.
- **G-D** `E4C_ATTRIBUTION`: report a vs c. If `c ≥ a − 0.02`, the E3 failure is
  attributed to **trainability**, not tying.
- **G-E** `E4C_LEAN_LEDGER`: trainable vs buffer vs frozen bytes reported separately.
- **G-F** fail-closed: unscorable if the backbone shard SHA differs, the split replay
  differs, or any receipt key is missing.

## Kill criterion

**No trainable arm above the 0.440 marginal baseline after 1,000 steps → `TERMINATE`.**
(The roadmap text hard-coded 0.432 — that is the OLD split's marginal. The bound is
parameterized per construct: 0.440 here. Hard-coding 0.432 would be a stale constant.)

## Parameter ledger (OBSERVED from the live class, counted not estimated)

| Component | Params | fp32 bytes |
|---|---:|---:|
| untied readout `[896 × 151936]` | 136,134,656 | 544,538,624 |
| AdamW state (2 moments fp32) | — | 1,089,077,248 |
| tied matrix (frozen buffer) | 136,134,656 | 544,540,598 (file) |
| frozen backbone shard | ~494 M | 988,097,824 (file) |

## Verdicts

`E4C_GATES_PASS` / `E4C_MOVEMENT_FAIL` / `E4C_ATTRIBUTION_*` / `BLOCKED_CONSTRUCT` /
`BLOCKED_INFRA`. No promotion, no main change. Bounds not renegotiated post-hoc.
