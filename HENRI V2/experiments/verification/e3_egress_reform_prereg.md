# E3 — Egress-Head Reform (sealed prereg)

**Spec:** HENRI-SPEC-2026-09-10-E3-EGRESS-REFORM
**Carrier:** carrier/e3-egress-reform (base `5d4a399` = carrier/p1-pk-diagnostics
tip; carries e1/e2 code + the Gate 3.1 diagnostic)

## Motivation (measured, not inferred)

Gate 3.1 (`#a0a35408`, carrier/p1-pk-diagnostics @ `5d4a399`) measured the E2
contrastive head's **raw feature point** against the frozen 151,936-token table:
P@1 = 0.016, P@5 = 0.033 vs bounds 0.285 / 0.640 → `GATE31_FAIL`.

Root cause (audited from live code, not assumed): the E1/E2 head is trained with
a contrastive objective whose target is the **window teacher centroid**
(`build_window_pairs`: `target = mean(teacher_emb[window tokens])`, L2-normalized),
while Gate 3.1 scored **token-level next-token retrieval** (gold = first BPE token
of the window's last word). E3 reforms the **readout** and the **objective** at
token level. Bounds are NOT renegotiated.

## Frozen artifacts (fail-closed on absence/mismatch)

- E2 checkpoint `e2_egress_production.pt` sha256 prefix `08747c70`
  (614,545,228 B; keys verified `{config, model_state, telemetry}`)
- `teacher_embeddings.pt` sha256 prefix `48b174e9` (544,540,598 B) — frozen
  Qwen2.5-0.5B token table [151936, 896], the table E1/E2 trained against
- `tokenizer.json` + `model.safetensors` (Qwen rev `060db6499f32…`, `88c14255…`)
- Corpus `wikitext2_train.parquet` sha256 prefix `e83889ba` (36,718 rows)

## Split (identical to the sealed E2 partition and Gate 3.1)

`e2_calibrate.build_pairs_ordered(sents, E1Config(device="cpu"), emb, tok, want)`
— **no seed argument** (builder default 20260909), exactly the call
`e2_calibrate.run()` makes. First 10,000 windows = calibration; next 1,000 =
evaluation. Gold `y_i = tok(window)[len(tok(prefix))]` with the BPE
prefix-stability guard (`GOLD_TOKEN_PREFIX_MISMATCH`, fail-closed).
Label: `CONDITIONAL_SAME_CORPUS_HELDOUT` (eval windows never trained on; the
split was measured once before by Gate 3.1 — disclosed).

## Arm A — probe-distribution readout (ZERO training)

The E2 mechanism read at token granularity (the distribution-producing object the
contrastive objective actually shapes):

1. `feats_i = E2head(Psi_i)[0]` from the frozen E2 checkpoint
2. `s_j = cos(normalize(feats_i), normalize(E_j))` over all 151,936 tokens
3. `w = softmax(top-k(s)/tau)`, k = 16, tau = 0.07 (E2 frozen constants)
4. `z_hat = normalize(sum_j w_j E_j)` (k-NN cluster centroid)
5. token scores `= cos(z_hat, normalize(E))`; P@1 / P@5 / NLL as defined below

Controls (diagnostic, non-gating): untrained-init head through the same probe.

## Arm B — generative CE head (token-level objective)

- Architecture: `E1EgressHead` (Stiefel W1 [65536 → 256] QR-retracted, LayerNorm,
  SwiGLU → [896], target LayerNorm) **warm-started from the E2 checkpoint**.
- Readout: **frozen tied teacher readout** `logits = normalize(feats) @ normalize(E)^T`.
  The head's own `lm_head` is overwritten with `E` and `requires_grad=False`; it
  is never called — disclosed as a dead parameter in any export.
- Loss: `CE(logits, y) + 0.02 * isometry_penalty(W1)`; AdamW lr 3e-4, wd 1e-4 on
  adapter params; Stiefel QR retraction each step (stiefel_lr 1e-4); 3 epochs,
  batch 32, seed 20260908, fp32, CUDA. Scaffold: 16 windows, 2 epochs, batch 8.
- Gates: G1 `||W1^T W1 - I||_F <= 1e-4`; G2 CE descent `last < 0.9 * first`;
  G4 seeded per-block O(8) rotation changes token argmax >= 8/16.

## Metric definitions

`P@1 = mean I[argmax(scores_i) = y_i]`; `P@5 = mean I[y_i in top5(scores_i)]`;
`PPL = exp(mean -log softmax(scores_i)[y_i])` — **teacher-anchored single-shot NLL**,
NOT autoregressive LM perplexity.

## Pre-registered bounds (user-supplied; unchanged from Gate 3.1)

`P@1 >= 0.285` and `P@5 >= 0.640`, evaluated per arm on the sealed eval split.
Bounds are module constants, never env-overridable. **Single run, no retries.**

## Verdicts

- `E3A_PASS` / `E3A_FAIL` — Arm A vs both bounds.
- `E3B_PASS` / `E3B_FAIL` — Arm B vs G1/G2/G4 AND both bounds.
- `E3_PASS` only if both arms pass; `E3_FAIL` otherwise; `BLOCKED_INFRA` on any
  artifact/hash/device failure.
- Export `e3_egress_production.pt` ONLY on Arm B full pass (fail-closed).
- **No promotion, no main change.** A FAIL is sealed verbatim; no post-hoc
  threshold adjustment.
