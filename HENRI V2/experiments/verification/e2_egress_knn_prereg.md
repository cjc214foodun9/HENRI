# E2 — k-NN Softmax Egress Calibration (sealed prereg)

**Spec:** HENRI-SPEC-2026-09-09-E2-KNN-SCALE
**Carrier:** carrier/e2-egress-knn-scale
**Base:** c8fc94a0 (E1 carrier tip — E2 builds on e1_egress_calibration.py, which is
NOT on main; E1 stays fail-closed, its 46-pair eval is CONSUMED and never replayed)

## 1. Premises (audited before sealing)

- Roadmap claims ">=10,000 / 125,000 / >100,000 authentic pairs" inconsistently.
  Direct read: `source_and_stage_phase5_expanded_corpus.py` exists on main but emits
  **19 hand-authored items** (verified bytes, 4 domains). No authenticated 125k-pair
  corpus exists -> roadmap scale premise FALSIFIED. This prereg pins a real source.
- Corpus: `Salesforce/wikitext-2-raw-v1` @ HF rev `b08601e04326c79dfdd32d625aee71d232d685c3`,
  license `cc-by-sa-3.0` + `gfdl` (authentic Wikipedia prose; provenance-pinned;
  license-audited). This is egress-head CALIBRATION data (same class as E1's Corberi
  text, sanctioned); it is not ARC/benchmark answer data.
- Teacher: Qwen/Qwen2.5-0.5B rev `060db6499f32faf8b98477b0a26969ef7d8b9987`, shard
  `model.safetensors` sha256 `88c14255…`, 988,097,824 B, table [151936, 896]. Already
  on remote (/root/e1-calib/teacher_embeddings.pt, hash-pinned). Frozen, zero-trainable.

## 2. Architecture (unchanged geometry, reformed probe)

E1 head chain: unpack wave [8192,8]->131072 -> factorized Stiefel down-proj
(QR retraction, no Cayley) -> LayerNorm -> SwiGLU -> target LayerNorm -> feat z in R^896.
E2 probe (default-OFF `HENRI_E2_EGRESS=1`, never imported by production runner unless
explicitly consumed): for each window feature z:
  s_j = cos(z, T_j)/tau, j=1..151936  (T = frozen teacher table)
  w = softmax(s_topk); z_hat = sum_j w_j T_j   (k-NN softmax cluster centroid)
  align = cos(z_hat, y_target), y_target = L2-normalized mean of T[token_ids(window)]
k=16, tau=0.07 (frozen). No dense [131072,131072]; factorized blocks only.

## 3. Corpus pipeline (deterministic, session-safe)

wikitext-2-raw-v1 parquet -> sentence split -> tokenize with pinned Qwen tokenizer
(remote tokenizer.json sha verified) -> windows of <=16 words (typ. <=32 BPE tokens) -> pairs
(window_text, token_ids). Ordered by (file, line); calibration 10,000 windows,
evaluation 1,000 disjoint windows. Sentence-disjoint by construction (distinct
sentences; no overlap). Seed 20260909. Split rule fixed before any training.

## 4. Gates (G1-G4, strict; G3 redefined for E2 probe)

- G1 isometry: ||W^T W - I||_F <= 1e-4   (unchanged, E1 measured 1.24e-11)
- G2 descent: last-epoch mean loss < 0.9 * first-epoch mean   (unchanged)
- G3 retrieval (NEW metric): margin = align_trained - max(align_untrained, align_random)
  >= +0.05 on the 1,000-window eval set; plus diagnostic P@1 (fraction of eval windows
  where argmax_j s_j is a target token id) and P@k (top-16 contains a target id).
  Negative control: shuffled target ids must NOT beat aligned targets.
- G4 causal consumption: 16/16 per-block rotation probes (unchanged, E1 measured 16/16)
Kill: G3 margin <= 0 after full run -> E2_GATES_FAIL, checkpoint held, sealed negative.

## 5. Staging protocol (must be followed in order)

1. Scaffold: <=16 windows, <600s, colab of E1 scaffold receipt shape.
2. Transition audit: no mock/stub/subsample shrink/hardcoded path; verify real waves
   (HighOrderCodec) + real teacher rows.
3. Full scale: 10k calib / 1k eval, GPU-exclusive RTX 5090, exact-SHA clean worktree,
   overlay `7557238908` preflight, DSN set -a.
4. On G1-G4 PASS only: export e2_egress_production.pt + SHA-256 receipt. Promotion to
   any default flip REQUIRES_APPROVAL (separate Stage 3 contract — not sealed here).

## 6. Prohibited (explicit)

- Reuse of E1's 46-pair eval, any consumed split, or re-opened sealed receipts.
- Claiming "125k pairs", "multi-domain crystalline", or Phase-5 script as corpus.
- Any main mutation, any default flip, any AAII/ARC dispatch in this carrier.
- k-NN probe compared against derived statistics instead of real wave features.
