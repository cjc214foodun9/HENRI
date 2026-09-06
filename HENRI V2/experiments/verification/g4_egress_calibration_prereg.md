# Carrier G4 — Egress Calibration (Zone C world knowledge + HENRI pipeline)
## Pre-registration (2026-09-05)

Branch: `carrier/g4-egress-calibration`. Base: `origin/main` @ `f2204692ce01b3d2eb1ca9715c480a33953b5396`.
Worktree: `C:/Users/chan/henri-worktrees/carrier-g4-egress`. `/henri-bundle` active.

## 1. Objective

Measure the HENRI egress chain against the live K5 Zone C world-knowledge corpus
(27 sources / 3,433 chunks / 262,144-byte `[8192,8]` float32 waves, prod DSN) and emit a
calibrated receipt. NO semantic task score. NO AAII claim. NO DB writes (read-only SELECT).
No checkpoint mutation (diagnostic load only).

## 2. Arms (all read-only, deterministic)

**ARM R — retrieval diagnostic (pipeline: query → K5 codec → pgvector HNSW → top-k).**
- 16 source-grounded queries: 13 `computing/*.rst` + 3 distinct Gutenberg sources (unique
  extraction per file; probe defect fixed — 3 txt files shared one sentence).
- Metrics: correct-chunk rank in top-5, top-1 sim, ms latency; null distribution
  (15 random unit vectors) for margin calibration.
- Acceptance R_a: unique query `P@1 >= 0.80` AND `median(grounded top1 sim) > 2 * null_mean`.

**ARM U — unbinder causal-consumer diagnostic (sealed K2/U2 method).**
- Overlay `henri_decoder_checkpoint.pt` (sha `7557238908…`, 799,034,119 B) loaded into
  `HENRINeuralEgressUnbinder`; `forward` + greedy argmax over `code_vocab_map`.
- Seeded per-block O(8) rotation (row-norm preserving) on 16 canonical chunk waves.
- Metrics: token-bytes changed, logits moved, A-repeat / A-restore byte-exactness,
  mismatched-query control.
- Verdict labels: `CAUSAL_CONSUMER_DIAGNOSTIC` (if rotation-sensitive) +
  `SEMANTIC_CAPACITY_BLOCKED` (if semantic readout not demonstrated). NEVER task accuracy.

**ARM H — Hopfield lexical-snap capacity verification (pipeline contract).**
- Word engrams from corpus vocabulary (M in {250, 1000, 2500, 5000, 10000}), K5 codec
  waves (flat D=65,536, L2-normalized) into `ContinuousHopfieldCleanup(beta=8.0)`.
- Metrics: P@1 exact, P@1 one-word-substituted, random-negative precision.
- Verdict: `CONTRACT_VERIFIED` if `P@1(exact) >= 0.98` at M ≤ 10,000; else `FALSIFIED`
  with actual values recorded.

## 3. Bounds

- Wall ≤ 600 s. Queries 16. M ≤ 10,000. Seed fixed (20260905).
- Env: `ZONE_C_PROD_DSN` from `/workspace/zonec_prod.env`; torch CUDA; exact-SHA worktree.
- Receipt: per-row JSONL + final `g4_egress_calibration_receipt.json`; pull + SHA-256.

## 4. Honesty / kill rules

- Any DB write, optimizer step, checkpoint save, or promoted flag = FAIL.
- `score_eligible` is never true. No accuracy vs AAII items.
- Any arm infra failure = `BLOCKED_INFRA` row, never silent substitution.
- Local CPU tests are software checks only; the receipt comes from the CUDA run.
