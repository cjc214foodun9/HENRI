# E1 — Calibrated Egress Projection Head (frozen-backbone teacher alignment)

**Spec ID:** HENRI-SPEC-2026-09-E1-EGRESS-CALIBRATION
**PREREG SHA-256:** computed at seal (see receipt below)
**Date:** 2026-09-08
**Carrier branch:** `carrier/e1-egress-calibration` @ base `10f5f23` (main, G7 + launch-fix)

## Purpose

Train a calibrated projection head that maps canonical HENRI real wave states
`[num_blocks, 8]` (D = 65,536, G7 HighOrderCodec / zone_c_world_knowledge_codec family)
into the frozen embedding space of a pinned, license-clear LLM teacher
(Qwen2.5-0.5B, apache-2.0, repo sha `060db6499f32faf8b98477b0a26969ef7d8b9987`).
This replaces NO production path. It is a new default-OFF calibration carrier.

## Premise audit (supplied packet, disposition)

| Supplied component | Disposition |
|---|---|
| Skeleton (Procrustes warm-start → Stiefel → InfoNCE → receipted export) | `BOUNDED_IMPLEMENTABLE` — reused as design language |
| `SyntheticHolographicCorpusDataset` | `BLOCKED_MISSING_PREMISE` — random waves/teacher = mock loop; fixture only, never efficacy evidence |
| `cayley_retraction` at D=65,536 | `CONFLICTS_WITH_LIVE_CODE` — [131072,131072] ≈ 69 GB; use factorized thin-SVD Stiefel (QR reduced), no dense [D,D] |
| `OrthogonalProcrustesInitializer` on 256 batch | `BLOCKED_MISSING_PREMISE` — rank ≤ 255; use calibration split N ≥ 4096 real pairs |
| Gold-walk predicate gate (`score ≥ 1.0` → export) | `FALSIFIED` as promotion gate — self-constructed composite, vacuous; replaced by external heldout alignment |
| `trigram_weight` in loss | `FALSIFIED` — dead config; removed |
| `trust_remote_code=True` | `CONFLICTS_WITH_LIVE_CODE` — unnecessary for Qwen2.5; removed |
| "auto-couples into henri_decoder.py" | `REQUIRES_APPROVAL` — never automatic; E1 is default-OFF and no runner import |

## Representation boundary (frozen)

- Wave side: real `[num_blocks, 8]` float32 (NUM_BLOCKS=8192, BLOCK_DIM=8) via
  `g7_highorder_codec.HighOrderCodec.encode()` on text windows. Flatten to `[B, 65536]`
  at the head boundary. Assert `d_model == num_blocks * block_dim`.
- Teacher side: `Qwen/Qwen2.5-0.5B` input embeddings `[151936, 896]`, frozen,
  revision pinned `060db6499f32`, per-shard SHA-256 verified on disk before use.
- Head geometry: `[65536] → Stiefel W1 [65536, d_bottleneck] → LayerNorm →
  SwiGLU → [d_target=896] → LayerNorm`. Logits optional (no token-id supervision
  in E1; CE head reserved for a later carrier).
- Dense-ban: no parameter or intermediate with both dims ≥ 65536. Factorized only.

## Training contract

- Windows: text from `corberi_2609_04732.txt` (27,208 B, sha recorded), split by
  sentence; calibration windows = last ≤ 24 words before the next word.
  Pair `(wave, target)`: `wave = codec.encode(prefix)`; `target = mean of Qwen
  BPE token embeddings of the full window`, L2-normalized.
- Seed 20260908. Calibration/eval split DISJOINT by sentence order
  (first 80% calib, last 20% eval; recorded explicitly).
- Loss: InfoNCE contrastive (temperature 0.07) + 0.02·‖W₁ᵀW₁−I‖_F² (isometry).
  No CE term in E1 (declared).
- Optimizer: AdamW (adapter/norm only), lr 3e-4, wd 1e-4, clip 1.0.
  Stiefel param updated by QR retraction of (I − lr·grad·gradᵀ-style) step:
  factorized: W ← qr(W − lr·G) reduced, sign-corrected. No Cayley.

## Flags and gates

- Default-OFF: `HENRI_E1_EGRESS=1` gates head construction; production
  `henri_decoder.py`/`production_arc_run.py`/`henri_egress.py` MUST NOT import
  the trainer module (contract test greps).
- Promotion gate (heldout, single-use, sealed before run):
  G1 isometry: ‖W₁ᵀW₁ − I‖_F ≤ 1e-4 (reduced QR, production shape, CUDA).
  G2 calibration loss descent: mean loss last 20% < 0.9 × first 20% of first epoch.
  G3 eval NDCG/P@1 (k = 4, batch 256): E1 head > random-wave baseline (1/√d) AND
  > untrained-init head, with ≥ 0.05 absolute margin. (Real-waves baseline.)
  G4 causal consumption: a seeded O(8) per-block row rotation changes argmax
  teacher-embedding retrieval for ≥ 8/16 probes (diagnostic, mirrors K2/U2).
- Kill criteria: any G fails → `E1_FALSIFIED` / `E1_BLOCKED_*`; checkpoint held
  (no export). G2 failure with loss NaN → `BLOCKED_INFRASTRUCTURE`.
- Scaffold protocol (≤16 pairs, ≤600 s, toy D=4096 CPU): prove shapes, loss
  finite/descent, retraction engages. Transition audit: mock-stub scan,
  subsample-shrink scan, hardcoded-answer scan. Only then full-scale GPU run,
  GPU-exclusive (no concurrent pytest/ARC).

## Approval state

- No main change in this carrier. Promotion to any production egress path is a
  separate approval gate (sealed human decision).
- E1 never writes Zone C, never touches `models/`, never exports unless G1–G4 all pass.

## Receipt

To be sealed with: prereg sha256, code sha256, remote verify RC, gate table, verdict.
