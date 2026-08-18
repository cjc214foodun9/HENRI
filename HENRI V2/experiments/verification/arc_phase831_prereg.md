# Phase 8.31 — Algebraic (No-BPTT) Semantic Action Head + Pipeline Profiling
## Pre-Registration (2026-08-18) — NOT yet executed for production.

Source: `G:\My Drive\HENRI_Inbox\8.31.pdf` — sha256 `b0eb8084c528217072b0f3dfd2604cac74dea1317f161e1b08f3a8434b52dafa`, 3 pp, 6,406 chars (extracted `C:\tmp\eight31_full.txt`).
Branch: `feat/phase831-algebraic-action-head` @ base `main@5005410` (verified 511/4/0 remote, 512/3 local). Worktree `C:/tmp/henri-831-wt`.

## 1. Calibration mechanisms (PDF §3) vs live code audit
| PDF mechanism | PDF file | Live file (audited) | Status |
|---|---|---|---|
| A. Closed-form Procrustes `W_task = Y X†` | universal_data_transducer.py | universal_data_transducer.py — NO Procrustes | **NEW WORK** (low-rank, D=65,536-safe) |
| B. Hopfield lexical snap | hopfield_cleanup.py + henri_egress.py | `henri_egress.py:22` already imports `ContinuousHopfieldCleanup`; `store_engrams`/`hard_retrieve` live | ALREADY WIRED (no-op to redo) |
| C. In-situ SGLD creep | wave_jepa.py | `henri_decoder.py:133 adapt_in_context_sgld` (callers: verify_egress_closed_loop.py, henri_decoder.py:705) | LIVE ELSEWHERE — module-name correction |
| Wiring target | production_arc_run.py action path | `arc_action_head.py` loader (`head.weight`/`head.bias`, strict provenance, `trained_action_head_active` gated) | Existing BPTT-head contract; algebraic head must produce SAME artifact contract or a validated sibling |

## 2. Honest calibration boundary (ARC)
- ARC-AGI-3 arcade games expose NO public demonstration pairs (`BLOCKED_NO_DEMOS` observed ×20 run4, ×1 smoke).
- No authorized labels → calibrate-from-pairs reports **`BLOCKED_NO_ACTION_TRAJECTORIES`**; never derive labels from hidden evaluation files or pseudo-demos.
- The algebraic head module, validator, and gates are built and software-verified on synthetic fixtures; this is NOT capability evidence.
- Until an artifact passes every gate: `trained_action_head_active=false`, `score_eligible=false`, `diagnostic_only=true`, terminal reason `ACTION_HEAD_NOT_CALIBRATED`.

## 3. Pre-registered gates (G1–G10)
| Gate | Criterion | Status target |
|---|---|---|
| G1 | Artifact schema `henri.algebraic-action-head.v1` + exact action ordering == canonical | 100% match |
| G2 | Out-of-vocabulary emission | 0 (typed fail-closed) |
| G3 | Held-out true action rank | ≤ 2 |
| G4 | Held-out margin (sim_top1 − sim_top2) | ≥ +0.05 |
| G5 | Held-out enum accuracy vs matched random/legal-frequency control (n_heldout ≥ 20, binomial p<0.05) | PASS |
| G6 | Generic decoder ON + algebraic head OFF → still ineligible (contract) | score_eligible=false |
| G7 | Missing/corrupt/wrong-basis artifact | typed `ActionHeadError` |
| G8 | ACTION6 `(GameAction, data)` payload acceptance measured independently | conjunctive |
| G9 | Default-OFF path byte-identical (max tensor diff 0.0) | PASS |
| G10 | Matched-seed A/B vs current EFE policy (same envs/seeds/budgets, sequential GPU) | post-calibration |

Kill: ANY gate fails → mode default-OFF, unpromoted. No calibration labels → `BLOCKED_NO_ACTION_TRAJECTORIES`.

## 4. Speed optimization discipline (PDF §2/§3.1)
- Build the stage profiler FIRST; optimize only measured bottlenecks (median/p95/p99, VRAM peak, allocations, kernel counts; environment/network time separated from HENRI compute).
- Stages: frame ingress+segmentation → wave encode → candidate construction → EFE/RT/Sagnac scoring → action decode → ACTION6 payload → `game.step` → Zone C → telemetry.
- Bounded candidates (after measurement): carrier/vocab caching, batched candidate scoring without `[B,M,D]`, preallocated input-device scratch, Triton only with CUDA-input guards, no `.item()` in hot loops, JSON/hash off critical path, Zone C downsampling, CUDA Graph only after shape-equivalence proof.
- PDF's C++ harness + batched-subprocess envs are architectural proposals — measured profiling decides.
- Existing observed rates: B512 ≈ 110 Hz, B4096 ≈ 15 Hz (Phase 8.28). **20 kHz = TARGET_GOAL, not evidence.**

## 5. Sequencing
1. Module + strict artifact validator (this commit). 2. Deterministic algebraic calibration. 3. Held-out gates. 4. Wire into live caller ONLY after gates pass (default-OFF flag `HENRI_ARC_ALGEBRAIC_ACTION_HEAD`). 5. Remote verification post-run5 (GPU-exclusive). 6. Profiler run on remote. 7. Matched A/B.
