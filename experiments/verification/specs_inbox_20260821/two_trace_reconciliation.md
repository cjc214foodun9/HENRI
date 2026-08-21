# Spec-to-Live Two-Trace Reconciliation (2026-08-21)

Documents governed: HENRI-SPEC-EPISTEMIC-ENGRAM-ENGINE-2026 (`316536b5…`),
HENRI-SPEC-VLA-VERIFICATION-2026 (`34f5e2bb…`) — copies under
`experiments/verification/specs_inbox_20260821/`.
Status: governance artifacts for the NEXT phase. They are not executable
authority, benchmark evidence, or permission to ingest evaluation content.

Evidence labels: OBSERVED (live tool/CUDA/DB), DERIVED, INFERRED,
TARGET_GOAL (spec-declared target, not measured), FALSIFIED (live
measurement contradicts), CONFLICT (spec vs live contract mismatch).

---

## Trace 1 — Epistemic Engram Engine (spec → live path)

1. Spec requirement: Zone C engrams as content-addressable epistemic world
   knowledge in Z_256^D, retrieved into the wave loop. [TARGET_GOAL]
2. Live schema: `HENRI V2/migrations/zone_c_schema.sql:63`
   `zone_c_engrams(time, axiom_id, domain_tag, phase_vector VECTOR(2000),
   sagnac_stress)` hypertable + `zone_c_engrams_domain_time_idx` btree.
   OBSERVED.
   - CONFLICT: spec mandates DUAL tables `zone_c_ast_engrams` +
     `zone_c_action_engrams` with dual-subspace isolation; live DB has ONE
     table `zone_c_engrams`. The dual names appear only in
     `scripts/staging/ingest_mbpp_codebook.py:12`.
3. Write path: `darwinian_phase_swarm.py:367 checkpoint_wave →
   segment_cache.checkpoint`; runner gate `production_arc_run.py:2289`
   (`step % CHECKPOINT_EVERY == 0 and not learning_frozen()`).
   OBSERVED: with `HENRI_FREEZE_LEARNING=1` the engram count is STATIC at
   10,826 across the arm A v2 window (frozen eval suppresses writes).
   The CLASS48 v1 arm ran WITHOUT the freeze and wrote +123 engrams
   (10,703 → 10,826) → classified `BLOCKED_INFRASTRUCTURE` (evidence:
   `/tmp/ab_arm_a_CONTAMINATED*` on Vast).
4. Read path: `production_arc_run.py:1385-1410` — every 5 steps
   `zonec_bridge.retrieve(state_wave.cpu(), top_k=4)` (or legacy
   `orch.segment_cache.retrieve`), blend `0.7*state + 0.3*recalled`,
   renormalize, then relaxation → EFE action selection. OBSERVED in log:
   `recall 0/4` alternating (mechanism engagement, not task success).
5. External outcome: per-env scorecard + `levels_completed`; telemetry
   `production_run_*.jsonl`. Trace is causally closed for arm A v2.

## Trace 2 — Benchmark Goal (canonical → metric)

1. Datasets: HumanEval (164), MBPP (257), staged under
   `data/official_benchmarks/staged_eval_suites/` with manifest digests.
   OBSERVED (staging exists); canonical LF digests enforced.
2. Evaluator: two-mode sandbox (namespace probe gate / container-rlimit);
   REPL unit tests; `score_eligible` gated by `checkpoint_load_status==LOADED`
   + zero execution errors. OBSERVED.
3. Live item outcomes (measured, NOT spec):
   - HumanEval 2/50 (8.39 campaign wave-AST egress) — FALSIFIED vs spec 164/164.
   - MBPP 15/500 best (run15) — FALSIFIED vs spec 257/257.
   - MMLU-Pro 81.6% / GPQA 72.2% — NO live item-level evidence; TARGET_GOAL.
   - ARC-AGI-3 62.75% — NO live evidence; best observed sp80 Level 1 (4.76)
     — TARGET_GOAL vs spec; CLASS48 arm A in flight (frozen).
4. Spec "TOTAL GAUNTLET 83.22% REPEATABLE VERIFIED" — FALSIFIED by the
   item-level record above; do not cite as OBSERVED.

## Contracts with live-code discrepancies (recorded for next-phase packet)

- Sagnac veto: spec gate >= 0.10; live production default `tau_veto = 0.35`
  (`sagnac_mcts_planner.py:142`) with dynamic expansion; CLASS48 Gate S uses
  0.35. CONFLICT (must be resolved in the next packet, not silently).
- O-VSA Zone A binding: spec mandates; live default-OFF gated import
  (runner ~1566-1574), O-VSA INGRESS FALSIFIED (`65aef1b`). CONFLICT.
- I(Psi;Y) > 0.85 bits: measured 0.39 nats on the 500-step SGLD protocol;
  units differ; TARGET_GOAL, not verified.
- VRAM ceiling 12.5 GB: no conflict (run uses 9.8 GiB GPU). OBSERVED.

## Realization gate for the functional goal

The spec goals are realized through staged pre-registered packets with
external-outcome gates. CLASS48 (in flight, frozen) is the current packet.
Next packet (after CLASS48 closes): one smallest causal mechanism from the
reconciled list, default-OFF flag, clean Zone C snapshot/isolation plan,
external-outcome gate, cheap kill criterion. Do NOT reopen sealed
codec/ranking classes without a materially new semantic representation with
an authentic causal consumer.
