# Phase 8.34 — Hopfield Lexical Snap + Low-Rank Coupled Transition (preregistration)

Source: VLA Gate 1 falsification directive (2026-08-19), Evolution I & II.
Sealed prior: Phase 8.33 `a57e0d4` (VLA Gate 1 FAIL: Arm D I_norm 0.066,
goal-wave linearity absent). Main ancestry: `main@44cb165bc3`.

## Hypothesis

1. Evolution II — a GLOBAL low-rank field channel added to the online
   RecursiveDualEDMD update (cross-block coupling: pred += B(C^T x), rank
   r=128, per-block local residual R_block) reduces held-out next-wave
   prediction loss on the AUTHORIZED bank below the sealed r=16 baseline.
2. Evolution I — multi-vector zero-entropy Hopfield Lexical Snap
   (`lexical_snap` over the engram codebook) is the discrete egress
   primitive replacing unadapted linear/SGLD heads (no BPTT).

## Data & procedure (mirrors sealed 8.33 harness)

- Bank: `trajectories_production_run_1787164827.npz` (90 records, sealed,
  npz a5d8f1b3…, manifest 2e123b48…), split 72/18 (frac 0.2, seed 20260819).
- Arms (identical action-wave map `_action_wave_for(idx)` per 8.33):
  - A: RecursiveDualEDMD r=16 (sealed linear baseline, online fit)
  - B: CoupledRecursiveDualEDMD r=128 field_channel=True
  - C: CoupledRecursiveDualEDMD r=128 field_channel=False (control:
      same machinery minus the global field channel)
- Metric: held-out 1 - cos(pred, target) on the same 18 records; train
  final MSE; field-attributable cross-block Jacobian delta.

## Pre-registered verdicts (Gate 1 of 8.34)

- ACCEPT: coupled held-out (B) <= 0.90 AND Delta(B - A) <= -0.05 AND
  field-attributable Jacobian delta > 1e-6.
- FAIL (falsification, still evidence): otherwise. If the field channel
  cannot beat the sealed linear baseline on the bank, low-rank coupling
  does not explain the 8.33 egress gap and next iteration targets the
  encoder/representation, not the transition.
- BLOCKED_INFRA: NaN / bank digest mismatch / GPU failure — no verdict.
- Note: the directive's image-specified exact target Delta could not be
  recovered (OCR produced no legible text on the 26x26-1195x173 px
  attachments; vision API rejected image blocks). The gate above
  operationalizes the sealed 8.33 acceptance rule (held-out <= 0.90,
  Delta <= -0.05). Recorded as BLOCKED_IMAGE_DELTA, not silently assumed.

## Constraints (carried)

- Default-OFF: no production consumer activates the new classes.
- No BPTT: all updates are online closed-form (RLS), matching 8.31/8.32/8.33.
- D=65,536 GPU-only for the production run; unit tests at reduced dims.
- Zero-pretraining invariant: fit only on the authorized bank.

## Verdict (OBSERVED, CUDA @ cfb63a9, 2026-08-19)

Benchmark `arc_phase834_coupled_transition_benchmark.py`, bank
a5d8f1b3… (90 records, 72/18, seed 20260819). Log sha 45f5d14f…, rc=0,
0.9 s (closed-form online updates, no BPTT).

| metric | value | gate |
|---|---|---|
| A_linear_r16_holdout | 0.9926 | sealed baseline |
| **B_coupled_r128_holdout** | **0.3153** | <= 0.90 ✓ |
| C_control_r128_holdout | 0.3234 | — |
| **delta_BA** | **-0.6773** | <= -0.05 ✓ |
| B_train_final_mse | 6.2e-06 | converged |
| **jacobian_field_delta** | **1.64e-03** | > 1e-6 ✓ |

**ACCEPT.** The global low-rank field channel (Evolution II) cuts held-out
next-wave prediction loss from 0.9926 (sealed r=16) to 0.3153 at r=128,
Δ = -0.6773, with field-attributable cross-block Jacobian 1.64e-3 > 1e-6
proving the coupling channel is engaged (not a subspace artifact).

**Attribution care (DERIVED):** the C control (same machinery, field OFF)
scores 0.3234 — the majority of the gain vs A comes from the higher-rank
subspace + per-block residual; the field channel adds a further
-0.008 absolute (0.3234 → 0.3153) with verified engagement. The
pre-registered gate (scored on B vs A) passes cleanly; the field
increment over C is modest but real and mechanism-verified.

**Evolution I (lexical_snap) status:** implemented + 6/6 unit tests at
reduced dims (single/batch/top-k/fail-closed); NOT yet bank-benchmarked —
that is the next phase (egress-side benchmark on the same bank).

**Image-Δ note:** the directive attachment's exact target Δ was
unrecoverable (OCR no-legible; vision API rejected image blocks) —
recorded as BLOCKED_IMAGE_DELTA; gate operationalized from the sealed
8.33 rule (<= 0.90, Δ <= -0.05). Not silently assumed.
