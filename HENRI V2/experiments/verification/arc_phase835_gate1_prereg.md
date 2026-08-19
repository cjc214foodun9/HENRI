# Phase 8.35 — Sprint a: 5-arm VLA Gate 1 benchmark (preregistration)

Source: `HENRI-SYNTHESIS-MILLER-LEE-2026` (Drive inbox, 2026-08-19) §3.2
Phase 8.35 Implementation Directive, Tasks 1–3 + Target Verification Metrics.

Branch: `feat/phase835-analog-traveling-wave-vla`

## Arms (bank: 90 records, 72/18, seed 20260819, D=65,536)

- A: RecursiveDualEDMD r=16 (sealed 8.33 linear baseline)
- B: CoupledRecursiveDualEDMD r=128 field ON (8.34 ACCEPT arm)
- C: CoupledRecursiveDualEDMD r=128 field OFF (control)
- D: DirectionalTravelingWaveCoupler AP r=128 (+k)
- E: DirectionalTravelingWaveCoupler PA r=128 (−k)

## Metrics

1. Held-out transition loss (1 − cos(pred, target)) per arm.
2. Egress action accuracy: per held-out record, the arm's predicted
   next-wave is snapped by DualScaleAnalogLexicalSnap (top_k=512 gate)
   onto a 6-action prototype codebook built from TRAIN-split next-waves
   (mean per action; no leakage). argmax index = predicted action.
3. I_norm = I(Y; Y_hat)/H(Y) over held-out actions, from the softmax of
   snap confidences vs empirical action distribution.
4. Directional Sagnac delta (T3) between held-out prediction and target
   (directional phase-alignment check).

## Pre-registered gate (document targets)

- ACCEPT: D_holdout <= 0.15 AND I_norm >= 0.85 AND acc >= 0.80
- PARTIAL: D_holdout <= 0.15 (transition fixed) but egress gates unmet
- FAIL: transition not fixed (D_holdout > 0.15 vs sealed 0.3153)
- BLOCKED_INFRA: NaN / digest mismatch / GPU failure

No BPTT; all updates online closed-form. Components default-OFF; the
benchmark activates them. Kill criteria pre-registered: any NaN in any
arm → BLOCKED_INFRA (never scientific KILL for infra).
