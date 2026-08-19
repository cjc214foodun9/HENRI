# Phase 8.33 — Non-linear Macro-Option Wave-JEPA (preregistration)

Source spec: `G:\My Drive\HENRI_Inbox\Wavejepa.txt` (217 lines, read 2026-08-19).
Live code: `HENRI V2/wave_jepa.py` (105 lines) — linear `RecursiveDualEDMD`
predictor (r_rank=16), ZERO consumers in `production_arc_run.py` (benchmark-only).
`efe_planner.py` live transition = `LowRankCoupledTransition` (linear coupled).

## Hypothesis
A non-linear macro-option phase-attractor transition (option codebook K=32,
opt_dim=512, compressed latent L=2048, GELU/LayerNorm phase-coupling core,
hyperspherical JEPA loss + Sagnac stress) predicts held-out next-state waves
on AUTHORIZED ARC trajectories better than the linear R-EDMD baseline:

    L_JEPA = 1 - cos(Psi_pred, Psi_target) + 0.15 * Delta_Sagnac

## Kill experiment (cheap, uses existing evidence)
Data: authorized bank `trajectories_production_run_1787164827.npz`
(90 records, sealed: npz a5d8f1b3…, manifest 2e123b48…, split 72/18 as ingest).
Procedure: fit linear R-EDMD and non-linear JEPA on the SAME 72-train split,
evaluate on the SAME 18 held-out waves.

## Pre-registered verdicts
- ACCEPT (proceed to planner seam): held-out JEPA loss < held-out linear
  loss by >= 0.05 absolute AND train JEPA loss < 0.5 (converged) AND
  Sagnac stress < 0.5 on held-out.
- KILL (archive module, default-OFF forever): non-linear held-out loss >=
  linear held-out loss (no real gain) OR NaN/divergence in training.
- Infrastructure (BLOCKED_INFRASTRUCTURE): bank corrupt / digest mismatch /
  GPU failure — no science verdict.

## Constraints (carried)
- Default-OFF: no production caller activates the module (flag
  `HENRI_ARC_NONLINEAR_JEPA=1` required; wave_jepa `use_nonlinear=False` default).
- No backprop into the encoder: W_compress is a FIXED Stiefel buffer; only
  the option codebook + transition core train (no-BPTR, matching the
  8.31/8.32 calibration discipline).
- D=65,536 GPU-only for the full-scale run (W_compress = 512 MB buffer).
  Unit tests run at reduced dims.
- Zero-pretraining invariant: fitting happens ONLY on the authorized bank.
- Corrections vs doc (recorded, not silent): per-component unit-modulus
  normalization KEPT (FHRR convention; doc divides by L = mean homodyne);
  norm clamps 1e-8 (NaN guard); egress seam lifts latent → real
  [num_blocks, 8] wave (planner boundary contract).

## Addendum (2026-08-19, kill-runner audit)
- The bank stores `actions_onehot` + `action_names` only — NO action waves.
  Both arms condition on the action INDEX via the identical deterministic
  per-action wave map (`_action_wave_for`); JEPA option id = onehot argmax
  (K=32 >= 6 actions). Constraint recorded, not silent.
- Verdict metric = ambient full-wave `1 - cos(pred, target)` on the SAME
  18 held-out records (split mirrors the calibrator: frac 0.2, seed
  20260819). JEPA train metric = latent JEPA loss.

## Verdict (OBSERVED, CUDA @ f3a9299, 2026-08-19)
Log sha 316aaf9a, bank sha a5d8f1b3…, 90 records (72/18), 400 epochs, 6.4 s.

| metric | value | gate |
|---|---|---|
| edmd_holdout_loss | 0.9926 | baseline (~random on S^D-1) |
| jepa_holdout_loss | 0.9085 | — |
| delta_holdout | **-0.0841** | <= -0.05 ✓ |
| jepa_train_jepa_loss | 0.0158 | < 0.5 ✓ (converged) |
| jepa_sagnac_holdout | **0.9915** | < 0.5 ✗ |

**CONDITIONAL — not promoted, not killed.** The non-linear macro-option
transition core is REAL: it converges (train 0.0158) and generalizes better
than the linear R-EDMD baseline (Δ = -0.0841). The ambient held-out gate
FAILS: predicted full waves are still ~orthogonal to targets (cos ≈ 0.09,
sagnac 0.99). Per prereg, the module stays default-OFF and does NOT enter
the planner seam.

**Attribution (INFERRED):** the bottleneck is the latent→ambient LIFT, not
the transition. Train-in-latent loss 0.0158 proves the transition learns;
ambient orthogonality mirrors the 8.32 calibrated-head result (MSE 24.22):
wave→meaningful-content projection fails at the egress boundary. Both
failures point at the SAME gap: the wave representation does not place
task-relevant structure where a linear lift can read it.

**Next options:** (1) latent-space held-out evaluation on the same bank
(target compressed too — isolates lift vs transition, ~1 min work);
(2) representation/egress work (carrier dominance, K1 functor lessons);
(3) leave sealed as evidence, planner seam untouched.
