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

## VLA Gate 1 egress verdict (OBSERVED, CUDA @ 309ae63, 2026-08-19)

5-arm egress experiment `arc_phase833_egress_sgld_experiment.py`
(A linear-obs / B SGLD-obs / C linear-goal / D SGLD-goal / E closed-loop
JEPA→head), production D=65,536, 500 SGLD steps, same bank (90 records,
72/18, seed 20260819). Clean-run log sha `4a23ae7999…` (crash run @
4b68735 sha `24bdbf06…`, Arm E float() bug, rc=1, no verdict — fixed at
309ae63, re-run rc=0).

| arm | acc | top1_rank | I_norm | entropy (nats) |
|---|---|---|---|---|
| A linear obs (psi_t) | 0.111 | 3.78 | 0.301 | — |
| B SGLD obs (psi_t) | 0.111 | 3.61 | 0.298 | 0.192 |
| C linear goal (next_wave) | 0.167 | 3.06 | 0.167 | — |
| **D SGLD goal (next_wave)** | **0.222** | **3.06** | **0.066** | **0.169** |
| E closed-loop (JEPA pred) | 0.056 | 2.72 | 0.228 | pred_cos_goal 0.093 |

**VLA Gate 1 verdict: FAIL** (Arm D: I_norm 0.066 < 0.85, acc 0.222 <
0.80, entropy 0.169 < 0.896). Gate: PASS requires I_norm ≥ 0.85 AND
acc ≥ 0.80 AND entropy < 0.5·ln(6).

**Interpretation (OBSERVED, per pre-registered falsification):** SGLD
egress adaptation does NOT restore I(Ψ_goal; Y) under goal-wave
conditioning. Training converges (Arm D train_final_loss 0.0981, Sagnac
0.0556, yield never fired) yet held-out discrimination is at chance —
including Arm C (linear on goal wave, 0.167) and Arm E (predicted wave
cos 0.093 to target). Goal-wave linearity is ABSENT in the current
representation.

**Governance outcome (per roadmap):** falsification is evidence-backed.
Egress head machinery (Option 2 SGLD path) is not the bottleneck fix.
Next iteration targets REPRESENTATION STRUCTURE (Evolution I & II:
Hopfield Lexical Snap + Low-Rank r=128 Field Coupling), not head
machinery. No VLA Gate 1 PASS, no benchmark scores, no planner wiring
claimed.

**Fixed defects (sealed):** Arm E `pred_cos_goal` parenthesization bug
(multi-element tensor → ValueError on float(), verdict never emitted;
committed as 309ae63).
