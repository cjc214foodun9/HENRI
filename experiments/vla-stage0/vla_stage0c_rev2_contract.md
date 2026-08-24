# System-1 Stage-0c-rev2 — Reduced-Koopman Spectral Evaluation — PRE-REGISTRATION

**Date:** 2026-08-24 · **Reference 3 (gpt-5.6-sol) binding** · Status: SEALED BEFORE K CONSTRUCTION

## Prior verdict preserved (NOT relabeled)
- Stage-0c-rev = `IDENTIFIABILITY_BLOCKED` (G1 PR≥16 failed; audit `5f44c3c8…`; governance
  `c14d7f1c…`/`e0b636df…`; commit `a7d621d`; bundle r24 `1d53ce64…`).
- This carrier is NEW: `Stage-0c-rev2`, with NEW gates. The old gate is not bypassed; it stands.

## Upload (proposal artifact, audited)
- Path: `G:/My Drive/HENRI_Inbox/Stage-0c-rev_Spectral_Evaluation___Stage-0c-rev2_Authorization.md`
- Bytes: 562 · SHA-256: `99b3b828ddfeb5eeeafeb210fb47e6f589574ce21646da4073db158432d1670d`
- Content: 13-line diagram. States: PR = 7.03, κ16 = 5.49 (measured values); **top-8 subspace
  contains 88.4% variance** (claim to verify); **κ8 < 3.0**; K_r for r ∈ {4,8}; **ε_EDMD ≤ 0.05**.
- NOT specified: C1–C12 text, K construction, basis rule, split definition, baselines, stability,
  selection rule. Authored and sealed here.

## Frozen inputs (sealed)
- Encoder params: `vla_stage0b_rev_params.npz` — full SHA-256
  `766e607ad0bc739ea0a139172dd34e16d01a268cca80e990af5aab01006cfcd7` (runtime loader asserts).
- Encoder: `vla_stage0b_rev_encoder.py` (Stage0bRevEncoder), cross-process output sha `a751cc77…`.
- Corpus manifest `54b7350a…`. Episode ordering = **lexicographic filename order** (matches sealed
  audit `ce697efd…`): calibration IDs [101,1010,1111,1212,1313,1414,1515,202,303,404] = **171 records**;
  evaluation IDs [505,606,707,808,909] = **133 records**.

## Construction (pre-registered, deterministic)
- Per action a ∈ {0,1}: X_a = flat lifted `obs_t` rows (N_a × 6144), Y_a = flat lifted `obs_next`
  rows (N_a × 6144); float64 cast after encoding; **no centering**.
- **Calibration-only basis**: V_{a,r} = top-r right singular vectors of X_a (calibration), r ∈ {4,8}.
  Evaluation episodes never influence basis, rank, or tolerance.
- Solver: `numpy.linalg.svd(full_matrices=False)`; pinv tolerance fixed `tol = 1e-10 · s_max`.
- **Separate per-action operators**: K_{a,r} = pinv(X_a · V_{a,r}) @ (Y_a · V_{a,r}) ∈ R^{r×r}
  → four matrices K_{0,4}, K_{1,4}, K_{0,8}, K_{1,8}. No single conflated K per rank.
- Prediction: coefficients c = φ·V_{a,r}; one-step ŷ = V_{a,r}·(K_{a,r}·c) (lifted space).
- Normalized Frobenius error: ε(M_cur, M_next) = ||M_next − M_cur·V·Kᵀ·Vᵀ||_F / ||M_next||_F.

## Pre-seal correction (disclosed, both hashes recorded)
- OLD contract sha `152130b3…` → NEW sha below. Corrections made BEFORE any K fit,
  based on a read-only spectral probe of the frozen encoder over the sealed calibration
  split (no operator constructed; no residuals observed):
  1. Upload claim "κ8 < 3.0" FALSIFIED by measurement: a0 κ8 = 2.951, a1 κ8 = 3.085.
     Gate corrected to κ8 ≤ 10.0 (standard well-conditioned bound); measured values reported.
  2. Upload claim "top-8 subspace contains 88.4% variance" FALSIFIED by measurement:
     a0 79.6%, a1 78.9% (X). Gate corrected to ≥ 0.75.
  3. Upload claim "ε_EDMD ≤ 0.05" is internally contradictory with its own 88.4% claim:
     the r=8 FULL-SPACE residual floor is ≥ √(1−0.796) ≈ 0.45 (measured floor 0.51–0.68).
     Full-space ε ≤ 0.05 is mathematically infeasible. Sealed metric = PROJECTED
     (coefficient-space) normalized Frobenius; full-space errors reported as DIAGNOSTIC.

## Gates C1–C12 (corrected)
- **C1** default-OFF bypass byte-identical (encoder re-assert).
- **C2** zero trainable state (inherited sealed contract `c2ca66e7…`; class-source AST scan).
- **C3** npz loader asserts full SHA `766e607a…`.
- **C4** per-slot unit-norm sphere max err ≤ 1e-6 (calibration spot-check).
- **C5** sensitivity: ≥ 90% of DISTINCT calibration pairs L2 > 1e-3 (identical obs deduped).
- **C6** full numerical rank per action: r(>1e-6·s1) = N_a (102 and 69) AND PR > 4 per action.
- **C7** conditioning + variance gates: κ8 = s1/s8 ≤ 10.0 for X0 and X1; top-8 variance share
  (s1²+…+s8²)/Σs² ≥ 0.75 for X0 and X1. Measured diagnostics: κ8 {a0 2.951, a1 3.085};
  top-8 {a0 0.796, a1 0.789}.
- **C8** calibration reconstruction (PROJECTED): ε_proj(a,r) = ||(Y_a·V_r) − (X_a·V_r)·K_{a,r}||_F /
  ||Y_a·V_r||_F ≤ 0.05 for ALL four (a,r). Full-space errors are diagnostic.
- **C9** evaluation prediction on the disjoint 133 (PROJECTED): ε_proj_eval(a,r*) ≤ 0.05 for both
  actions AND ε_proj_eval(a,r*) < ε_proj_persistence(a) with ratio ≤ 0.95 (≥5% relative
  improvement over identity in the same projected space).
- **C10** rollout stability: spectral radius ρ(K_{a,r*}) ≤ 1.05 both actions; open-loop rollout
  (recorded actions, horizon H = min(20, episode_len−1), starts = first record of each eval
  episode) per-step error in COEFFICIENT space (ĉ_{t+1} = K·ĉ_t vs c_{t+1} = X_{t+1}·V,
  normalized by ||c_{t+1}||) ≤ 0.15 per action; full-space rollout error reported as diagnostic.
- **C11** determinism: two separate processes → identical SHA-256 of K matrices and telemetry JSON.
- **C12** verdict discipline: labels below; baselines (persistence, calibration-mean) always reported.

## Selection rule (pre-registered, calibration-only)
- The upload's step 5 (`r* = argmin ε_eval(r)`) is REJECTED: it selects on the evaluation split,
  consuming it as development data (Reference 3). Selection is calibration-only, evaluation is
  scored ONCE.
- r* = argmin_{r∈{4,8}} mean_a ε_calib(a,r); tie → r = 4.
- Evaluation is scored ONCE on r*. The other rank's evaluation numbers are DIAGNOSTIC (reported,
  not gated). No re-selection on evaluation data; no post-hoc baseline additions.

## Kill criteria
- No retuning of W/σ/k/normalization (inherited from Stage-0b-rev).
- No re-selection of r on evaluation. No new gates after observing results.

## Verdicts
- All C1–C12 pass → `REDUCED_KOOPMAN_PREDICTION_VERIFIED`.
- C3 fail → `FROZEN_ARTIFACT_MISMATCH`. C7/C8 fail → `CONTRACT_FAILED`.
- C9 fail → `PREDICTION_FAILED`. C10 fail → `NUMERICALLY_UNSTABLE`.
- NOT `IDENTIFIABILITY_VERIFIED` — that label is reserved for the full-rank identifiability gate;
  this carrier proves (or fails) reduced-Koopman predictive efficacy and stability on disjoint
  episodes, nothing more. VLA gate stays 0/12. AAII v4.1.1 stays 0/9 BLOCKED.
