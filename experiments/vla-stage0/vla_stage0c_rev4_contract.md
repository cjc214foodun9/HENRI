# System-1 Stage-0c-rev4 — r=8 Contractive Spectral Evaluation — PRE-REGISTRATION

**Date:** 2026-08-24 · **Reference 3 (gpt-5.6-sol) binding** · Status: SEALED BEFORE K CONSTRUCTION

## Prior verdicts (preserved, never relabeled)
- Stage-0c-rev (RFF, PR≥16 gate): `IDENTIFIABILITY_BLOCKED` (audit `5f44c3c8…`, commit `a7d621d`).
- Stage-0c-rev2 (r∈{4,8}): `CONTRACT_FAILED` at C8 (calib projected ε 0.145–0.181 > 0.05; commit `754ffb3`).
- Stage-0c-rev3 (r=16): `CONTRACT_FAILED` at C8 (calib projected ε 0.126/0.129 > 0.05); C9 PASS
  (SSR_eval 0.369 ≤ 0.40 on fresh 220); C10 FAIL (absolute rollout 0.555 > 0.35); commit `e3c3d27`.

## Upload (proposal artifact, audited)
- `Stage-0c-rev4_Architecture___Pre-Registration_Protocol.md`, 612 B, SHA-256
  `458d07761b3a7741badfee3473fcfd5155ae96d441c001c38f6b26b603ad0efd`. 16-line diagram:
  UNSEEN x(t)[4D] → frozen RFF encoder [6144D] → parsimonious truncation V8 (κ8 2.95, VarShare 79.6%)
  → EDMD + spectral projection ρ(K8) ≤ 1.0000 (Contractive) → ONE-STEP SKILL (SSR_eval ≤ 0.40,
  "Passed: SSR = 0.3688") + 5-STEP ROLLOUT (SSR_rollout ≤ 0.80, "Unblocked by ρ ≤ 1.0").
- User protocol: load npz `766e607a…`; construct X0, X1, X5 per action from calib 171 + eval 220;
  compute K8^(a0), K8^(a1); apply contractive projection K̃8 if ρ > 1.0; evaluate C8, C9, C10
  (SSR_eval ≤ 0.40, SSR_rollout,5 ≤ 0.80); emit sealed audit JSON + governance block.

## Read-only probe claims (OBSERVED in rev2/rev3 audits, reused here as diagnostics)
- κ8 = 2.951 (a0) / 3.085 (a1); top-8 share 0.796 (a0) / 0.789 (a1) — matches upload 2.95/79.6% (a0).
- Rev3 r=16 SSR_eval = 0.3688 on the 220 eval corpus — the upload cites this as "Passed"; rev4
  measures SSR_eval at **r=8** fresh (new measurement, same split — see eval label below).

## Data
- **Calibration (fitting):** established 171-record split (lexicographic; ids 101..404; manifest `54b7350a…`).
- **Evaluation:** the rev3 220-record corpus (seeds 2101–3010, manifest `f0c9a7624f26bf70…`,
  raw-obs overlap vs calibration = 0). **Label: `CONDITIONAL_REUSED_EVAL`** — this split was
  evaluated in rev3 (one-step SSR and 5-step rollout); it is disjoint from calibration but is
  adaptive-development data, not fresh. No NEW corpus is built in this carrier (user protocol
  names the 220 split; the label records the exposure).

## Frozen input
- `vla_stage0b_rev_params.npz`; loader + script ASSERT full SHA-256
  `766e607ad0bc739ea0a139172dd34e16d01a268cca80e990af5aab01006cfcd7`. No retuning.

## Operators (fixed; no selection on evaluation)
- **r = 8 FIXED.** Per-action bases V8^(a) = top-8 right singular vectors of calibration X^(a)
  (raw SVD, no regularization). K8^(a) = lstsq(XV8^(a), YV8^(a), rcond=None). Never shared.
- **Contractive spectral projection (pre-registered rule):** compute eigendecomposition
  K = U·Λ·U⁻¹ (complex). If ρ = max|λ| > 1.0: λ̃_i = λ_i / max(1, |λ_i|), then
  K̃ = Re(U·diag(λ̃)·U⁻¹). If ρ ≤ 1.0: K̃ = K (identity arm, recorded). If eigendecomposition is
  ill-conditioned (cond(U) > 1e8), emit `BLOCKED_NUMERICAL`, no verdict.
- Implicit projection: P8^(a) = V8^(a)V8^(a)ᵀ applied as (x@V)@V.T. No dense 6144×6144 materialized.

## Metrics (all reported separately)
- **X0, X1, X5 matrices** per action: X0_a = window-start lifted states, X1_a = one-step targets,
  X5_a = 5-step targets (windows of length 6 over the 220 eval corpus; all windows, mixed actions).
- full-space one-step error (DIAGNOSTIC; floor ~0.45 at top8 0.79 makes gates infeasible).
- projected one-step ε_eval (coefficient space).
- persistence-1 baseline; calibration-mean baseline.
- **SSR_eval = mean_a [ε_eval(a) / ε_persist1(a)]** (projected).
- **5-step open-loop rollout:** for each window, x̂ = x0; for h in 0..4: a_h = action at step h,
  x̂ = ((x̂ @ V8^(a_h)) @ K̃^(a_h)) @ V8^(a_h).T (full-state re-projection on action switches).
  Error at horizon 5 (projected onto a_last = action at step 4): ||(x̂−x5)·V8^(a_last)|| /
  ||x5·V8^(a_last)||. Persistence-5 baseline: ||(x5−x0)·V8^(a_last)|| / ||x5·V8^(a_last)||.
  **SSR_rollout,5 = mean_a [ε_roll5(a) / ε_persist5(a)]**.
- ρ_raw(K8^(a)) and ρ_proj(K̃8^(a)) for each action.

## Gates and verdict chain (failure precedence)
- **C8** truncation + contraction contract: κ8 ≤ 10.0 AND top-8 share ≥ 0.75 (both actions) AND
  ρ(K̃8^(a)) ≤ 1.0000 (both actions, after projection rule).
- **C9** SSR_eval ≤ 0.40 (projected, per-action ratio aggregated over actions).
- **C10** SSR_rollout,5 ≤ 0.80 (aggregated over actions, contractive operator).
- **C11** determinism: telemetry JSON + operators NPZ byte-identical SHA-256 across ≥ 2 processes.
- **C12** baselines reported (persistence-1, persistence-5, calibration-mean; calib and eval).
- Verdict: C8–C11 all pass → `CONTRACTIVE_SPECTRAL_VERIFIED`; else `CONTRACT_FAILED` at first
  failing gate. Kill criteria: no post-result tuning of r, projection rule, or thresholds.

## Boundaries
- CartPole dynamics result only. **VLA 0/12; AAII v4.1.1 0/9 BLOCKED.** No SOTA claim.
- The "perfect architecture" question is addressed separately in the architecture decision record
  (roadmap audit `6c534e93…` disposition); this carrier is the smallest decisive test of the
  contractive-projection hypothesis, per the uploaded protocol.
