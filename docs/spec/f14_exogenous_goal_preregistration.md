# Carrier F14 — Exogenous Goal Ingress & Non-Degenerate Waypoint Steering — Pre-Registration

- Directive: `Project_HENRI_F13_Post-Mortem_Audit___Exogenous_Goal_Ingress_Directive.md`
- Directive SHA-256: `bda4a5060e80290b967096f5990a4d6bd4fb5017f2b291078cb6ba15051ea507` (21,174 B, 240 lines)
- Directive ID: `HENRI-DIR-2026-08-F13-POSTMORTEM-EXOGENOUS-GOAL`
- Branch: `carrier/f14-exogenous-goal` (base: F13 HEAD `87623e0`)
- Governance: ledger target record 1,084+; prior seal `F13_GATES_VERDICT #e144bd0e…` RATIFIED by directive

## 1. Mechanism

Hierarchical goal steering with an EXOGENOUS goal source, replacing F13's self-referential first-frame goal (D1):

1. **Exogenous goal synthesis** (per episode, from demonstration pairs `(X_i, Y_i)`):
   `W_task = StiefelRetract((1/M) Σ_i Ψ_Y,i ⊛ Ψ_X,i†)`, `Ψ_goal = Normalize(W_task @ Ψ_0)`.
   Substrate: flat real `[64]` (D=64 verification substrate per directive header); `⊛` = rank-1 outer-product binding; Stiefel retract = SVD polar `U V^T` (orthogonal).
2. **Pre-Gate PG1**: `|⟨Ψ_0, Ψ_goal⟩| ≤ 0.90` across cohort. Violation → PRE-FLIGHT KILL (fail-closed, zero steps).
3. **Tier 1 — Slerp waypoint**: `Ψ_wp(τ) = [sin((1−τ)θ)·Ψ_t + sin(τθ)·Ψ_goal] / sin θ`, `θ = arccos(⟨Ψ_t, Ψ_goal⟩)` (signed; see D2), `τ = 0.25`. sin θ < 1e-8 → linear fallback (norm-preserved) to avoid NaN.
4. **Tier 2 — vectorized beam search** (depth K=8, beam 8): `J(a_1:8) = |⟨Ψ̂_{t+8}, Ψ_wp⟩| − α Σ_k Δ_Sagnac(k)`, `α = 0.05`. Single batched einsum + topk per depth (F13 C11-verified vectorization).
5. **Tier 3 — commit first action**; signed valence `Δν_t = |⟨Ψ_{t+1}, Ψ_wp⟩| − |⟨Ψ_t, Ψ_wp⟩|`.
6. **Tier 4 — Hebbian creep**: `M_a ← Normalize(M_a + η·Δν_t·Ψ_t)`, `η = 0.05`; zero-valence guard (skip when Δν == 0, literal `M + 0·Ψ = M`, determinism contract).

## 2. Gates (pre-registered, from directive §4)

| Gate | Criterion | Failure action |
|---|---|---|
| PG1 | `max_env |⟨Ψ_0, Ψ_goal⟩| ≤ 0.90` | PRE-FLIGHT KILL (degenerate goal ingress) |
| G1 | 1,800 steps / 12 envs, latency ≤ 5.0 ms/step | KILL K1 |
| G2 | ≥ 1 of 12 solved (Score > 0.0%) | KILL K2 |
| G3 | mean Δν_goal ≥ +0.0200 | KILL K3 (tightened vs F13's > 0.0) |
| G4 | mean Sagnac (K=8 horizon, roll[0,0] vs goal) ≤ 0.050 | KILL K4 |

## 3. Pre-registered deviations (disclosed, not silent)

- **D1 (BLOCKED_NO_PUBLIC_DEMOS)**: the directive's goal source requires demonstration pairs `(X_i, Y_i)`. OBSERVED on the live substrate: `LocalEnvironmentWrapper`, inner game object, `arc_agi` package, and all 12 env files expose ZERO demo/train/examples surfaces; the only sanctioned channel is `resolve_demos(manifest)` (`arc_public_ingress.py`) via `HENRI_ARC_PUBLIC_INGRESS_MANIFEST`; **no provenance-pinned manifest exists in repo or Drive inbox**. Per sealed governance (never fabricate demos; private levels = hidden targets), the LIVE gauntlet pre-flights `BLOCKED_NO_PUBLIC_DEMOS` (fail-closed, zero steps) when no manifest is supplied — honest negative, same class as F13's TimesFM-3 §4 block. Mechanism engagement is proven by a SYNTHETIC-manifest smoke (plumbing-only; NO G2/G3/G4 claims from synthetic data).
- **D2 (Slerp signed θ)**: the directive's θ expression `arccos(|⟨·⟩|)` folds anti-aligned pairs and breaks constant-angular-velocity geodesics, contradicting its own "exact geodesic Slerp" language. Implemented θ = arccos(⟨a,b⟩) (signed, standard geodesic); tests verify norm preservation, endpoints, and geodesic monotone alignment. PG1 still uses |cos| per directive.
- **D3 (W_task scale)**: D=64 flat substrate (directive header "D=64 Verification Substrate"); dense [64,64] functor (16 KB fp32). Production D=65,536 factorized functor is out of scope for this bounded carrier (future work, per F10 ban on dense [D,D]).
- **D4 (zero-valence creep guard)**: carried from F13 (determinism contract; literal formula).
- **D5 (G4 instrument)**: identical to F13 (`sagnac_delta(roll[0,0], goal)` per step) for cross-carrier comparability (F10 0.0269 → F11 0.0157 → F12 0.0253 → F13 0.0390).

## 4. Execution order

1. TDD: `tests/contract/test_f14_exogenous_engine.py` (contracts C1–C11: slerp geometry, functor recovery + orthogonality, PG1 fail-closed, vectorized≡naive beam equivalence, valence sign, zero-valence guard, batched horizon boundary, determinism, fail-closed no-manifest, synthetic-manifest engagement).
2. Implement `experiments/verification/arc_f14_exogenous_engine.py` (flag `HENRI_F14_EXOGENOUS=1` fail-closed; CLI per directive command + optional `--ingress-manifest`).
3. Local suite + full regression from repo root.
4. Commit + push; remote detached-worktree CUDA verify @ exact SHA.
5. Live gauntlet (directive command): no manifest → pre-flight `BLOCKED_NO_PUBLIC_DEMOS` receipt; synthetic-manifest smoke (2 envs × 30 steps, engagement telemetry only).
6. Seal `F14_GATES_VERDICT`; deliver scorecard + lessons.

## 5. Kill experiment / cheapest falsification

If, WITH a manifest (future carrier), PG1 fires on ≥ 1 cohort env → exogenous goal ingress itself is degenerate on that substrate (KILL). If engagement smoke shows goal synthesis non-degenerate (PG1 pass) but beam search selects actions with Δν_goal ≤ 0 → the steering primitive (not the goal source) is falsified. The synthetic smoke pre-registers these verdict classes.

## 6. Falsifiable claim

"HYPOTHESIS: an exogenous, non-degenerate goal source (functor-projected terminal target) generates positive goal-convergence valence (G3 ≥ +0.02) and enables task resolution (G2 ≥ 1) under the F13-verified vectorized beam." Live verification BLOCKED by D1 until a provenance-pinned manifest exists; the smoke tests the mechanism, not the task outcome.
