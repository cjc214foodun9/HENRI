# Carrier F15 — Interactive Trajectory Goal Steering — Pre-Registration

- Directive: `Project_HENRI__Carrier_F15_Interactive_Trajectory_Goal_Steering_Directive.md`
- Directive SHA-256: `e2e3d1efa22e26909b504aee173bf1d6b3891311bf4e2a219e8ec9ec02146808` (14,818 B, 193 lines)
- Directive ID: `HENRI-DIR-2026-08-F15-TRAJECTORY-GOAL-STEERING-ORDER`
- Branch: `carrier/f15-trajectory-goal` (base: F14 HEAD `6db04c1`)
- Governance: ledger target record 1,093+; prior seal `F14_1_GATES_VERDICT #f80943cd…` RATIFIED by directive (Option 2 selected)

## 1. Mechanism

Trajectory-grounded goal steering (per directive §2), replacing F14's demo-manifest goal source:

1. **Goal extraction** (per env, from the F3 v2 verified trajectory bank):
   `Ψ_goal = Normalize(Ψ(s_{t*}^{(e)}))`, `t* = env_indices[-1]` = terminal row of the env's longest non-reset sequence (`env_indices` from JSONL rows for `env == env_id`, assert `len ≥ 30`).
2. **Pre-Gate PG1**: `|⟨Ψ_0, Ψ_goal⟩| ≤ 0.90` across all 12 cohort envs. Violation → PRE-FLIGHT KILL (fail-closed, zero steps). PRE-COMPUTED on the live bank (OBSERVED 2026-09-01): max overlap 0.4508 (ft09) — all 12 ≤ 0.90.
3. **Tier 1 — Slerp waypoint**: `Ψ_wp(τ) = [sin((1−τ)θ)·Ψ_t + sin(τθ)·Ψ_goal]/sin θ`, `θ = arccos(⟨Ψ_t, Ψ_goal⟩)` (signed; F14 D2), `τ = 0.25`. `sin θ < 1e-8` → linear fallback.
4. **Tier 2 — vectorized beam search** (K=8, beam 8): `J(a_{1:8}) = |⟨Ψ̂_{t+8}, Ψ_wp⟩| − α Σ_k Δ_Sagnac(k)`, `α = 0.05` (F13 C11-verified vectorization, ported from F14).
5. **Tier 3 — commit first action**; signed valence `Δν_t = |⟨Ψ_{t+1}, Ψ_wp⟩| − |⟨Ψ_t, Ψ_wp⟩|`.
6. **Tier 4 — Hebbian creep**: `M_a ← Normalize(M_a + η_fast·Δν_t·Ψ_t)`, `η = 0.05`; zero-valence guard (F13/F14 D4).

## 2. Gates (pre-registered, from directive §3)

| Gate | Criterion | Failure action |
|---|---|---|
| PG1 | `max_env |⟨Ψ_0, Ψ_goal⟩| ≤ 0.90` | PRE-FLIGHT KILL (degenerate trajectory goal) |
| G1 | 1,800 steps / 12 envs, latency ≤ 5.0 ms/step | KILL K1 |
| G2 | ≥ 1 of 12 solved (Score > 0.0%) | KILL K2 |
| G3 | mean Δν_goal ≥ +0.0200 | KILL K3 |
| G4 | mean Sagnac (K=8 horizon, roll[0,0] vs goal) ≤ 0.050 | KILL K4 |

## 3. Pre-registered deviations (disclosed, not silent)

- **D1 (goal dimension bridge)**: bank waves are D=65,536 (`psi` rows: [8192,8] block waves, first-row norm 90.51, terminal rows unit); the verified steering engine substrate is D=64 (F14 D3; dense expD [8,65536,65536] = 137 GB infeasible, F10 ban). Bridge: deterministic block mean-pool [65,536]→[4096]→PatchIngress(d=64)→[64], matching the live psi0 boundary. No learning, no tunable params.
- **D2 (live bank schema vs directive pseudocode)**: npz key is `psi` (directive: `trajectories`); JSONL field is `env` (directive: `env_id`). OBSERVED on live bank (`trajectories_production_run_f3v2_manifest.json` schema `henri.arc-trajectory-bank.v1`, npz sha `9e3c01b4…` pinned). Also: "longest non-reset sequence" implemented as the env's contiguous row block ending at the last row (per directive pseudocode `env_indices[-1]`).
- **D3 (PG1 pre-computation)**: PG1 overlap pre-computed on the REAL bank (OBSERVED: max 0.4508) — the directive's §1.1 claim `|⟨Ψ_0, Ψ_goal⟩| ≪ 0.9000` is confirmed; runtime still asserts per env (fail-closed if any > 0.90).
- **D4 (zero-valence creep guard)**: carried from F13/F14 (determinism contract).
- **D5 (G4 instrument)**: identical to F14 (`sagnac_delta(roll[0,0], goal)` per step) for cross-carrier comparability.

## 4. Execution order

1. TDD: `tests/contract/test_f15_trajectory_engine.py` (C1–C12: bank schema/load, terminal extraction, PG1 fail-closed degenerate bank, PG1 pass, schema errors, env-field parsing, beam equivalence, valence sign, zero-valence guard, no-bank fail-closed, 12-env PG1 pre-flight, determinism).
2. Implement `experiments/verification/arc_f15_trajectory_engine.py` (flag `HENRI_F15_TRAJECTORY=1` fail-closed; CLI per directive command with `--trajectory-bank`/`--trajectory-jsonl`).
3. Local suite + full regression from repo root.
4. Commit + push; remote detached-worktree CUDA verify @ exact SHA.
5. Live gauntlet (directive command): 12 envs × 150 steps = 1,800, seed `20260914`, K=8, τ=0.25, η=0.05, bank `/root/f3-run/telemetry/f3_bank_capture_v2/`.
6. Seal `F15_GATES_VERDICT`; deliver scorecard + lessons.

## 5. Kill experiment / cheapest falsification

If PG1 fires on ≥ 1 live env (bank terminal goal degenerate vs live first frame) → trajectory goal extraction is degenerate on this substrate (KILL, pre-flight, zero steps). If engagement shows positive goal separation (PG1 pass) but Δν_goal ≤ 0 on the live loop → the steering primitive (beam + waypoint) is falsified against the trajectory goal, same class as F13's mirror-steering finding.

## 6. Falsifiable claim

"HYPOTHESIS: a trajectory-grounded terminal-state goal (F3 v2 bank, per env) generates positive goal-convergence valence (G3 ≥ +0.02) and enables task resolution (G2 ≥ 1) under the F13-verified vectorized beam on the 12-env live gauntlet." PG1 is pre-verified on the real bank; G1–G4 measured live on vast-5090.
