# Carrier F21.1 — Vectorized Batched Horizon & Spectral-Capped EDMD — Pre-Registration

- Directive: `Project_HENRI_F21_Post-Mortem_Audit___Vectorized_EDMD_Optimization_Directive.md`
- Directive SHA-256: `5cac800b68f37009a9141eca6b9aba1b1016721abf1d68348d329f5fa3c175df`
- Directive bytes/lines: 20,586 B / 243 lines
- Directive ID: `HENRI-DIR-2026-08-F21-POSTMORTEM-VECTORIZED-EDMD`
- Evaluated commit: `09ebc31` (carrier/f21-edmd-generators) — local HEAD matches
- Governance: F21 verdict ratified (`F21_GATE_G1_FAILED`, seal `#22f19872`, ledger 1,131; live ledger now 1,134 verified, head `9288be13`)
- Branch: `carrier/f21-1-vectorized-edmd`; prereg target: `docs/spec/f21_1_vectorized_edmd_preregistration.md`
- Seal events: `F21_1_PREREG_SEALED`, `F21_1_GATES_VERDICT` (ledger 1,132+)

## Mechanism (verbatim from directive §1/§3)

1. Koopman per action: `K_a = (Y_aᵀX_a)(X_aᵀX_a + λI)⁻¹` (λ = 1e-3, normal-equation form, as F21).
2. Stiefel retraction: `W_a = U_a V_aᵀ ∈ SO(D)` from SVD(K_a).
3. Skew principal log: `D_a_raw = ½(Logm(W_a) − Logm(W_a)ᵀ)`.
4. **Spectral cap**: `D_a* = D_a_raw · min(1.0, ω_bound / σ_max(D_a_raw))`, `ω_bound = π/32 ≈ 0.0982 rad/step` (directive diagram §1 literal scalar-rescale form; equivalent to the §1.1 eig-projection bound for skew matrices — max angular frequency ≤ ω_bound, so total unroll phase ≤ K·ω_bound = π/4).
5. Pre-computed transitions: `T_a = exp(D_a*) ∈ SO(64)`; powers `T_pow[a,k] = T_aᵏ` for k=1..K compiled offline into `[n_actions, K, D, D]`.
6. Runtime batched unroll: `Ψ_horizon[a,k] = einsum('akde,de->ak', T_pow, Ψ_t)` — ONE batched contraction for all actions × all K steps (F21: 56 sequential kernel launches → F21.1: 1).
7. Selection: `a* = argmax_a [ |⟨Ψ_{t+K}(a), Ψ_goal⟩| − β Σ_k Δ_Sagnac(k,a) ]`, `β = 0.015` (calibrated regulator per directive §3.1.3).
8. Plasticity unchanged: `M_{a_t} ← normalize(M_{a_t} + η_fast·max(0, Δν_t)·Ψ_t)`; `Δν_t = |⟨Ψ_{t+1}, Ψ_goal⟩| − |⟨Ψ_t, Ψ_goal⟩|`.

## Data & bounds (verbatim directive §5)

- Bank: `/root/f3-run/telemetry/f3_bank_capture_v2/trajectories_production_run_f3v2.npz` + `.jsonl` (pinned `9e3c01b4…`, N=1,536, 7 actions — F21 disclosed deviation carries forward).
- 12 F15 `DEFAULT_ENVS` named envs × 150 steps = 1,800 steps; seed `20260921`; horizon K=8; ω_bound 0.0982; β 0.015; device cuda.
- Receipt: `/tmp/henri_f21_1_vectorized/f21_1_gates_receipt.json`; flag `HENRI_F21_1_VECTORIZED=1` (default-OFF fail-closed).

## Gates (directive §4)

| Gate | Criterion | Failure |
|---|---|---|
| PG1 | per-action in-sample recon `⟨exp(D_a*)Ψ_t, Ψ_{t+1}⟩` ≥ 0.70 (measured on CAPPED generators) | PRE-FLIGHT KILL `F21_1_EDMD_FIT_COLLAPSE` (spectral projection distortion) |
| G1 | mean step latency ≤ 5.0 ms | KILL K1 `F21_1_GATE_G1_FAILED` |
| G2 | ≥ 1 of 12 envs solved (score > 0) | KILL K2 `F21_1_GATE_G2_FAILED` |
| G3 | mean Δν_goal ≥ +0.0200 | KILL K3 `F21_1_GATE_G3_FAILED` |
| G4 | mean single-pass K=8 Sagnac ≤ 0.0500 | KILL K4 `F21_1_GATE_G4_FAILED` |

Verdict precedence: PG1 kill → G1 → G2 → G3 → G4; all-pass = `F21_1_PASS`. Receipt keys: `verdict, steps_done, mean_latency_ms, sagnac_raw_mean, mean_delta_nu_goal, goal_align_first, goal_align_last, per_action_recon, creeps, n_actions, seed, envs_solved, wall_s, omega_bound, beta_sagnac, horizon`.

## Disclosed deviations

1. `n_actions = 7` from the live bank (directive's a∈{0…7} example); 7 pre-verified.
2. Spectral cap implemented as scalar rescale (diagram literal); σ_max from `torch.linalg.svdvals`, applied to the skew form; equivalence to eig-projection asserted in tests.
3. PG1 measured on capped generators (directive §4 literal) — cap may lower recon vs F21's 0.7156.
4. Batched unroll uses precomputed matrix powers + one einsum (directive §1.2's `einsum('b...d,kde->b...e')` adapted to the [n_actions,K,D,D] power stack).
5. λ = 1e-3 carried from F21 (directive leaves λ unvalued).
6. Goals per env via F15 `resolve_trajectory_goal` from the pinned bank; fallback seeded random goal (F21 behavior) only if resolution fails.
7. Latency timed over the full per-step block (step_once + exp apply + alignment), matching F21's G1 measure.
8. Verdict labels use `F21_1_*` per directive §5 step 4 (`F21_1_GATES_VERDICT`).
9. Engine file `arc_f21_1_vectorized_engine.py`; contract tests `test_f21_1_vectorized_engine.py`; PG1/verdict helpers imported from the F21 engine (live code reuse, not duplication).

## Kill experiment summary

Cheapest decisive probes, in order: (a) spectral-cap unit tests (σ_max ≤ ω_bound on healthy banks with large raw spectra; identity scaling below bound); (b) einsum-vs-loop equivalence on random ψ (max diff ≤ 1e-5); (c) PG1 on capped generators ≥ 0.70 (pre-flight kill); (d) live gauntlet G1 ≤ 5 ms (vectorization) and G4 ≤ 0.05 (cap) with G3 ≥ +0.0200 as the steering kill.
