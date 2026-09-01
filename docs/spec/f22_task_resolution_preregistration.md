# Carrier F22 — Dynamic Affordance Sub-Goal Stepping & Metric-Realigned Task Resolution — Pre-Registration

- Directive: `Project_HENRI_F21.1_Post-Mortem_Audit___Task_Resolution_Directive.md`
- Directive SHA-256: `841c73a5b43058261ba31b0a17760f0674f8a3126e5a111ecb631f35666eaa49`
- Directive bytes/lines: 20,690 B / 235 lines
- Directive ID: `HENRI-DIR-2026-08-F21-1-POSTMORTEM-TASK-RESOLUTION`
- Evaluated commit: `5707e3b` (carrier/f21-1-vectorized-edmd) — local HEAD matches
- Governance: F21.1 verdict ratified (`F21_1_GATE_G2_FAILED`, seal `#6ec5eba3`, ledger 1,137 verified; receipt `d14d6269…` reconciled exactly: G1 0.3301 ms, G3 +0.0719, G4 1.1504 mis-attributed goal-distance, envs_solved 0/12, wall_s 0.595 → substrate loop had NO live arcade interaction)
- Branch: `carrier/f22-task-resolution`; prereg target: `docs/spec/f22_task_resolution_preregistration.md`
- Seal events: `F22_PREREG_SEALED`, `F22_GATES_VERDICT` (ledger 1,140+)

## Mechanism (verbatim from directive §3)

1. **Tier 1 — Offline Affordance Chain & Empirical Lie Matrix Compilation:**
   - `𝒲^(e) = ExtractGeodesicWaypoints(𝒟_bank^(e), stride=15)` — per-env waypoint chain from the verified trajectory bank, greedy geodesic sampling at uniform intervals Δθ = 0.35 rad, 4–6 intermediate waypoints + terminal goal.
   - `T_a = exp(SpectralCap(Logm(W_a^EDMD), π/32)) ∈ SO(64)` — F21.1 capped generators reused verbatim (`compile_generators_capped`, ω_bound = π/32 ≈ 0.0982).
2. **Tier 2 — Fast Batched Einsum Horizon Beam Search (K=8):**
   - `a* = argmax_a [ |⟨Ψ̂_{t+8}(a), Ψ_wp,k(t)⟩| − β Σ_{k=1..8} Δ_Sagnac(k) ]` — F21.1 batched `t_pow` unroll (one contraction), scored against the ACTIVE waypoint, not the distant terminal.
3. **Tier 3 — Dynamic Waypoint Advancement & Langevin Barrier Escape:**
   - `|⟨Ψ_{t+1}, Ψ_wp,k⟩| ≥ 0.60 → k ← k+1` (waypoint advance threshold).
   - Reset event → inject `T_active = 0.50` into action scores for 3 steps (Langevin barrier escape).
4. **G4 realigned:** `Δ_Sagnac = 1 − |⟨T_a Ψ_t, Ψ_axiom⟩|` (physical step-to-step operator coherence vs axiom reference; NO longer goal distance).
5. **Live interaction (G2 prerequisite):** real Arcade loop per F15 pattern — `arcade.make(name)` → `reset()` → `frame[0]` → flat → pad 4096 → PatchIngress → Ψ_t [64] → select → `game.step(action)` → terminal (`obs.state.name == GAME_OVER`) → reset; solved = `levels_completed` increase.

## Data & bounds (verbatim directive §5)

- Bank: `/root/f3-run/telemetry/f3_bank_capture_v2/trajectories_production_run_f3v2.npz` + `.jsonl` (pinned `9e3c01b4…`, N=1,536, 7 actions).
- 12 F15 `DEFAULT_ENVS` named envs × 150 steps = 1,800 steps; seed `20260922`; horizon K=8; ω_bound 0.0982; waypoint advance threshold 0.60; Langevin temp 0.50; device cuda.
- Receipt: `/tmp/henri_f22_resolution/f22_gates_receipt.json`; flag `HENRI_F22_RESOLUTION=1` (default-OFF fail-closed).

## Gates (directive §4)

| Gate | Criterion | Failure |
|---|---|---|
| PG1 | per-action in-sample recon `⟨exp(D_a*)Ψ_t, Ψ_{t+1}⟩` ≥ 0.85 (CAPPED generators) | PRE-FLIGHT KILL `F22_EDMD_FIT_COLLAPSE` |
| G1 | mean interactive step latency ≤ 2.0 ms | KILL K1 `F22_GATE_G1_FAILED` |
| G2 | ≥ 1 of 12 live envs solved end-to-end (levels completed ≥ 1) | KILL K2 `F22_GATE_G2_FAILED` |
| G3 | mean per-step directional waypoint valence Δν_wp ≥ +0.0200 | KILL K3 `F22_GATE_G3_FAILED` |
| G4 | mean single-pass Sagnac physical loss vs axiom ≤ 0.0500 | KILL K4 `F22_GATE_G4_FAILED` |

Verdict precedence: PG1 kill → G1 → G2 → G3 → G4; all-pass = `F22_PASS`. Receipt keys: `verdict, steps_done, resets, mean_latency_ms, sagnac_axiom_mean, mean_delta_nu_wp, waypoint_align_first, waypoint_align_last, waypoint_advances, langevin_escapes, per_action_recon, creeps, n_actions, seed, envs_solved, env_levels, wall_s, omega_bound, beta_sagnac, horizon, waypoint_advance_thresh, langevin_temp`.

## Disclosed deviations

1. Waypoint extraction = greedy geodesic sampling over bank states (`acos(|cos|)` geodesic angle, stride 15, Δθ = 0.35 rad, capped 4–6 waypoints) + F15 `resolve_trajectory_goal` terminal as the final waypoint. If < 2 waypoints survive, fallback `[first, terminal]` (disclosed; healthy banks expected ≥ 4).
2. β = 0.015 carried from F21.1 (directive §5 CLI omits β; formula shows β; optional `--beta-sagnac` default 0.015, §5 args remain verbatim).
3. Ψ_axiom = deterministic seeded unit reference on the D=64 verification substrate (directive's axiom baseplate is Zone C D=65,536; disclosed substitute, fixed across the run).
4. Per-k Sagnac penalty inside scoring = `1 − |cos(Ψ̂_k(a), axiom)|`; G4 telemetry = mean single-pass `1 − |cos(T_{a*} Ψ_t, axiom)|` for the selected action (physical operator coherence).
5. Langevin escape: score jitter `j ← j + sqrt(2·T_active)·randn(n_actions)` for 3 steps after each reset; T_active = 0.50.
6. Flag `HENRI_F22_RESOLUTION=1` (directive names no flag; F21/F21.1 pattern).
7. Verdict labels `F22_*` per directive §5 step 4 (`F22_GATES_VERDICT`).
8. Engine `arc_f22_resolution_engine.py`; contract tests `test_f22_resolution_engine.py`; PG1/compile helpers imported from the F21.1 engine (live code reuse).
9. G1 timed over the FULL interactive step (frame → ingress → unroll → select → game.step → alignment), stricter than F21.1's substrate-only timing.
10. PG1 threshold raised 0.70 → 0.85, G1 budget tightened 5.0 → 2.0 ms per directive §4.

## Kill experiment summary

Cheapest decisive probes, in order: (a) waypoint-chain extraction tests (≥ 4 unit waypoints, monotone geodesic progress, terminal = F15 goal); (b) advancement state machine (|cos| ≥ 0.60 → k+1); (c) Langevin escape (reset → 3 noisy steps → jitter decay); (d) PG1 ≥ 0.85 on capped generators (pre-flight kill); (e) live gauntlet G2 ≥ 1/12 with G3 ≥ +0.0200 and realigned G4 ≤ 0.05.
