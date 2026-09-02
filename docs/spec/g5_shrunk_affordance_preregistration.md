# Carrier G5 Pre-Registration — Empirical Bayes Shrinkage & Sample-Gated Dual-Subspace Affordance

**Directive:** `Carrier_G5_Master_Directive___G4_Post-Mortem.md` — `HENRI-DIR-2026-09-V3-CARRIER-G5-SHRINKAGE-DIRECTIVE`
SHA-256 `fc2dd03ac8e56dcec83b3a1c79f7dbd44f2765000c0315c26a69e6a46e21613c`, 17,334 B, 315 lines.
Sealed `G5_PACKET_INGESTED #ee5cbe97` (ledger @1,178).
Branch: `feat/carrier-g5-shrunk-affordance`. Parent: `G4_GATES_VERDICT #9cbf1ed1` (24th sealed falsification).

## Mechanism

Lever 1 — Empirical Bayes shrinkage of per-block variance:

- s²_m(a) = Var_t(‖Ψ_{t+1}^(m) − Ψ_t^(m)‖) over the action's transitions (CENTERED, G4 `per_block_displacement_variance`; directive §1.1 statistic — the directive's own code block uses an uncentered mean-of-squares; prereg locks the centered form per §1.1).
- σ²_prior(m) = (1/|A|) Σ_a s²_m(a) — cross-action pooled spatial prior.
- α_a = ν₀ / (ν₀ + N_a·d), ν₀ = 64, d = 8 (monotone decreasing in N_a — C2). α(a6, N=10) = 64/144 ≈ 0.444; α(a0, N=56) = 64/512 = 0.125.
- σ̂²_m(a) = (1 − α_a)·s²_m(a) + α_a·σ²_prior(m).
- top-k = argmax_k σ̂²_m(a) over m ∈ {1..8192}, k = 64 — PER-ACTION invariant support (directive §3.2 text; the shipped code's per-sample top-k is rejected).

Lever 2 — Sample-gated dual-subspace routing (PG3):

- If N_moving(a) < 40 → **D=64 bridge arm**: `_bridge_to_d64_batch` (VERIFIED origin `arc_f21_edmd_engine.py:79`; G4's import from `arc_f21_1_vectorized_engine` is a latent dead import — f21_1 re-exports only `PatchIngress`; G4 never exercised it). Bridge = block-mean → unit-normalize ×64 → PatchIngress → [N, 64] → view [N, 8, 8]. Fit per-block 8×8 ridge transitions on MOVING rows; score = mean quadratic residual over the 8 bridge blocks (C1 within the arm).
- Else → **shrunken top-k arm**: per-(action, topk-block) 8×8 ridge transitions on MOVING rows; score = mean quadratic residual over the SAME shrunken top-k support (C1 within the arm).
- Routing rule applies to EVERY action: **a4 (N_moving=15) and a6 (N_moving=10) route to the bridge**. a4's top-k arm scored 1.0 in G4; its bridge-arm AUC is unknown → a real falsifiable risk, per directive targets a0–a4 ≥ 0.95.

Pi_a = sigmoid((θ_a − r_a)/τ_a); θ/τ calibrated per arm from the action's moving/blocked residual distributions (G4 `calibrate_theta_tau`).

## Gates

| Gate | Metric | Threshold | Failure action |
|---|---|---|---|
| PG1 | min_action_auc (stratified subset N=128) | ≥ 0.8800 | HALT_IMMEDIATELY_SEAL_FALSIFICATION |
| PG1a | per-action subset AUC targets | a0–a4 ≥ 0.9500; a5 ≥ 0.8800; a6 ≥ 0.8800 | HALT_IMMEDIATELY_SEAL_FALSIFICATION |
| PG2 | flat norm drift ‖‖Ψ‖−1‖∞ | ≤ 1e-6 | HALT_IMMEDIATELY_SEAL_FALSIFICATION |
| PG3 | routing + top-k CPU == CUDA tensor identity | exact | HALT_IMMEDIATELY_SEAL_FALSIFICATION |

Full-bank AUC and per-route (bridge vs shrunken top-k) AUC recorded as DIAGNOSTICS, not gates.

## Execution contracts

- C1_FUNCTIONAL_HOMOLOGY: within each arm, the score evaluates the exact functional minimized during fit (mean quadratic residual over the arm's support).
- C2_SHRINKAGE_STABILITY: α_a monotonically decreases with N_a; σ̂² is a convex combination; α ∈ [0,1].
- C3_G3_W0_GATEWAY: a PG1 pass flags WavePacketPathSearch as an eligible candidate planner (W0) — NOT wired in this carrier.

## Bounds

- Bank: `trajectories_production_run_f3v2.npz` (sha `9e3c01b4…`, pinned; same 12 envs × 150 as F22/F23/G1–G4).
- Seed 20260928. k=64. Ridge 1e-2. τ_stall 0.90. ν₀=64. sample_threshold 40. bridge_blocks 8.
- Verdicts: `G5_AFFORDANCE_FIT_COLLAPSE` (any PG1/PG1a/PG2/PG3 violation; 0 live steps) or the carried live-loop verdicts (G5_GATE_G1/G2/G3/G4_FAILED / G5_ALIGNED_AFFORDANCE_VERIFIED) only after PG1 passes.
- Flag: `HENRI_G5_SHRUNK_AFFORDANCE=1` (default-OFF fail-closed).
- Kill experiment: a6 (10 moving) must reach ≥ 0.88 AUC via the bridge arm; a4 (15 moving) must reach ≥ 0.95 via the bridge arm. Either failure = falsification.
