# Carrier P1 Pre-Registration: Goal-Grounded Policy Steering

**Document ID:** `HENRI-SPEC-2026-09-V3-CARRIER-P1-POLICY-GROUNDING` (prereg)
**Packet:** `Carrier_P1_SpecContract___Alignment_Probe.md` (SHA-256 `06e667c33134f82924c0d9500dfa8c8ee8ab5c1e2dd6484478296b4161fd3989`, 18,819 B / 388 lines)
**Carrier:** `P1_GOAL_GROUNDED_POLICY_STEERING`
**Branch:** `feat/carrier-p1-policy-grounding`
**Base commit:** `4c71d4d` (G7 branch tip); probe commit `7c52a77`; surgical-patch commit (this carrier).
**Causal parent:** G7 verdict `G7_AFFORDANCE_FIT_COLLAPSE` `#25e96e09` @1,198 (27th falsification; representation fit SOLVED twice, live action→outcome coupling UNSOLVED: 1,800 steps, 0 solved, 0 waypoint advances, mean_delta_nu_wp 0.0).
**Seed:** `20260930` (identical to G7 launch #2 — PG1 stratified-subset draws stay comparable).
**Bank alignment prerequisite (sealed @1,205 `da597d5f`):** `probe_trajectory_bank_alignment` on the live bank — aligned=true, 1,536 rows = 1,536 lines, 12 envs contiguous, terminals exact, misalignments 0, terminal discrepancies 0. Probe file SHA `0e6d268c…`, commit `7c52a77`.

## Mechanism (packet §1, eq. 2.2–3.2; live-code reconciled)

G7 live policy scored `j(a) = align · (π_aff(a))^H` with `align` computed over an action-free D=64 rollout — identical for every action → argmax degenerated to the affordance prior → 0 solved / Δν 0.0.

P1 replaces the action-blind align with the **action-conditioned potential drop**, computed with the SAME route-aware operators that define the affordance residual (C1 homology preserved):

- Potential: `V(ψ) = 1 − |⟨ψ_norm, g_norm⟩|²`, `ΔV(a) = V(ψ_t) − V(T_a ψ_t)` = `|⟨T_a ψ_t, g⟩|² − |⟨ψ_t, g⟩|²`
- Score: `j(a) = (clamp(ΔV(a), −1, 1) + 1) · (π_a)^H` (π_a = G7 calibrated `exp(−r_a/τ_a)`)
- Topk/dense arm: candidate from per-block `T_m` on the action's top-k full-wave blocks; ΔV measured against the **full-wave terminal goal** `g_full [8192,8]` from the bank.
- Bridge/sparse arm: candidate from bridge `T_m` on the D=64 state; ΔV measured against the D=64 waypoint.
- Full-wave goals: `build_p1_full_goals` from bank rows (row == jsonl line, per-env last line), bound per env via the guarded `p1_bind_env_goal` hook in `run_gauntlet` (no-op when absent → G4–G7 default path byte-identical).

**Default-OFF:** `HENRI_P1_GOAL_STEERING=1` + inherited `HENRI_G7_CALIBRATED_AFFORDANCE=1`. P1 module imported lazily ONLY under the flag; flag absent → G7 engine, no P1 import, no P1 code path (differential proof in tests).

## Bounds (frozen)

| Parameter | Value |
|---|---|
| Envs | 12 (ar25, sc25, tr87, cd82, lp85, wa30, ft09, g50t, sk48, bp35, ka59, sb26 — same ids as G7 launch #2) |
| Steps | 150 per env → 1,800 total |
| Bank | `/root/f3-run/telemetry/f3_bank_capture_v2/trajectories_production_run_f3v2.{npz,jsonl}` |
| Output | `/tmp/henri_p1_goal_steering/` (fresh per launch) |
| CLI | G7 engine entrypoint (`arc_g7_calibrated_engine.py`) with P1 env vars; flags: `--device cuda --steps-per-env 150 --seed 20260930 --horizon 8 --omega-bound 0.0982 --waypoint-advance-thresh 0.60 --tau-stall 0.90 --top-k 64 --ridge 0.01 --nu0 64.0 --sparse-threshold 20 --dense-threshold 40 --tau-lo 0.05 --tau-hi 2.0` |

## Gates

**Pre-flight (inherited from G7 main(), MUST pass before any live step; any failure → `G7_AFFORDANCE_FIT_COLLAPSE`, exit 1):**
- PG1 global min_action_auc_subset ≥ 0.9000; PG1a a0–a4 ≥ 0.9500, a5/a6 ≥ 0.8800 (N=256 stratified subset)
- PG2 norm drift ≤ 1e-6; PG3 piecewise routing + top-k + τ_a CPU==CUDA identity; C2 dense α == 0.0 exactly

**Live (P1 verdict vocabulary, `_decide_verdict` order = fail-closed precedence):**
| Gate | Metric | Threshold | Failure action |
|---|---|---|---|
| LG1 | mean Δν (goal-cosine progress on observed trajectory) | ≥ 0.0500 | `P1_GATE_LG1_STAGNATION` → seal falsification |
| LG2 | solved levels | ≥ 1 (of 12 envs) | `P1_GATE_LG2_SOLVED_FAILED` → seal falsification |
| LG3 | on-device kernel latency, CUDA-event scope: `score_all_actions` local path only (NOT the remote-arcade round trip — G7's inherited 2.0 ms wall-clock gate timed the wrong boundary, disclosed, not repeated here) | ≤ 2.0 ms | `P1_GATE_LG3_LATENCY_FAILED` (perf flag, not seal basis alone) |
| — | affordance engagement | updates > 0 after steps > 0 | `P1_NO_AFFORDANCE_ENGAGEMENT` → seal falsification |
| — | g4 affordance mean | ≤ 0.0500 | `P1_GATE_G4_AFFORDANCE_FAILED` |

Success verdict: `P1_POLICY_GROUNDING_VERIFIED`.

**Engagement telemetry required in receipt:** `policy_mode == "P1_GOAL_STEERING"`, `p1_score_calls > 0`, `p1_mean_potential_drops` finite (per action), `p1_kernel_latency_ms` present on CUDA. Zero score calls with a P1 receipt → harness defect → `BLOCKED_INFRA`, not a science verdict.

**C3/W0:** WavePacketPathSearch stays GATED. Unseal ONLY on `P1_POLICY_GROUNDING_VERIFIED` + separate approval (packet YAML `C3_W0_AUTHORIZATION`). No W0 work in this carrier.

## Kill criteria (pre-registered)

1. LG1 fails → `P1_GATE_LG1_STAGNATION`: the action-conditioned potential drop does not move the observed trajectory toward the goal → seal falsification, NO retry with tuned hyperparameters (τ_a, horizon, clamp are frozen this carrier).
2. LG2 fails → `P1_GATE_LG2_SOLVED_FAILED`: no level completion in 1,800 steps → seal falsification.
3. Pre-flight PG1/PG1a/PG2/PG3/C2 failure → inherited `G7_AFFORDANCE_FIT_COLLAPSE`, exit 1, no live steps (representation regression would confound the policy measurement).
4. Bank alignment probe fails on the live bank → ABORT, reindex bank, no launch.

## Verdict classes

`P1_POLICY_GROUNDING_VERIFIED` | `P1_GATE_LG1_STAGNATION` | `P1_GATE_LG2_SOLVED_FAILED` | `P1_GATE_LG3_LATENCY_FAILED` | `P1_GATE_G4_AFFORDANCE_FAILED` | `P1_NO_AFFORDANCE_ENGAGEMENT` | `BLOCKED_INFRA` | `G7_AFFORDANCE_FIT_COLLAPSE` (pre-flight).
