# Carrier G8 Phase B — Sub-Goal Waypoint Promotion Results

**Verdict: `G8_FALSIFY_SUBGOAL_REACHABILITY`** (sealed 2026-09-02)

**Prereg:** `#fe2bcff4` @1,228 — `docs/spec/g8_subgoal_engine_preregistration.md`
**Packet:** `Carrier_G8_PhaseB_Engine_Wiring_and_Promotion_Protocol.md` (SHA `2c5f70b5…`)
**Engine:** `cb26d816` on `feat/carrier-g8-subgoal-steering`
**Receipt:** SHA `1266424f…` — `/tmp/henri_g8_subgoal/g8_gates_receipt.json`
**Run:** 12 envs × 150 steps, seed 20260930, `HENRI_G8_SUBGOAL=1`, GPU free, log EXIT:0, 0 infra markers

## Gate results

| Gate | Rule (prereg) | Result | Evidence |
|---|---|---|---|
| Reachability | G8 branch entered, meter live, receipt fields present | **PASS** | `policy_mode=G8_SUBGOAL_STEERING`, `g8_meter_active=true`, `g8_promote_threshold=0.6` present, `p1_score_calls=1800` |
| G8-2 | ≥6/12 envs with ≥1 promotion | **FAIL** | `g8_promotions_total=0`, `g8_envs_with_promotion=0`, all 12 env rows 0 |
| G8-3 | ≥1/12 envs solved | **FAIL** | `envs_solved=0` |
| LG1 (flag) | Δν_wp ≥ 0.0100 | **FAIL (flag only)** | `mean_delta_nu_wp=1.11e-05` |
| LG3 (flag) | kernel ≤ 2.0 ms | **FAIL (flag only)** | `p1_kernel_latency_ms=10.48` |

## Meter-trap check

`mean_delta_nu_wp = 1.11e-5` differs from the P1 inherited terminal-goal value (2.07e-4) — the guarded `_g8_meter_ref` hook fired and measured against the active waypoint. Not an inherited-path artifact.

## Interpretation

- Engagement is full: G8 branch entered every step, meter active, action-discrimination calibration passed (`pg1_min_auc=1.0`, pg2/pg3 true), 447 creeps, 17 resets.
- Zero promotions across 1,800 engaged steps with a live near-zero waypoint meter: alignment against the active waypoint never reached the 0.60 threshold (inferred from zero promotions + Δν_wp ≈ 1.1e-5; per-step alignment telemetry was not persisted — next carrier should record per-env `g8_align_max`).
- Pattern reproduces P2-0 at the sub-goal level: actions wiggle state but do not displace it toward any staged waypoint. Sub-goal goal semantics (G8) do not close the action→outcome coupling gap. C1 (action semantics / per-block rotors) remains the correct next sprint per the master plan ordering.

## Falsification chain

29th sealed task-level falsification (32nd carrier). 0 solved envs. W0 stays gated on a non-zero live task completion.

## Next actions

1. C1 design (per-block so(8) rotor action generators, factorized) with its own prereg — do NOT reopen G8 tuning without new evidence.
2. Future G-series carriers must persist per-env alignment maxima (`g8_align_max`) for threshold-reachability attribution.
3. LG3 kernel-batching candidate (448 tiny einsum launches) is a perf carrier, not seal basis.
