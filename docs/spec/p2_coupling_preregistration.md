# Carrier P2-0 Pre-Registration: Action→Outcome Coupling Measurement

**Carrier:** `P2_ACTION_OUTCOME_COUPLING` (31st carrier; measurement-resolved rerun)
**Source:** M1 closeout `Carrier_M1_Closeout__Measurement_Line_Repair___Telemetry_Disambiguation.md` (SHA `688703bb…`) — Candidate P2; bounds inherit the sealed P1 prereg `docs/spec/p1_goal_grounded_policy_preregistration.md` (`f4f73d54…`, ledger `a38874e5` @1,206).

## Hypothesis

P1's policy trajectory made real progress toward waypoints that the stale-`psi64` meter hid as `mean_delta_nu_wp: 0.0`. With the M1 repair, re-measuring the SAME trajectory yields Δν > 0.

## Design

- Engine: `arc_g7_calibrated_engine.py` at M1 tip `3d519d2`, flags `HENRI_G7_CALIBRATED_AFFORDANCE=1 HENRI_P1_GOAL_STEERING=1` (identical to P1 launch).
- Bounds (identical to P1): 12 envs × 150 steps, seed **20260930**, horizon 8, bank `trajectories_production_run_f3v2.npz/.jsonl`, env list ar25 sc25 tr87 cd82 lp85 wa30 ft09 g50t sk48 bp35 ka59 sb26.
- Output: `/tmp/henri_p2_coupling/`, receipt `p2_gates_receipt.json`.
- Same seed is DELIBERATE: policy is deterministic → same actions → only the measurement line changed. Δν≠0 is fully attributable to M1.

## Verdicts (pre-registered)

| Verdict | Condition |
|---|---|
| `P2_PROGRESS_CONFIRMED` | mean Δν > 0.05 **or** any waypoint advance > 0 — P1 falsification revised to PARTIAL (policy progressed; meter was broken) |
| `P2_NO_PROGRESS` | Δν ≈ 0 AND advances = 0 — P1 steering never moved the progress metric (orthogonal-to-task failure; deeper carrier needed) |
| `P2_SOLVED` | any env solved ≥ 1 — first live solve in project history |
| `BLOCKED_INFRA` | infra failure before step 0 |

Non-verdict telemetry: per-step Δν distribution (now meaningful), waypoint index trajectory, ΔV(a) drops, G4 affordance residuals.
