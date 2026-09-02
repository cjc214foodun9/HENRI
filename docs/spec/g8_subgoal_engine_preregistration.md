# Carrier G8 Phase B — Engine Wiring Pre-Registration

**Packet:** `Carrier_G8_PhaseB_Engine_Wiring_and_Promotion_Protocol.md`
(SHA-256 `2c5f70b51062b4c63015995f6114353b26e45d5ebf5df088550a19f148e520d5`,
159 lines / 8,959 B, `HENRI-SPEC-2026-09-V3-CARRIER-G8-PHASE-B-WIRING`).
Sealed ledger event: `G8_PHASEB_PREREG_SEALED #fe2bcff4` @ 1,228.
Base: `feat/carrier-g8-subgoal-steering` @ `71276baf` (Phase A tip; local == origin).

## Mechanism

- Per env, the engine binds a **waypoint chain** `[K, D]`: bank rows selected by
  `arc_g8_waypoint_extractor.extract_waypoints` (curvature peaks + terminal),
  normalized to S^{D-1}, flat. Rank-1 row storage; NO dense operators
  (packet memory invariant: ~6.9 MB for 12 × 9 × 64 KB).
- The active target is `chain[k*]`, NOT the terminal attractor.
- **Promotion:** at each step, if `|align(psi_full_t, chain[k*])| >= 0.60`
  (packet threshold), increment `k*` (capped at `K-1`); scoring then
  references `chain[k*+1]`.
- **Scoring:** the P1 potential-drop policy `j(a)` is retained verbatim but its
  goal reference is the ACTIVE waypoint (P1 machinery reads `_p1_goal_full`,
  which G8 rebinds per promotion).
- **Meter (M1 re-target):** the repaired Δν measurement is computed in the
  FULL domain against the ACTIVE waypoint reference captured at decision time:
  `c_t = |align(psi_full_t, ref)|`, `c_next = |align(psi_full_{t+1}, ref)|`
  (post-step actual frame, M1 discipline). Guarded hook in
  `arc_g4_aligned_engine.run_gauntlet`; default path (no `_g8_meter_ref`
  attribute) is byte-identical. Meter values are NOT comparable to the legacy
  P1/P2 [64]-domain meter (different domain, different reference policy) —
  disclosed, and G8-LG1 is a FLAG, not a kill.

## Gates (packet gate matrix)

| Gate | Bound | Action on breach |
|---|---|---|
| G8-2 | envs with ≥1 promotion ≥ 6/12 | `G8_FALSIFY_SUBGOAL_REACHABILITY` |
| G8-3 | solved envs ≥ 1/12 (1,800 steps) | `G8_FALSIFY_TASK_SOLVE_LG2` |
| G8-LG1 | mean_delta_nu_wp ≥ 0.0100 (new meter) | `G8_FLAG_LOW_VELOCITY_PROGRESSION` (flag) |
| G8-LG3 | mean latency ≤ 2.0 ms | `G8_FLAG_KERNEL_PERF_REGRESSION` (flag) |

Verdict precedence (mechanism reachability before outcome): no-affordance
engagement → G8-2 → G8-3 → LG1 → LG3 → G4 → `G8_SUBGOAL_STEERING_VERIFIED`.

## Bounds (identical to P1/P2-0 for comparability)

12 envs × 150 steps = 1,800 arcade steps, seed `20260930`,
`HENRI_G8_SUBGOAL=1` + `HENRI_P1_GOAL_STEERING=1` +
`HENRI_G7_CALIBRATED_AFFORDANCE=1`; trajectory bank
`trajectories_production_run_f3v2.npz` (sha `9e3c01b4…`) + its JSONL;
fresh output dir `/tmp/henri_g8_subgoal/`; GPU-free preflight; setsid nohup;
receipt `g8_gates_receipt.json`.

## Kill experiments (cheapest falsification first)

1. `G8_FALSIFY_SUBGOAL_REACHABILITY` — fewer than 6/12 envs ever promote
   `k* > 0`: the staged-waypoint hypothesis does not even engage on live
   trajectories.
2. `G8_FALSIFY_TASK_SOLVE_LG2` — promotions occur but 0 solved: staged goals
   re-rank actions without changing task outcome (P1-class coupling failure
   recurs at waypoint granularity).

## Failure mode / fallback

Flag absent → G8 module never imported (lazy import; default G7/P1 path
byte-identical, differential proof at the launcher). Env missing from chain
bank → falls back to P1/G7 scorer and legacy meter.
