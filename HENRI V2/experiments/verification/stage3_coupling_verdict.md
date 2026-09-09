# Stage 3 Verdict — VLA Perception-Action Coupling (measured)

**Carrier:** carrier/stage3-coupling @ `457bbc3d` (base 10f5f23; prereg `49f9969f` sha256 2a961dda…)
**Verify:** GPU-exclusive vast-5090, clean detached worktree @ `457bbc3d`, overlay 7557238908, HAS_DSN=True

## Remote gates (OBSERVED)

| Gate | Result |
|---|---|
| Full suite | FULLSUITE_RC=0 — 1339 passed / 12 skipped |
| Contracts (A+B+wiring+adjacent) | CONTRACT_RC=0 — 20 passed |
| ARM_OFF (HENRI_DELTA_GAIN=0) | RC=0 — 128 rows, delta_gain_valid=0, status SCORECARD_DELTA_OK ×60 |
| ARM_ON (HENRI_DELTA_GAIN=1) | RC=0 — 128 rows, delta_gain_valid=60, nu_events=60, progress_events=0 |

## Mechanism A verdict: `A_EVALUATED_WIRED_ENGAGEMENT_PENDING` → `NO_ENGAGEMENT_PROMOTION_WITHHELD`

- Wiring proven (dead-flag trap avoided): ON arm emits `delta_gain_valid=60`, OFF arm emits 0; both arms report `scorecard_delta_status=SCORECARD_DELTA_OK` 60/60; `nu` values all 0.0 (Counter{0.0: 60}); `scorecard_levels_completed=0` all rows.
- Outcome-level engagement unmeasured: zero `levels_completed` progress events in the 60-step window on both arms → no `delta=1` event → the delta-gain never changed an update. Per prereg: mechanism NOT falsified, promotion WITHHELD.
- Factor-space only: scalar `lr_eff` gate on `field_V/field_W/block_residual`; no `[D,D]` formed; contract `test_A_default_path_no_dense_allocation` passed.
- Causal ordering: `train_ctx["scorecard_delta"]` captured same step as triple stash; update deferred to next step against OBSERVED wave (subtraction-tautology guard intact).
- Closed defect during verify: `UnboundLocalError: scorecard_delta_status` (ON arm, zero rows) — variable bound only inside `if EXTERNAL_OUTCOME_EFE:`. Fixed with run-scope init + regression contract test; second verify at exact fix tip is the result above.

## Mechanism B verdict: `B_EVALUATED_CONTRACT` (live-path engagement blocked by design)

- `AxiomaticDeficiencyError` raises on m<2 and on per-block σ₂/σ₁ ≤ 1e-6 (fixtures verified in remote contract run).
- Runner catch `GOAL_ADAPTER_AXIOMATIC_DEFICIENCY` fail-closed: no identity assignment, no action fallback; specific except precedes generic.
- No `W_task = I` fallback exists in compile path (tested).
- Live engagement NOT exercised: smoke did not enable `HENRI_GOAL_ADAPTER`; ARC envs expose `examples: None` (BLOCKED_NO_DEMOS), so the fail-closed path is the designed default; contract coverage is the evaluation level reached.

## Dispositions vs user decision (all honest)

1. Wire Δs into Dual-EDMD — `BOUNDED_IMPLEMENTABLE` → implemented in live learner (`EFEPlanner.train_transition_step`, factorized `LowRankCoupledTransition`); user's dense rank-1 formula = 34 GiB at D=65,536 → banned (matches sealed prereg); `henri_unified_vla.py` has no live consumer (docs/smoke only) so target was remapped to the live path.
2. Procrustes + rank<2 error — core `ALREADY_IMPLEMENTED` (per-block U Vᵀ, orthogonality 7.2e-6); delta = typed error + rank gate + runner catch, implemented.
3. ff-only promote + gauntlet — NOT executed. Promotion needs fresh approval AND outcome engagement gateway; main untouched `10f5f23`; AAII v4.2 remains NOT_EVALUATED; ARC score eligibility unchanged (E2 head is a retrieval probe, not a calibrated action head).

## Claims

- `OBSERVED`: all RC, counts, telemetry values, SHAs (direct tool reads / remote logs / local re-analysis).
- `DERIVED`: action divergence 32/124 (process-level stochastic arms; per-arm independence expected).
- `NO_ENGAGEMENT_PROMOTION_WITHHELD`: declaration per prereg, not a mechanism falsification.
- `BLOCKED`: P@1/P@k diagnostic (separate carrier, surfaced); live ARC B engagement (BLOCKED_NO_DEMOS by design).

## Next action

1. (Optional) longer/richer window or env with observed level progress to produce ≥1 delta=1 event → then A engagement measurable.
2. Fresh approval for promotion only after engagement evidence; benchmark dispatch remains a separate gate.
