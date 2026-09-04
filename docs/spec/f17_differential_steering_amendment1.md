# Carrier F17 — Pre-Registration Amendment 1 (run-1 telemetry defect)

- Prereg sealed: `F17_PREREG_SEALED #cf3d2590…` @ ledger 1,107 (commit `0eca77f`)
- Amendment sealed: commit `1bc54bd` (fix)

## Defect (run-1, OBSERVED)

- `killing_gamma_std_mean: NaN` in the run-1 receipt.
- Root cause: engine line 392 used `torch.std()` (sample std, correction=1) over
  per-step candidate pools. A single-candidate pool (ft09-class envs expose one
  available action) yields `degrees of freedom <= 0` → NaN → the per-step NaN
  poisoned the aggregate mean.
- Gate consequence: `_verdict` E1 engagement check used `gstd <= threshold`;
  `NaN <= x` is False → the engagement gate was BYPASSED (fail-open) in run-1.

## Fix (pre-registered amendment)

1. Population std `gams.std(correction=0)`: single-candidate pools report 0.0
   variation (finite), not NaN.
2. Fail-closed E1 gate: non-finite `killing_gamma_std_mean` (NaN/Inf) now returns
   `F17_FALSIFIED_NO_ENGAGEMENT` instead of proceeding to gate iteration.
3. C16 regression test covers both behaviours.

## Engagement in run-1 (DERIVED, not telemetry-NaN)

`killing_gamma_min_mean = −0.3312`, `killing_gamma_max_mean = +0.5301` (both
finite, OBSERVED). min ≠ max ⇒ there exists at least one step where γ varied
across candidates ⇒ the Killing-form warp was candidate-differential (not the
F16 common-mode class). Mechanism engaged; steering still absent
(Δν_goal −3.2e-05, 0 solved).

## Run-2 plan (same bounds, per retry discipline)

- Bounds unchanged: 12 envs × 150 = 1,800 steps, seed `20260916`, K=8,
  κ_diff 0.75, μ_damp 0.15, β_sagnac 0.05.
- Run-1 receipt preserved as `/tmp/henri_f17_differential/f17_gates_receipt_run1.json`.
- Run-2 receipt is the valid E1 telemetry source; verdict from run-2.
