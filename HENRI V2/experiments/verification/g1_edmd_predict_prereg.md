# Carrier G1 — EDMD Latent Composition Predictor (ARC task loop)
## Pre-Registration (2026-09-05)

Branch: `carrier/g1-edmd-predict` @ base `origin/main` (8eafe95d).
Worktree: `C:/Users/chan/henri-worktrees/carrier-g1-edmd`.
Module: `HENRI V2/henri_edmd_predict.py` (new). Runner: `HENRI V2/production_arc_run.py` (Layer 0c + flag).

## 1. Mechanism

Fit `RecursiveDualEDMD` (online Koopman operator, r_rank=16, lambda_forget=0.98,
reg=1e-4, O(r^2 * D) per update, no BPTT) from PUBLIC demonstration pairs
(X_i, Y_i) encoded by the live `HENRIVisionEncoder` as [num_blocks, 8] real
phasor waves. For the unseen test X, the predicted solution wave is:

    hat_PSI_Y = T(PSI_X_test, G),  G = normalize(mean_i Y_train_i)  (goal prototype)

`hat_PSI_Y` becomes the **goal anchor** (Layer 0c, default-OFF) so the EFE
planner acts toward a composed target rather than a retrieved/identity target.
Zero pretraining. No writes to the repository. No hold-out leakage: G and the
fit use TRAIN pairs only.

## 2. Data path

demo_pairs (public arcade examples) → `predict_solution_grids` →
tokenizer.encode_spatial_grid per pair → [m, 8192, 8] → per-row unit normalize
→ RecursiveDualEDMD (leave-one-out: hold-out index = last pair) → gate → goal
wave [8192, 8] → `goal_wave` (Layer 0c) → planner λ_goal.

## 3. Pre-registered acceptance/rejection criteria

Gate `G1.2` (same thresholds as the sealed FUNCTOR gate, `arc_task_functor.py`):

| Criterion | Threshold | OK | Underfit |
|---|---|---|---|
| held_out_cos = cos(hat_PSI_Y_h, PSI_Y_h) | > 0.30 | PASS | FAIL |
| margin = held_out_cos − cos(PSI_X_h, PSI_Y_h) | > +0.10 | PASS | FAIL |

- Status `EDMD_PREDICT_OK` iff BOTH criteria hold; else `EDMD_PREDICT_UNDERFIT`
  with `predicted_wave=None` (fail-closed).
- Typed failure statuses: `BLOCKED_NO_DEMOS`, `BLOCKED_EMPTY_DEMOS`,
  `BLOCKED_IMPORT_FAILED`; runner emits `GOAL_EDMD_NO_DEMOS`,
  `GOAL_EDMD_UNDERFIT`, `GOAL_EDMD_FAIL_CLOSED` and falls through to lower
  goal layers (Zone C / preference / identity).
- Zero-pretraining invariant: the operator is fit ONLY from in-context demo
  pairs at test time; no checkpoint, no corpus pretraining.

## 4. Resource limits

Fit: m ≤ 5 pairs × O(r^2 * D) ≈ 5 × 16^2 × 65536 ≈ 84 M FLOPS (trivial).
Module CPU-testable; device follows input tensors. No new envs/databases.

## 5. Expected benefit

Composition capacity for the action path: EFE acts toward a PREDICTED solution
wave instead of the observed/identity wave; sets up egress-from-prediction
(c3-next) on the ARC task loop. If it underfits, flag stays inert (default-OFF)
and no score path changes.

## 6. Failure mode / kill experiment

Kill: `EDMD_PREDICT_UNDERFIT` on a real public-demo episode → the carrier is
measured against the prereg gate and stays default-OFF (telemetry
`GOAL_EDMD_PREDICT` event carries held_out_cos / identity_cos / improvement).
Cheapest kill experiment: the contract test's random-data case must return
UNDERFIT with `predicted_wave=None` (fail-closed proof) and the deterministic
PASS fixture must satisfy both criteria (gate-logic proof).

## 7. Verification plan

1. Contract test `tests/contract/test_g1_edmd_predict.py` (CPU, deterministic):
   module import, status typing, shapes, fail-closed random data, deterministic
   PASS fixture (r_rank = d_model = 16, y_i = normalize(x_i + G) with G ⊥ a,
   self-consistent mean), runner flag default-OFF + Layer 0c presence.
2. Default-path invariance: `git diff origin/main` shows only additive changes;
   flag default "0" → default run byte-identical behavior.
3. Remote CUDA production run (A/B, `HENRI_EDMD_PREDICT=1` vs default) requires
   `APPROVE_REMOTE_RUN` at dispatch; not part of this commit.

## 8. Seals

No sealed artifact is modified: `henri_goal_adapter.py` (sealed 0341a278),
`arc_task_functor.py`, `efe_planner.py`, K3/C1 transports remain untouched.
New module `henri_edmd_predict.py`; runner additions only.
