# CLASS50 Pre-Registered Packet

**Document Identifier:** HENRI-PACKET-CLASS50-UNFROZEN-PREDICTION-DRIFT-2026
**Status:** DRAFT — not approved, not executed
**Baseline Commit:** `ef0ef49` (CLASS49 attribution refactor, sealed `4c439d2`)
**Predecessor Verdict:** CLASS49 `ATTRIBUTION_REFACTOR_NEUTRAL_GATE3_INCONCLUSIVE`

---

## 1. Problem Statement

CLASS49 Gate 3 returned `INCONCLUSIVE` under the F4 sanity floor: with
`HENRI_FREEZE_LEARNING=1` on both arms, per-step mean Sagnac delta was
0.9877 (arm A, n=1234) and 0.9860 (arm B, n=1098). Both means are above the
0.95 floor, so the relative differential (δ = −0.0017) cannot adjudicate
prediction drift over a saturated ~1.0 error channel.

Root-cause hypothesis (INFERRED from NotebookLM corpus F4, 2026-08-21):
freezing disables online EDMD parameter updates. The low-rank transition
operator `T = V·W† + R_block` cannot track environmental state changes, so
the predicted next state stays fixed while the exteroceptive observation
evolves, forcing the Sagnac error floor near 1.0.

## 2. Scientific Hypothesis (Falsifiable)

- **H1:** With `HENRI_FREEZE_LEARNING=0`, online EDMD (dual Woodbury /
  thin-SVD, no d² tensors) tracks the environment. Per-step mean Sagnac
  delta de-saturates below 0.95 on at least one arm, and the relative
  differential stays within δ ≤ +0.005.
- **H0 (kill):** At least one arm remains ≥ 0.95 under unfrozen learning.
  The drift is not caused by the frozen operator; online EDMD does not
  resolve prediction drift. Seal as
  `PREDICTION_DRIFT_NOT_RESOLVED_BY_UNFREEZE`. Do not retune.

## 3. Design

| Item | Specification |
|---|---|
| Environment pool | 25 matched ARC-AGI envs × 60 max steps (same pool as CLASS49/48) |
| Arm A | Baseline EFE planner (`HENRI_ARC_RT_MCTS=0`) |
| Arm B | RT-MCTS planner (`HENRI_ARC_RT_MCTS=1`) |
| Learning mode | `HENRI_FREEZE_LEARNING=0` on BOTH arms (the treatment) |
| Seed | `20260822` (matched with CLASS49) |
| Runner | `production_arc_run.py` at `ef0ef49` — no new code changes |
| Telemetry | `/tmp/class50_arm_a/`, `/tmp/class50_arm_b/` (isolated) |
| GPU | vast-5090, sequential scheduling, no concurrent CUDA jobs |
| Decoder overlay | SHA `75572389083455a371546b40500b6614abfc3a245cfa0db9eba74c183a974060` in fresh worktree, verified pre-launch |

### Learning-write closure (CLASS49 guard is live in prod)

- Every write carries valid `run_id`, `arm_id`, `commit_sha` (fail-closed
  guard, OBSERVED in CLASS49 dev smoke).
- Engram counts recorded pre/post per arm (expect `post − pre == writes`).
- Write counts per arm logged from telemetry.
- Learning engagement telemetry per step: `plasticity.mean_alpha`,
  `frozen_fraction` (must be non-zero engagement for the unfrozen arm).

### Contamination controls

- Zero ARC task pre-ingestion into Zone C or model stores (standing
  zero-pretraining invariant; no demo reconstruction).
- Matched seed draws for the paired comparison.
- Telemetry isolation per arm; shared verdict path fresh-mtime rule.
- No code changes between arms (identical config, only the flag differs).

## 4. Pre-Registered Gates

**Gate 1 — Attribution Isolation (PASS/FAIL).**
Metric: un-attributed DB writes during run. Requirement: exactly 0.
All writes carry run_id, arm_id, commit_sha.

**Gate 2 — Task Performance Delta (PASS/FAIL).**
Metric: Δscore = Score_B − Score_A. Requirement: Δscore > 0.

**Gate 3 — Unfrozen Drift Discrimination (PASS/FAIL — no INCONCLUSIVE state).**
Metric: per-step mean Sagnac delta per arm (field `sagnac_delta`,
finite-only, n reported; same field as CLASS49).
Requirements (all three):
- (a) `min(mean_A, mean_B) < 0.95` — de-saturation on at least one arm;
- (b) `δ = mean_B − mean_A ≤ +0.005`;
- (c) `mean_B ≤ 0.995`.

Verdict: PASS iff (a) AND (b) AND (c). FAIL if any bound fails — including
both arms ≥ 0.95 (H0 kill: the unfreeze treatment was applied and did not
de-saturate the channel).

**Gate 4 — Subspace Retrieval Isolation (PASS/FAIL).**
Metric: cross-domain query leakage events. Requirement: exactly 0 action→ast
or ast→action queries. With learning ON, additionally require store
reconciliation: `post_engram_count − pre_engram_count == writes` and zero
ast-family rows in the action namespace.

### Learning-dynamics kill thresholds (non-science verdicts)

- NaN/inf in `loss_ema` or `transition_loss` on either arm → `BLOCKED`.
- `plasticity.mean_alpha == 0` on an unfrozen arm → `BLOCKED_LEARNING_NOT_ENGAGED`
  (learning did not engage; not a science verdict).
- Attribution failure on any write → Gate 1 FAIL → seal, no promotion.

## 5. Recovery Rules

Infrastructure failure (env download timeout, CUDA OOM from contention,
SSH drop) → `BLOCKED_INFRASTRUCTURE`. Preserve artifacts under distinct
names with SHA-256, probe connectivity, relaunch the SAME config. No code
changes mid-run (CLASS49 arm A r1 precedent).

## 6. Promotion Boundary

- `HENRI_ARC_RT_MCTS` remains default-OFF throughout; main untouched.
- Promotion to main requires: all four gates PASS **and** explicit user
  approval **and** main-branch release convergence (FF push from a verified
  clean candidate; GitHub main = clean Vast deployment SHA; no history
  rewrite).
- Seal event + receipt on completion (schemas `henri.governance-event.v1`,
  `henri.class50-receipt.v1`), committed to `accuracy/fidelity-remediation`
  or successor branch.

## 7. Approval Gate

This packet is a DRAFT. No execution, no store mutation, no learning-state
change until the user approves this document. Approval method: explicit
reply in the active session (mirrors CLASS49 Option-1 approval).
