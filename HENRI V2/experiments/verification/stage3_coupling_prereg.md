# Stage 3 — VLA Perception-Action Coupling (sealed prereg)

**Spec:** HENRI-SPEC-2026-09-09-STAGE3-COUPLING
**Carrier:** carrier/stage3-coupling (base origin/main 10f5f23, clean worktree)
**Governance:** prereg-before-code; record via henri_audit.py after sealing.

## Scope (two bounded mechanisms, one named carrier, two verdict gates)

### Mechanism A — exteroceptive delta-gain into the transition learner (DEFAULT OFF)
- **Target (live code):** `HENRI V2/efe_planner.py` `train_transition_step` (add keyword
  `outcome_delta: Optional[float] = None`); runner `HENRI V2/production_arc_run.py`
  (`HENRI_DELTA_GAIN` flag, `train_ctx["scorecard_delta"]`, valence override).
- **Rationale (falsifiable):** the runner already computes authoritative scorecard
  progress (`arc_scorecard_delta.detect_level_progress`, winner channel, irreversible
  `levels_completed` increase) and feeds it only to `_vla_causal_planner`. The live
  Koopman learner (`LowRankCoupledTransition`, factorized K = V·Wᵀ + block residual)
  receives a *motion-based* valence (`PROGRESS_VALENCE`) — the external scorecard delta
  never gates the transition update. Mechanism: when `HENRI_DELTA_GAIN=1`, the scorecard
  delta (0/1) overrides the valence passed to `train_transition_step` at the deferred T1
  boundary (causal: `train_ctx` built AFTER `task_progressed` computed same step;
  update applied next step against the OBSERVED wave, never planner prediction).
- **Formula (factor space, NOT the dense rank-1 proposal):**
  `lr_eff = lr * (0.25 + δ/2)` (surprise gate, unchanged) `* (1/(1+ν_s))` for ν_s=1
  (progress crystallizes) or `* (1+ν_s)²` for ν_s<0 (fail damped); ν_s = scorecard delta
  when flag on, else existing valence. Update lands on `field_V`, `field_W`,
  `block_residual` factors only. **No `[D,D]` tensor is ever formed** (34 GiB ban).
- **Contract:** `outcome_delta=None` → byte-identical behavior to today (default path).
- **Engagement predicate:** runner emits per-step `delta_gain_valid`
  (bool(flag AND scorecard_delta_status == SCORECARD_DELTA_OK)); smoke records
  `scorecard_delta_nu` values. **NO_ENGAGEMENT precedence:** if the bounded smoke
  records zero progress events (delta=0 every step), verdict =
  `NO_ENGAGEMENT_PROMOTION_WITHHELD` — mechanism not falsified, promotion withheld.
- **Kill:** local contract contradicts E1/E2/default-path byte identity; dense
  allocation detected (contract scan); engagement zero without a wiring defect.

### Mechanism B — AxiomaticDeficiencyError on degenerate demonstration rank (DEFAULT OFF)
- **Target:** `HENRI V2/henri_goal_adapter.py` — add `class AxiomaticDeficiencyError(RuntimeError)`;
  `HenriTaskOperator.compile_from_demos` raises when `m < 2` or per-block rank < 2
  (second singular value ≤ 1e-6 × σ_max, float32 safe).
- **Audit fact:** Procrustes core (`U_k V_kᵀ` via SVD) is ALREADY implemented; no
  `W_task = I` silent fallback exists. Delta = typed error + rank gate + runner
  catch-site fail-closed: status `GOAL_ADAPTER_AXIOMATIC_DEFICIENCY`, goal_wave stays
  None, NO identity assignment, NO action-policy fallback (planner proceeds without
  goal exactly as today).
- **Contract:** m=1 raises; m=2 collinear (Y=X) raises; m=2 generic passes with
  orthogonality_err ≤ 1e-4; m=0 still routed by existing `GOAL_ADAPTER_NO_DEMOS`.

### Diagnostic enrichment (part of this carrier, per user precondition)
Engagement telemetry for A (`delta_gain_valid`, `scorecard_delta_nu`) and B
(`axiomatic_fired` counter). P@1/P@k of the E2 egress probe is a SEPARATE carrier
(E2-runner extension), surfaced, not bundled.

## Verification sequence
1. TDD: `tests/contract/test_stage3_coupling.py` (RED before implementation).
2. Local contract run (CPU, isolated interpreter, relative PYTHONPATH).
3. Two atomic commits (A, B) on carrier/stage3-coupling; push.
4. Remote GPU-exclusive (vast-5090 idle): clean detached worktree @ exact SHA,
   overlay 7557238908 staged, `set -a` DSN, full suite + contract + bounded smoke
   A/B: env `ka59-38d34dbb`, seed 20260908, 60 steps, `HENRI_ARC_SCORECARD_DELTA=1`,
   arms `HENRI_DELTA_GAIN=0|1`, explicit DONE marker per arm.
5. Each verdict sealed via henri_audit.py; verdict doc committed to carrier.
6. **Promotion + benchmark dispatch = separate fresh approval.** No main change in this carrier.

## Pre-registered acceptance
- A_EVALUATED: delta_gain_valid=True on ≥1 step AND transition loss finite;
  byte-identity default path; RUNS_RC=0.
- B_EVALUATED: raise fires on degenerate fixture (unit) and runner catch produces
  fail-closed status without goal fallback.
- PROMOTION: both evaluated AND bounded smoke shows ≥1 delta=1 progress event
  (else NO_ENGAGEMENT_PROMOTION_WITHHELD).
