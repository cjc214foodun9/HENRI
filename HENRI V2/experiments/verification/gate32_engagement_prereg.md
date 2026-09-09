# Gate 3.2 — Mechanism A Engagement Verification (sealed prereg)

**Carrier:** carrier/stage3-coupling (tip 0009d5a8; Mechanism A implemented + wired)
**Purpose:** resolve `NO_ENGAGEMENT_PROMOTION_WITHHELD` (#e02e3100) by measuring
scorecard-delta engagement on tractable envs with observable level transitions.

## Protocol
- Envs (full ids verified live via available_environments this session):
  `ar25-0c556536`, `lp85-305b61c3`.
- Bounds: 1 env per arm, 60 steps, seed 20260908, GPU-exclusive vast-5090,
  clean detached worktree @ exact SHA, overlay 7557238908, ZONE_C_ENV=prod.
- Flags on BOTH arms (engagement probe): `HENRI_ARC_SCORECARD_DELTA=1
  EXTERNAL_OUTCOME_EFE=1 HENRI_DELTA_GAIN=1`.
- Telemetry predicates (per-step JSONL):
  1. `scorecard_delta_nu` == 1.0 on >= 1 step (authoritative ΔS event, via
     arc_scorecard_delta.detect_level_progress);
  2. `delta_gain_valid=true` on that step;
  3. `scorecard_levels_completed` strict increase (>= 1).

## Pre-registered verdicts
- `A_ENGAGED_DELTA_EVENT`: >= 1 progress event in either env AND arm RC=0
  → Mechanism A engagement MEASURED; promotion eligibility unlocked.
- `NO_ENGAGEMENT_STILL`: zero events in both envs → stay withheld; mechanism
  not falsified; next step = different env family.
- `BLOCKED_INFRA` on any RC != 0 with non-trivial error; `NO_ENGAGEMENT` is
  NEVER relabeled as PASS.

## Honesty constraints
- thermo_ratios / eta_eff: OBSERVED DIAGNOSTIC only (G8AB5 serialization fix),
  not a gate in this protocol.
- No main mutation in this protocol; ff-only merge remains a separate
  approval-gated step after Actions 1 AND 2 both pass.
