# RT-MCTS A/B Activation Packet (HENRI-CLASS48)

Doc ID: `HENRI-CLASS48-RT-MCTS-AB-2026-08-21`
Status: PRE-REGISTERED (written before arm execution)
Date: 2026-08-21
Approval: user directive (henri-agent-integration invocation, 2026-08-21): launch
pre-registered A/B evaluation with `HENRI_ARC_RT_MCTS=1`; paired envs; gate = positive
level-score delta vs EFE-only without increasing Sagnac veto oscillations.

## 1. Hypothesis

The Ryu-Takayanagi tensor-cut steering mechanism (`feat/phase822-ryu-takayanagi-mcts`,
merged at `76e0e26`, consumer `production_arc_run.py:1657` under `HENRI_ARC_RT_MCTS`)
minimizes entanglement entropy S_RT = Area(γ)/(4G_N) across the Wave-JEPA lookahead
tree. Lower tensor-cut area should bias candidate ranking toward trajectories with
less information scrambling, converting into measurable external level completion
gains over the EFE-only baseline on ARC-AGI-3 environments.

Prior evidence (sealed ACCEPT, phase822): env sp80 Level 1 completed, score 4.76.
This packet measures the claim in paired form.

## 2. Frozen configuration

- Code: `accuracy/fidelity-remediation` at the post-packet commit SHA (recorded in
  the governance event; runner + contract tests included).
- Checkpoint overlay: `HENRI V2/models/henri_decoder_checkpoint.pt` copied into the
  remote worktree from the persistent Vast deployment; SHA-256 + byte size verified
  BEFORE launch; `checkpoint_load_status == LOADED` required.
- Zone C: `ZONE_C_ENV=prod` with the host-resolved DSN. If `ZONE_C_PROD_DSN` is not
  set on the host, the run is `BLOCKED` (fail closed; no surrogate).
- Environment pool: all 25 Arcade envs in the same fixed order for both arms.
- Episode budget: 60 steps per env per arm (matched), `HENRI_SEED` matched per env
  across arms (seed s for arm A env e == seed s for arm B env e).
- Arm A: EFE-only (`HENRI_ARC_RT_MCTS` unset / 0). Arm B: identical config plus
  `HENRI_ARC_RT_MCTS=1`.
- Sequential GPU execution: arm A completes before arm B starts; isolated telemetry
  directories per arm; no competing GPU jobs.

## 3. Outcome variables (pre-registered)

Primary outcome: `Delta_score = sum(levels_completed_B) - sum(levels_completed_A)`
across the matched pool.

Pass: `Delta_score > 0.0`.

Gate S (Sagnac, ONE pre-registered statistic — absolute arm-B level, consistent with
the production veto threshold): `mean(per-step Delta_Sagnac)_B <= 0.35`.
Secondary veto-rate guard: `veto_count_B <= veto_count_A + 2`.

Branch engagement (mandatory before any verdict): `rt_mcts_branch_hits_B > 0` AND
evidence the RT-MCTS re-rank changed candidate ordering vs arm A (rank-change count
> 0 in telemetry). Zero hits = `NOT_EXERCISED`, never PASS/FAIL.

## 4. Kill rules

- Any arm exits non-zero (or DONE marker absent) -> `BLOCKED_INFRASTRUCTURE`; no
  scientific verdict, no promotion.
- Arm B wins only through veto suppression with `Delta_score <= 0` -> `NOT_PASS`.
- Arm B `Delta_score > 0` but Gate S fails -> `NOT_PASS` (hypothesis falsified on
  the safety channel), mechanism stays default-OFF.
- Score promotion additionally requires: action-semantics validity
  (`trained_action_head_active` unchanged per config), eligibility gate valid,
  checkpoint provenance recorded.

## 5. Decision rules

- Both arms PASS + engagement: `RT_MCTS_ACCEPTED_AB`; candidate for default-ON
  only after a second replication packet (pre-registered separately).
- Either arm fails: `RT_MCTS_FALSIFIED_AB`; `HENRI_ARC_RT_MCTS` stays default-OFF;
  evidence sealed immutably.
- Infra failure: `BLOCKED_INFRASTRUCTURE`; relaunch same SHA after cause repair.

## 6. Governance

- Packet + frozen SHA recorded in the audit chain event for this approval.
- Mixed-seal branch `feat/ccos-incommensurate-spatial-carriers` HELD (not merged);
  8.8-B extraction is a separate later PR.
- Tranche 1 archival (11 files) confirmed `ARCHIVED_NON_LIVE` in HEAD; dead
  top-level O-VSA import removed at `production_arc_run.py:48`; module file preserved.
