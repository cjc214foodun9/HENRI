# Gate 4 — Live ARC-AGI Gauntlet Pre-Registration (Prerequisite Manifest + Kill Criteria)

**Document:** HENRI Gate 4 Gauntlet Pre-Registration
**Status:** PREREGISTERED (hypothesis — execution receipts will upgrade to observed)
**Date:** 2026-08-26
**Candidate commit:** `804a793c9957806628ab3da23527b285a528787b` (branch `feat/temporal-navigation-t0`, remote-reconciled)
**Parent seal:** `74792b84-cd13-4e47-ad66-5f9de18248cf` (Gate 1 execution seal, observed)
**Source PDF:** `Information Theory, Learning Dynamics, and General Covariance.pdf`, SHA-256 `39b8aa1f69f7a85d2ec1b4188983341bdb9a4c25a68070d1a5f22c7b645105ae` — treated as PROPOSAL, not authorization for load-bearing change.

## 1. Purpose

Gate 4 unlocks only after every prerequisite field below is PINNED with observed evidence. Until then the live ARC-AGI gauntlet is `BLOCKED_PREREQUISITES`. No silent advancement.

## 2. Prerequisite manifest (pinned state as of 2026-08-26)

| Field | Status | Pinned value / evidence |
|---|---|---|
| Environment list | PINNED | 12 envs from `/workspace/gate1-run/HENRI V2/environment_files`: ar25, bp35, cn04, dc22, ft09, g50t, ka59, lp85, ls20, m0r0, re86, sb26 |
| Evaluator versions | PINNED | arcengine 0.9.3, arc_agi 0.9.9 (`pip show` OBSERVED) |
| Evaluator source manifest SHA-256 | PINNED | `9e2f99a0141094b124382360dcda1793082330a4102984d53f4197a30bec042f` (17 *.py files under site-packages/arcengine + arc_agi, sorted, hashed) |
| Evaluator raw manifest SHA-256 | PINNED | `6a5c2b2d9356a79e1c1c5a1ebcd631913bf5bd65ee441a45bf24cd707f794e00` |
| Candidate commit + clean worktree | PINNED | `804a793c…`; remote worktree `/workspace/gate1-run/HENRI V2`; local == remote == HEAD |
| Checkpoint overlay SHA-256 | PINNED | `75572389083455a371546b40500b6614abfc3a245cfa0db9eba74c183a974060` |
| Checkpoint state-dict SHA-256 | PINNED | `e6ede41e0834807fa84ce55bda89a49d9b036a2f2e4a4a40bbbf5670c1943470` |
| Checkpoint compatibility metadata | PINNED | down_proj.weight [2048,65536]; layer_norm [2048]×2; lm_head.weight [32000,2048] — matches §3A decoder contract |
| Seed | PINNED | `20260826` (gauntlet RNG derived from this) |
| Timeout | PINNED | 60 s per step; 3600 s total |
| Exposure history | RECORDED | Corpus ledger `/workspace/gate1-run/out/temporal_ledger.jsonl` (1472 rows) exposes replay envs; **heldout split NOT generated → BLOCKED_NOT_GENERATED** |
| Action-path preflight | PASS | `arc_action_probe.py` on ft09/ka59/m0r0: bare actions + ACTION6+payload change frame state; ACTION6 source=env_actioninput, screen-space coords, wave_unbind_status=WAVE_UNBIND_UNAVAILABLE |
| Score eligibility | BLOCKED | `ACTION_HEAD_NOT_CALIBRATED` — no provenance-validated semantic action head; `score_eligible=false` |

## 3. Pre-registered verdict classes and kill criteria

- `BLOCKED_PREREQUISITES` — any field above unpinned at gauntlet start (fail closed).
- `BLOCKED_ACTION_HEAD_NOT_CALIBRATED` — score eligibility false; external score claim forbidden.
- `BLOCKED_ENV_PIN_MISMATCH` — env list at run time differs from the 12 pinned ids.
- `BLOCKED_EVALUATOR_HASH_MISMATCH` — evaluator source manifest differs from `9e2f99a0…`.
- `BLOCKED_CHECKPOINT_HASH_MISMATCH` — overlay or state-dict SHA differs from pinned.
- `BLOCKED_SEED_OR_TIMEOUT_MISMATCH` — run config differs from pinned seed/timeout.
- `ARC_RHAE_SCORE` — live gauntlet outcome ONLY when: every prerequisite pinned, score_eligible=true, heldout split generated single-use, and contamination/exposure recorded.
- `FEW_SHOT_SCALING` (Gate 1, already observed `804a793c`) — NOT re-litigated here; Gate 4 depends on Gate 1 verdict only as a scientific precursor, not as a blocker.

## 4. Non-score-bearing preflight evidence (OBSERVED)

- `arc_action_probe.py` (production `(GameAction, data)` path) on ft09/ka59/m0r0: PASS — at least one exact production call changes meaningful frame state on each env. ACTION6 coordinate payload flows through `step_with_payload` with camera params; screen-space; `WAVE_UNBIND_UNAVAILABLE` recorded (wave unbinder not engaged in preflight; does not grant eligibility).
- Corpus regenerated at HEAD with Carrier A exporter: 1472 ledger rows, 529 payloads, missing_payload 0.

## 5. PDF disposition (proposal, not authorization)

Price→Lambda covariant gradient ascent, Natural Induction viscoelastic creep, second-order covariance pooling, Stiefel SGLD, epiplexity — consistent with existing HENRI mechanism families (qFHRR, SGLD, EDMD). No load-bearing math, schema, evaluator, or main-promotion change is authorized by this document alone. Each would require its own bounded design + approval.

## 6. Corpus consult (INFERRED, 2026-08-26)

NotebookLM bank `ca4bb787` (229 sources), session `3179135d-9cca-42a0-a626-3b192ea55cc2`:
- Supports Price-to-Lambda covariant gradient ascent (`g^ij = C_ij`), Natural Induction viscoelastic creep, second-order covariance pooling (bypasses the G2 lexical-snap blindspot), and epiplexity `S_T` as MDL dual — as HENRI mechanisms.
- Labels falsified in corpus: Newton-Schulz divergence under Langevin noise; isotropic Langevin thermalization; target-input context leakage; Kuramoto `r ≥ 0.95` as blanket termination (root tautology).
- Blocks: live ARC eval 0.0 / `BLOCKED_NO_DEMONSTRATIONS` under `HENRI_FREEZE_LEARNING=1`; fast-forward merge of `feat/phase827-production-promotion` pending CUDA `4667f08` + approval.
- These are `INFERRED` corpus syntheses; live code and CUDA telemetry override on conflict. The PDF's mechanisms map to existing HENRI families; no new load-bearing mechanism is authorized by this consult.

## 7. Next actions

1. Corpus consult (NotebookLM bank `ca4bb787`) on PDF themes — INFERRED labels.
2. Generate single-use heldout split with fresh seed (only after approval to run gauntlet).
3. Re-run all manifest hashes at launch; refuse mismatches.
4. Live gauntlet only after approval + all prerequisites PINNED.
