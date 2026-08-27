# Arm F — Successor-Feature Action Scoring (SFAS) — Pre-Registration

- **Date:** 2026-08-27
- **Branch/carrier:** `carrier/goal-adapter` (base `0833306`)
- **Seal:** recorded at commit time via `henri_audit.py`
- **Approval:** user approved S1 + R1–R4 fixes and the bounded carrier proposal ("Approved s1 and the r1-r4 fixes, and bounded carrier proposal. Please implement and wire in now").

## 1. Problem (measured)

Five-arm telemetry (env `ka59-38d34dbb`, seed `20260827`, 30 steps):

| Arm | goal_dist engaged | sagnac | mechanism |
|---|---|---|---|
| A baseline | — | 0.9993 | — |
| B learning | — | 0.9991 | LEARNING |
| C goal channel | 0.9957 | 0.9718 | ZONE_C 30/30 |
| D latent adapter | 0.996 | 0.9714 | LATENT 27/30 |
| E subspace proj | 1.0049 | 0.9710 | PROJECTED 27/27 |

`goal_dist ≈ 1.0` for EVERY candidate in every arm: the goal term is
ORDER-INVARIANT (it rescales all candidates, it cannot re-rank). Arm E
(KILL_SUBSPACE_PROJECTION, seal `#5260008a`) proved a shared-span projector
cannot manufacture discrimination the operator lacks.

## 2. Mechanism (S1)

Per-action successor features (Barreto et al. 2017):

```text
psi_a(s) = sum_{k=0}^{H-1} gamma^k * K_a^k * phi(s)
score(s, a) = cos(psi_a(s), phi(g))
efe' = efe + lambda_sfas * (1 - score)
```

- `K_a x := transition(x, a_wave)` — the LIVE `LowRankCoupledTransition`
  (efe_planner.py:158-178; FHRR bind + field channel; the action enters the
  prediction through the bind, so the successor feature is action-specific).
- Matrix-free: H forward passes of the operator, never a `[d,d]` tensor.
- Zero trainable: transition called read-only, detached; scorer holds no
  parameters.
- Blended EFE preserves argmin semantics; stable sort; missing scores keep
  raw EFE (no reorder for unscored rows).
- This is the ADAPTER lever (user constraint: no learner/codec tuning).

## 3. Bounds (pre-registered)

- Env `ka59-38d34dbb`, seed `20260827`, 30 steps (paired-measurement
  envelope from arms C/D/E).
- Flag `HENRI_SFAS=1` (default OFF). Requires a goal wave; with
  `goal_wave=None` the block is a no-op (fail-open, order byte-identical).
- Horizon `HENRI_SFAS_HORIZON` in [1, 4], default 2.
- Gamma `HENRI_SFAS_GAMMA` in [0, 1), default 0.9.
- Lambda `HENRI_SFAS_LAMBDA` >= 0, default 1.0.
- Remote Vast CUDA only; one run per GPU; run-scoped telemetry dir.
- Launcher: `cwd=/root/henri-goal-adapter/HENRI V2`, `ZONE_C_ENV=prod`,
  `set -a` before `source /workspace/zonec_prod.env`, `/venv/main/bin/python3`.

## 4. Kill gates (binding)

1. **ENGAGEMENT:** `sfas.reordered` True on >= 1 step of 30 (the scorer must
   actually re-rank at least once; zero reorders across all steps =
   `ENGAGED_EFFICACY_FALSIFIED` — same class as arm E).
2. **DISCRIMINATION:** paired EFE-table discordance (rank changes) > 0
   across the run; if every row's blended order equals the raw EFE order,
   the mechanism is inert (`FALSIFIED_NO_DISCRIMINATION`).
3. **ORDER-INVARIANCE CONTROL:** the contract test `scores_differ_across_actions`
   must pass on the remote CUDA suite (proves the scorer reads
   action-specific structure from the live operator at reduced scale).

Non-binding (diagnostic): goal_dist band, sagnac mean/variance — the goal
is to fix RANKING, not coherence.

## 5. R3 telemetry (superposition crosstalk, diagnostic)

Per-step `superposition_load = num_engrams / d_model` from the live Hopfield
cleanup store (M/D; crosstalk floor σ² ≈ M/D per HDC theory). Emitted as a
diagnostic canary; NO gate in this carrier. Kill experiment for R3 itself
(pre-registered, separate carrier): CC-OS ON vs OFF at M = 0.2D — lock rate
< 1% with factoring, > threshold without.

## 6. R1/R2/R4 dispositions (carried from the research deliverable)

- R1 Egress Grounding Wall: score eligibility stays fail-closed
  (`score_eligible=false` until provenance + held-out semantic gate);
  temperature scaling applies only to a calibrated logit head — no new
  spend in this carrier.
- R2 Solipsistic Limit Cycles: coupling to Δν exists (Beta-Bernoulli EIG);
  inertness is downstream of goal-discrimination — SFAS is the fix attempt;
  kill = per-action SFAS scores vs observed Δν paired bootstrap CI excludes
  0 (future carrier).
- R4 Tooling Isolation: no new surface; ring ops stay torch ops; the
  OpenAI-compatible bridge (port 8090) is the HF-pipeline surface.

## 7. Artifacts

- `HENRI V2/henri_successor_feature_scorer.py`
- `HENRI V2/tests/contract/test_successor_feature_scorer.py`
- Runner wiring in `HENRI V2/production_arc_run.py` (flag, SFAS block,
  telemetry `sfas` + `superposition_load`)
