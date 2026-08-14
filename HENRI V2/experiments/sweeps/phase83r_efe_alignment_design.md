# Phase 8.3r — EFE Alignment Experiment — Pre-Registration

Packet: `G:/My Drive/HENRI_Inbox/8.3 remediation.pdf` ("Master Architectural
Handoff & Continuity Brief + Epistemological Audit / Phase 8.3 R1 Postmortem")
- PDF SHA-256 `75e86eca759bffc426163adb5f9a0f0184c79f82fa99966e2490a6b61b6b7551`
- bytes 142,055; pages 5; text SHA-256 `db9b26446c9a462b6857a0292af4e44243acffe6dbf1f985c021c245e7acef0c`
- sibling `Phase 8.3 remeed.pdf.pdf` (SHA `ed63d352...`, 4 pages, text `2bac9a68...`) is NOT identical — not authorized, no action.

Base: `main` = `2218ec4` (remote reconciled). Branch `phase/8.3r-efe-alignment`
from clean `2218ec4` in external worktree `C:/tmp/henri-83r-efe-wt`.
R1 branch `phase/8.3-r1-representation` @ `7a3c7b5` is UNPROMOTED and its
verdict `R1_FALSIFIED` (sealed `d1edd9b3`) stays unpromoted; its engine is
NOT imported here. No R1-branch dependency.

## Packet requirements map (authoritative)

1. Carrier-dominance pathology SOLVED at similarity level (mask+ramp: translation
   rank 1 margin +0.2175; rotation rank 1 margin +0.3247) — measured, sealed.
2. NEW pathology to remediate: production EFE path ranked the true candidate
   last (`efe_true_rank` 10/10 translation) while sim-K1 ranked it first.
3. Packet root causes (to be tested, NOT assumed):
   - (i) epistemic term overpowering pragmatic attraction for large displacement;
   - (ii) single-pixel diagnostic boundary instead of the canonical 11-axiom
     Zone C baseplate;
   - (iii) sign-convention conflict turning the pragmatic goal attractor into
     an EFE repeller.
4. Actionable: issue ONE default-OFF experiment re-calibrating the EFE planner
   cost function so candidates with Rank-1 similarity achieve Rank-1 minimal EFE.
5. Reflection parity requires non-commutative Clifford algebra — OUT OF SCOPE
   (packet states this; sim-K1 reflection already sealed FALSIFIED +0.0129).
6. No rollout authorization anywhere in the packet; no demo pairs exist
   (20/20 BLOCKED, sealed `325a2db5` + `ff5fbec7`); synthetic known-transform
   pairs are software-integrity verification ONLY.

## Deterministic audit facts (live code, main 2218ec4)

- Selection: `HenriSwarmOrchestrator.plan_action` → `EFEPlanner.select_action`
  → `score_actions` sorted ascending by `efe` (argmin wins).
- `efe = pragmatic_weight*pragmatic - epistemic_weight*epistemic
  + lam*penalty - external_eig_weight*eig - external_task_weight*resonance`
  (efe_planner.py:939-943). External terms OFF in default config.
- `pragmatic = min_a (1 - <pred, axiom_a>) + lambda_goal*goal_distance
  - beta_pragmatic*max_resonance` (efe_planner.py:639-676). Lower = closer to
  an axiom. Resonance OFF with empty preference store.
- `epistemic = entropy * novelty` (>=0), SUBTRACTED from EFE: larger
  information gain lowers EFE (preferred). efe_planner.py:752-808.
- `constraint_penalty` = RMS off-manifold residual (None pre-first-fit →
  0.0); hard-reject threshold 0.38 (production default).
- T4 explore gate (select_action:1026-1035): if `loss_ema > accuracy_floor`
  pick max-epistemic action instead of min-EFE. A fresh untrained planner is
  in the explore regime — the R1 probe bypassed select_action (raw argsort),
  so it did NOT measure the production selection path.
- Production constants (production_arc_run.py): `LAMBDA_GOAL` default **0.0**
  (goal-distance term OFF), `BETA_PRAGMATIC` 1.0, `LAMBDA_CONSTRAINT_MAX` 5.0,
  `CONSTRAINT_REJECT_THRESH` 0.38, `USE_ZONE_C_AXIOMS` default 0, `SCALE`
  num_experts=1024 d_model=65536 r_rank=16 num_blocks=8192.
- R1 probe deviation (sealed): `lambda_goal=0.5` (NOT production default),
  single-pixel diagnostic boundary, fresh untrained transition. EFE inversion
  is therefore CONDITIONAL on those probe settings.
- Canonical boundary: `load_boundary_axioms(dsn=None, env_file=None,
  num_blocks=8192)` → `[11, 8192, 8]` + summaries; integrity-verified
  (per-block unit norm, 2000-dim projection cosine); fail-closed
  `BoundaryAxiomLoadError`. DSN via `ZONE_C_AXIOM_ENV_FILE` or
  `ZONE_C_PROD_DSN` (env file keys ending in `DSN`).
- Live encoder: `HENRIVisionEncoder(spatial_basis_kind="incommensurate",
  bg_mask=True)` — mask applied BEFORE phase accumulation (henri_vision_encoder
  ~133-145); this is the production default from `arc_spatial_basis`.
  `o_vsa_ingress_tokenizer.py` is dead code (imported, never instantiated).
- PSG on main: `compile_functor_wave`, `goal_bind` (requires w_task),
  `MacroOption`, `_apply_option_to_grid`, `build_macro_options` (kinds:
  identity/translate/rotate/color — NO reflection), `option_waves`,
  `score`/`score_batched`.

## Experiment (default OFF, flag `HENRI_ARC_EFE_ALIGNMENT=1`)

Module `efe_alignment_experiment.py` + CUDA runner
`experiments/performance/efe_alignment_cuda_check.py`. Reuses production
encoder, functor, goal_bind, options, EFE kernels, and the production
selection path. No surrogate planner, no game.step, no rollout.

Arm matrix (all 4 arms run; verdict driven by arm A only; B-D attribution):
- A: canonical 11-axiom boundary + `lambda_goal=0.0` (PRODUCTION config) — PRIMARY
- B: canonical 11-axiom boundary + `lambda_goal=0.5` (R1 probe config)
- C: single-pixel boundary + `lambda_goal=0.0`
- D: single-pixel boundary + `lambda_goal=0.5` (R1 reproduction)

Per arm, per transform (translation `translate(dx=1, dy=0)`, rotation
`rotate(90)`): 3 synthetic prompt pairs via the same option path; compile
functor; goal_bind(state); 8 macro-options (identity + 4 translations + 3
rotations; reflection EXCLUDED — packet out-of-scope, main's builder has no
reflection kind); production `score_actions` (full decomposition: pragmatic,
epistemic, constraint_penalty, goal_distance, efe, rejected) +
`select_action` (explore/exploit recorded) + loop/vmap agreement.

## Pre-registered gates (NO threshold tuning after observation)

- G1 boundary integrity (arm A/B): 11 axioms load, per-block norm in
  [1-1e-6, 1+1e-6], projection cosines recorded. Fail → BLOCKED_INFRASTRUCTURE.
- G2 decomposition consistency: recomputed EFE per candidate from logged
  terms agrees with logged EFE <= 1e-4. Fail → BLOCKED_INFRASTRUCTURE.
- G3 loop/vmap identity: max abs diff <= 1e-5. Fail → BLOCKED_INFRASTRUCTURE.
- G4 ALIGNMENT (arm A, both transforms): true option's EFE rank == 1 AND
  EFE margin (best_false - true) >= 1e-3. PASS → EFE_ALIGNMENT_PASS
  (single-pixel boundary + lambda_goal were the confounds; arms B-D reported
  for attribution). FAIL → EFE_ALIGNMENT_FALSIFIED (production cost function
  cannot rank rank-1-similarity candidates first even with canonical
  boundary; next lever = bounded cost-form experiments, separate approval).
- G5 production selection consistency (arm A, reported, not gating): whether
  `select_action` (explore or exploit mode) chose the true option; explore
  mode documented — G4 evaluates the cost-function ranking, G5 the live path.
- G6 default-OFF identity: flag OFF → FEATURE_DISABLED, no Zone C connection,
  no allocation, no EFE computation.
- G7 eligibility: every return path emits schema_id, score_eligible=false,
  diagnostic_only=true, authorizes_rollout=false.
- G8 no `game.step(` anywhere in new source.

## Deliverables

- sealed result JSON per arm (decomposition table, ranks, margins, chosen
  action, explore flag) + log + receipts;
- verdict seal committed on the branch (unpromoted);
- promotion to main only if G4 PASS and a separate release candidate passes
  the full gate chain (local suite + isolated remote CUDA suite + approval +
  fast-forward + clean deployment reconcile). FALSIFIED or BLOCKED →
  evidence seal, no promotion, no rollout.
