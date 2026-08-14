# Phase 8.3 R1 — Representation Discrimination Kill — Pre-Registration

Source packet (untrusted, verified): `G:/My Drive/HENRI_Inbox/Phase8.2.pdf` Lens A/B/C
- PDF SHA-256 `a41433e5…`; text `52242308…` (sealed Phase 8.2).
- Commit anchor: `main` = `2218ec4` (reconciled local = remote). Phase 8.2
  branch `phase/8.2-in-context-functor` @ `ff902d6` NOT merged (falsified engine).

## Accepted sealed facts (2026-08-14)

1. Phase 8.2 K1 measured `true_rank 37/128`, `margin −0.02875` at D=65,536 on
   the **legacy encoder arm** — the probe built `HENRIVisionEncoder` WITHOUT
   `spatial_basis_kind`/`bg_mask`, defaulting to collinear ramps + no masking.
   Evidence `f978428c…`.
2. Production default (main `2218ec4`) is `resolve_spatial_basis()` →
   `incommensurate` + `bg_mask=True` (Phase 7.8 P0-A1, G1 ACCEPTED for
   invertibility: cross-cosine < 0.05, LUT recovery 100%).
3. The FULL mask+ramp variant has **never** been tested against the K1 ranking
   gate (Phase 7.8 tested invertibility; mask-only functor gain +0.033 < +0.05
   was FALSIFIED). R1 is genuinely new as a ranked test.
4. Packet-cited `o_vsa_ingress_tokenizer.py` is DEAD CODE (imported in
   production_arc_run.py:47, never instantiated). Live encoder is
   `HENRIVisionEncoder` (mask applied before phase accumulation, lines 133–145).
5. Demo boundary binding: 20/20 envs `BLOCKED_NO_DEMONSTRATIONS`. R1 is a
   representation-discrimination experiment ONLY; no environment rollout.

## Hypothesis H1-R1

Zero-power foreground masking (color-0 cells excluded from phase accumulation)
combined with independent incommensurate X/Y phase ramps suppresses the static
carrier, so a goal bound via the in-context functor ranks the TRUE macro-option
first for translation, rotation, AND reflection at D=65,536.

## Candidate (default-OFF `HENRI_ARC_REPRESENTATION_R1=1`)

`representation_discrimination_engine.py` — reuses `compile_functor_wave`,
`goal_bind`, `options_from_grid`, `option_waves`, `score_batched` from the
production PSG engine. Adds an encoder constructor that applies:
- foreground zero-power masking (color == 0 excluded BEFORE phase accumulation);
- independent incommensurate x/y ramps (production `spatial_basis_kind=
  "incommensurate"` semantics).

Flag OFF → byte-identical to legacy (max tensor diff 0.0). Flag ON → changed
output (causal engagement proof). Never steps an environment.

## Kills (any fires → R1 FALSIFIED, default-OFF, no rollout, no threshold tuning)

- **K1 (per transform, production-scale CUDA):** for each of the 3 required
  transforms (translation, rotation, reflection), on synthetic single-object
  tasks with KNOWN true macro-option:
  `true_rank ≤ 2` AND `true_margin = Sim(true, Ψ_goal) − max_{a≠true} ≥ +0.05`.
  Independent per transform; ANY single failure falsifies R1.
- **K2 (causal engagement):** flag OFF vs ON must produce max tensor diff ≠ 0
  on identical input (OFF must be exactly 0.0 diff vs legacy baseline).
- **K3 (masking correctness):** a grid whose only foreground cell is color-0
  must NOT zero out the wave (bg_mask must not erase all foreground).
- **K4 (discipline):** `score_eligible=false`, `diagnostic_only=true`, no
  `game.step`, no SANS rows, held-out untouched, no rollout.

## Discrimination matrix (attribution, not gates)

1. Legacy control (collinear, unmasked) — measured already: 8.2 K1 FAIL.
2. Mask-only diagnostic — isolates carrier suppression.
3. Ramps-only diagnostic — isolates collinear-ramp collapse.
4. FULL mask+ramp candidate — ONLY this arm is eligible for R1 acceptance.

## Gates (order)

1. Pre-registration commit (this doc).
2. Engine implementation + contract tests + full local suite.
3. Commit/push `phase/8.3-r1-representation` from `main 2218ec4` (fresh clean
   worktree, zero status before edit).
4. Remote CUDA K1 probe ×3 transforms at D=65,536, GPU exclusive, exact SHA.
5. Seal verdict `R1_ACCEPT` (all 3 pass) or `R1_FALSIFIED` (any fail) +
   receipt. Rollout stays BLOCKED regardless.

## Telemetry (event `R1_K1_CUDA`)

`transform, true_rank, true_margin, sim_true, sim_best_false, goal_sim_obs,
num_options, agreement_max_abs_diff, device, sha256(result), score_eligible=false`.

## Flag

`HENRI_ARC_REPRESENTATION_R1=1` (default OFF). Engine never steps an
environment; no runner wiring; no rollout authorization.
