# Phase 8.1 Zero-Shot Symmetry Self-Consistency Functor — Pre-Registration

Source packet (untrusted, verified): `G:/My Drive/HENRI_Inbox/docs-phase8_1_preflight_postmortem_and_roadmap.md.pdf.pdf`
- PDF SHA-256 `6d8c53838c76c7506ca5edf96e6b6be3ca8a33955bd2d6f2d873c3e4ac70900f`
- extracted text SHA-256 `71ff771696ad5845ba90566bc35ebf8f09c69f52a4e0bddc1c2c2644f6404815`, 3 pages
- dated 2026-08-14; branch base `2218ec4` (main, FF-promoted after CUDA 406p / local 405p/1s / smoke boundary pass).

## Accepted sealed facts

1. 16/16 frozen envs (12 discovery + 4 held-out) expose `BLOCKED_NO_DEMONSTRATIONS` (evidence `325a2db5…`).
2. Demo-conditioned PSG (`HENRI_ARC_PSG=1`) remains blocked at the demo boundary; `W_task` cannot be compiled. The zero-shot mode below is a SEPARATE typed policy branch; `plan()`'s fail-closed Stage-1 contract is unchanged.
3. Smoke at `2218ec4`: PSG armed → `BLOCKED_NO_DEMONSTRATIONS` ×4 → EFE control arm; telemetry sealed (`79c9de3c…` log `1979358c…`).

## Hypothesis H1 (packet Stage 1)

D₄ orbit-average goal wave provides a directional pragmatic gradient:

    Ψ_goal = Normalize( (1/8) Σ_{g∈D₄} ρ(g) Ψ_obs )

enabling demo-free macro-option rollouts (Stage 2) that can eventually produce external progress (P(Δν≥2) ≥ 0.15).

## Vacuous-gate defect in the packet (pre-registered)

For any generic (asymmetric) unit grid wave at D=65,536, orbit vectors are near-orthogonal, so
`‖mean‖ ≈ 1/√8 ≈ 0.354` and `sim(Ψ_goal, Ψ_obs) ≈ 0.354` BY CONSTRUCTION. The packet's Stage-1 target `goal_sim > 0.30` is therefore trivially satisfied for every grid — internal coherence, not task grounding. For symmetric grids the orbit collapses and sim → 1.0. The 0.30 gate is INVALID as a pass criterion; it is replaced by discriminative kills below.

## Kills (pre-registered, any fires → H1 FALSIFIED)

- **K1 (discriminative ranking, synthetic):** 5 synthetic grids with KNOWN transforms (translate(1,0), translate(0,1), rot90, flip_h, color-swap). Compute Ψ_goal; rank candidate option waves `{g·Ψ_obs}` (8 D₄ + 4 translations). PASS requires: true g* ranks #1 AND margin over #2 ≥ 0.02 AND applying top-1 option reproduces the known target grid with cell accuracy ≥ 90%. Otherwise the goal wave is a constant rescaling, not directional.
- **K2 (flatness, CORRECTED):** EFE-score spread over all candidates `max − min < 1e-3` → uniform scoring → FALSIFIED (no candidate discrimination). NOTE: goal-cosine spread over the 8 D₄ members is ~0 BY CONSTRUCTION (orbit mean is equidistant from every orbit member — group-sum invariance), so cosine spread is recorded as telemetry, never as a kill criterion.
- **K3 (engagement, discovery 12):** zero-shot mode must engage (produce legal ACTION6 + payload) on ≥ 1 step per env AND produce ≥ 1 strict frame change (changed_cells > 0) vs control. Zero engagement or zero frame change = `NOT_EXERCISED`, never PASS.
- **K4 (vmap fidelity):** vmap-loop agreement ≤ 1e-6 (measured 2.384e-07 at D=65,536).
- **K5 (discipline):** `score_eligible=false` throughout; SANS rows stay 0; Stage 3 (SANS≥50, action-head SGLD) NOT authorized by this packet.

## Gates (order)

1. Pre-registration commit (this doc).
2. Local suite (contract tests: vacuous-bound proof, symmetric-orbit collapse, deterministic ranking, no game.step, flag OFF no-allocate).
3. Remote CUDA probe: synthetic K1/K2 control at production scale + 12-env K3 engagement probe (discovery only, held-out untouched), frozen, payload channel on, `diagnostic_only=true`.
4. Evidence seal: kill verdicts, freshness table (main / newest verified / deployed / active), `score_eligible=false`.

Acceptance = K1 + K2 + K4 + K5 pass AND K3 engages with ≥ 1 frame change per env. Stage-2 external target (P(Δν≥2)≥0.15) is a SEPARATE milestone requiring a full rollout approval; this packet only authorizes the probe.

## Telemetry (event `PSG_ZERO_SHOT`)

`status, goal_sim_obs, orbit_norm_raw, sim_spread, top_option, true_rank (synthetic), margin, cell_acc, changed_cells, functor_status=ZERO_SHOT_SYMMETRY, goal_source=SYMMETRY_ORBIT, diagnostic_only=true, score_eligible=false`.

## Flag

`HENRI_ARC_PSG_ZERO_SHOT=1` (default OFF), orthogonal to `HENRI_ARC_PSG` (demo-conditioned). Requires `HENRI_ARC_ACTION_PAYLOADS=1` (action completeness); else typed `BLOCKED_PAYLOAD_CHANNEL`.
