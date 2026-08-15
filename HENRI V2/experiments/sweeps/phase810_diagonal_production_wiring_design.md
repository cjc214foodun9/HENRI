# Phase 8.10 — Diagonal Transition Production Wiring (PRE-REGISTRATION)

Branch: `feat/diagonal-transition-production-wiring` from main `2218ec4`.
Base modules: sealed 8.9 `henri_frequency_domain_transition.py` (SHA
`362851629c...`), sealed 8.8 `henri_spatial_carrier_ingress.py` (SHA
`ff6a21282d...`) — ported byte-identical.

## Objective
Wire the sealed frequency-domain diagonal phase rotator into the PRODUCTION
transition path default-OFF, with external-outcome grounding evidence, and
honest ARC-AGI-3 status.

## Honest scope (recorded BEFORE code)
- ARC-AGI-3 SOTA scoring is BLOCKED (standing: 20/20 envs
  `BLOCKED_NO_DEMONSTRATIONS`; no provenance API; demos never fabricated).
- This phase delivers: (1) default-OFF diagonal transition in the production
  path (EFEPlanner), (2) closed-form training branch, (3) CUDA-verified
  functional evidence on analytic translation + physical control envs,
  (4) the exact remaining chain to SOTA recorded, un-faked.

## Mechanism
Production interface is REAL [num_blocks=8192, 8] per-block unit waves:
    forward(state_wave [B,8], action_wave [B,8]) -> [B,8] real
Sealed 8.9 is COMPLEX [D] phasors with per-action phase vectors. Bridge
(real -> phase, deterministic, gradient-flow-preserving):
    z_d = exp(j * arccos(clamp(w_d)))          # real wave -> unit phasor
    theta_d = pi * clamp(a_d)                  # action wave -> phase
    pred_d = cos(arccos(s_d) + theta_d)        # rotated real output
    out = per-block renormalize(pred)
Identity at zero phase: cos(arccos(s)) = s exactly -> forward(s, zero) == s.
Learned correction per action: phase_correction[a_idx] (nn.Parameter
[num_actions, D], zero init = identity); lazy index assignment by action-wave
cosine fingerprint (index-free, bounded to num_actions).
Closed-form training branch (8.9-B port): for each action index,
    Delta_theta = circular_mean(angle(z_tp1 . conj(z_t) . conj(phi_a)))
    correction[a] += lr * Delta_theta, lr = 1.0
Sagnac measured with the PRODUCTION metric: 1 - dot(pred, actual) /
(|pred| |actual|) on real flattened waves.

## Wiring (causal-consumer chain)
1. EFEPlanner.__init__: `use_diagonal_transition: bool = False`; when True,
   self.transition = FrequencyDomainDiagonalAdapter (same interface:
   forward(state, action) -> [B,8] real; .num_blocks, .block_dim, .d;
   .bind used only by the legacy EDMD branch).
2. update_transition_sgd: branch — diagonal path calls the closed-form
   adapter update (never touches field_V/field_W/block_residual).
3. train_transition_batch: branch — diagonal closed-form batch fit instead of
   the dual/Woodbury EDMD solve; same signature, returns pre-fit loss.
4. field_channel_wave / load_field_channel_wave: diagonal branch packs/restores
   the phase-correction buffers (checkpoint persistence).
5. HenriSwarmOrchestrator.__init__: pass-through kwarg.
6. production_arc_run.py: `HENRI_ARC_DIAGONAL_TRANSITION` env flag (default
   OFF) -> orchestrator -> planner -> adapter.
Flag OFF = production path byte-identical (adapter never constructed).

## Gates (pre-registered, no post-hoc tuning)
- G1 identity: forward(s, zero-phase action) -> cos-sim 1.0 vs s.
- G2 analytic recovery: CC-OS carrier-encoded translation triples; after
  closed-form fit, held-out REAL-metric Sagnac < 0.30 (8.8-C gate parity).
- G3 convergence: batch Sagnac < 0.05 within <= 3 fit calls on synthetic
  translation triples (kill: arccos bridge wrap ambiguity falsifies the
  real-domain bridge -> FALSIFIED verdict, no promotion).
- G4 default-OFF byte identity: flag OFF -> legacy path output unchanged
  (contract asserts forward equality); full suite passes.
- G5 latency: diagonal forward <= 1.0 ms at D=65,536 on RTX 5090.
- G6 grounding arms: physical-env episode with carrier ingress (Pendulum/
  CartPole ODE envs, state_to_wave bridge via carrier ingress default-OFF):
  report external frame deltas OBSERVED; ARC arcade grounding
  BLOCKED_NO_DEMONSTRATIONS (recorded, not faked).

## Kill criteria
- G3 fails -> arccos real-domain bridge FALSIFIED; keep module default-OFF;
  record kill lesson; report honestly. No promotion either way (sealed phases
  stay sealed; main untouched).

## SEALED VERDICT (2026-08-15) — KILL CONFIRMED, default-OFF, NO promotion
Local CPU contract evidence (deterministic seeds, NB=64, D=512):
- G1 identity: PASS (cos 1.0 at zero phase).
- G2 carrier regime (8.8 CC-OS production real waves): pre 0.0342 -> post
  0.0344 (held-out). Absolute floor < 0.30 PASSES, but the IMPROVEMENT gate
  (post < pre - 0.02) FAILS — the learned correction contributes ~nothing;
  the pre-fit score comes from carrier adjacency similarity, not the learned
  rotation. Identity-degeneracy trap avoided (the improvement gate was the
  discriminator).
- G3 learning budget (synthetic diagonal rotation, <= 3 fit calls = 75 steps
  @ 0.04): post 0.2852 >= 0.05 -> FAIL. Mechanism only converges asymptotically
  (3000 steps @ 0.1 -> 0.00013), far beyond the production per-transition
  budget (1 step @ lr 0.05).
- G4 default-OFF byte identity: PASS (legacy LowRankCoupledTransition;
  adapter never constructed). Full suite 414p/1s, no regression.
- ROOT CAUSE (kill lesson): diagonal phase rotation is exact only on
  per-element unit-modulus complex phasors (8.9 analytic regime). The
  production interface is REAL [8192, 8] per-block L2-normalized waves; the
  arccos projection sign-folds and norm-scales the phase
  (acos(cos(phi+delta)/c) != phi+delta), making the loss landscape shallow and
  the rotation unidentifiable within the production budget. The 8.9 exactness
  does NOT transfer to the production wave type.
- REMEDY PATH (HYPOTHESIS, needs fresh protocol + approval): run the
  transition in the COMPLEX phasor domain (per-element unit-modulus) with a
  complex->real projection only at egress, or change the production wave
  convention to preserve phase — a load-bearing representation change.
- ARC-AGI-3 SOTA: STILL BLOCKED (20/20 envs BLOCKED_NO_DEMONSTRATIONS;
  standing). No ARC scores were fabricated; grounding arms remain blocked.
- Wiring delivered default-OFF and contract-tested: FrequencyDomainDiagonalAdapter
  (append-only to sealed 8.9), EFEPlanner branches (init / train_transition_step /
  train_transition_batch / field_channel_wave / load_field_channel_wave),
  orchestrator pass-through, HENRI_ARC_DIAGONAL_TRANSITION runner flag,
  fail-closed guard vs learnable_actions.
- Evidence: contract test run 2026-08-15 (9/9), full suite 414p/1s; this
  document = sealed verdict record.

## Deviations (pre-registered)
- Bridge uses arccos/pi-linear mapping (production real interface), not the
  8.9 complex-index interface; sealed 8.9 classes untouched (append-only
  adapter in the same file).
- No vector Normalize (8.9 deviation #3 preserved): per-block renormalize
  only, matching production output convention.
- Single-step online update (train_transition_step) honors the runner's
  lr (default 0.05): damped closed-form update for noise robustness.
  EXACT 1-step convergence is reserved for fit_batch at lr=1.0 (G2/G3).
- Constraint penalty channel is inactive in diagonal mode: the diagonal
  branch never sets axiom_constraint (legacy EDMD-only); constraint_penalty
  returns None -> penalty 0.0, no hard-reject in diagonal mode.
