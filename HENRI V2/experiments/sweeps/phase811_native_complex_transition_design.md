# Phase 8.11 — Native Complex Wave Space Transition (PRE-REGISTRATION)

Branch: `feat/native-complex-wave-transition` from main `2218ec4`.
Source blueprint: `HENRI-POSTMORTEM-2026-08-PHASE8.10-FINAL`
(`G:/My Drive/HENRI_Inbox/Phase 8.10 Diagnostic & Phase 8.11 Strategy.pdf`,
SHA-256 `3200de6035be17e6e77dee1e2bd1d106`..., drive-inbox copy).
Approval: user 2026-08-15 (Step 2 of the Phase 8.10/8.11 instruction).
Phase 8.10 precedent: kill OBSERVED at D=65,536 (commit `8f7d207`, branch
`feat/diagonal-transition-production-wiring`); sealed, NO promotion.

## Objective
Resolve the wave-type contract conflict that killed Phase 8.10: the production
transition operates on REAL [8192, 8] per-block L2-normalized waves, where
diagonal phase rotation is unidentifiable within the online budget
(acos(cos(phi+delta)/c) != phi+delta). Phase 8.11 introduces a NATIVE COMPLEX
transition operator: the world-model latent transition executes in C^D as
per-element unit-modulus phasors; real conversion occurs ONLY at the egress
boundary. Goal: O(D) exact phase learning within the 3-step online fit budget.

## Honest scope (recorded BEFORE code)
- ARC-AGI-3 SOTA scoring remains BLOCKED (standing: 20/20 envs
  BLOCKED_NO_DEMONSTRATIONS). This phase does NOT claim ARC scores.
- This phase delivers: (1) NativeComplexWaveTransition module (default-OFF),
  (2) EFEPlanner / orchestrator / runner wiring (flag default OFF), (3)
  contract tests, (4) CUDA matrix at D=65,536 on RTX 5090, (5) honest
  accept-or-kill verdict with the exact remaining chain to SOTA recorded.
- The encoder STILL emits real per-block-normalized waves. The complex
  transition therefore must lift real -> complex at its boundary; that lift is
  lossy for production real waves (Phase 8.10 root cause). This phase tests
  BOTH: (a) the native-complex mechanism exactly (synthetic phasor
  trajectories, accept gate), and (b) the production real-wave transfer
  (improvement gate, expected FAIL -> records the next lever: complex-native
  state generation at the encoder, Phase 8.12).

## Mechanism
State space: Psi_t in C^D, per-element unit modulus (FHRR convention; NO
vector L2 normalization — 8.9 lesson #3: vector norm floors Sagnac at 1-1/D).

    transition:  Psi_{t+1} = Psi_t * exp(j * Theta_a)      (Hadamard, exact)
    fit:         Theta_a += lr * angle(Psi_next * conj(Psi_t))
                 (closed-form angle residual; exact 1-step for any diagonal
                 phase rotation; Fourier Convolution Theorem, Plate 1995)
    egress:      x_real = per_block_normalize(Re(Psi))     [8192, 8] real
                 (applied ONLY at the readout boundary)

Production interface (planner compat):
    forward(state_wave [B,8] real, action_wave [B,8] real) -> [B,8] real
    lift:  z = exp(j * acos(clamp(w)))     # lossy for production real waves
    rotate + egress-projection back to real.
    Indexing: action-wave cosine fingerprint (deterministic decoder engrams),
    bounded to num_actions; fail-closed vs learnable_actions.

## Wiring (causal-consumer chain)
1. `complex_phase_transition.py` (NEW): `NativeComplexWaveTransition`.
2. EFEPlanner.__init__: `use_complex_transition: bool = False`; when True,
   self.transition = NativeComplexWaveTransition (same interface:
   forward(state, action) -> [B,8] real; .num_blocks, .block_dim, .d;
   .rank=0; ._retract() no-op). Fail-closed ValueError when
   learnable_actions=True (fingerprint indexing needs deterministic waves).
3. update_transition_sgd: branch — complex path calls the adapter's complex
   closed-form update (never field_V/field_W/block_residual).
4. train_transition_batch: branch — complex closed-form batch fit (same
   signature, returns pre-fit loss).
5. field_channel_wave / load_field_channel_wave: complex branch packs /
   restores the action_phases buffer [num_actions, D].
6. HenriSwarmOrchestrator.__init__: pass-through kwarg.
7. production_arc_run.py: `HENRI_ARC_COMPLEX_TRANSITION` env flag (default
   OFF) -> orchestrator -> planner.
Flag OFF = production path byte-identical (adapter never constructed).

## Gates (pre-registered, no post-hoc tuning)
- G1 NATIVE-COMPLEX EXACTNESS (accept gate): 32 synthetic per-element
  unit-modulus phasor trajectory pairs (random theta_true per action), fit
  via closed-form angle residual in <= 3 fit calls; PRODUCTION real-metric
  Sagnac on egress-projected real waves <= 0.05. (Resolves the 3000-step
  real-domain failure.)
- G2 PRODUCTION REAL-WAVE TRANSFER (improvement gate): PRODUCTION real waves
  from the live ARC encoder (HENRIVisionEncoder.encode_spatial_grid on
  synthetic grids -> [8192, 8] per-block normalized; the 8.8 carrier ingress
  module lives on the sealed 8.8 branch, NOT main — production-faithful
  source is the live encoder); lift -> rotate -> egress; held-out pre/post
  real-metric Sagnac; gate post < pre - 0.02.
  EXPECTED FAIL (lossy acos lift + egress sign folding) -> records the next
  lever (complex-native encoder). A PASS would be a bonus (not required).
- G3 EGRESS CONTRACT: egress returns real [8192, 8] per-block unit (norm err
  <= 1e-4), dtype float32, finite. (Blueprint's tautological
  ||x_egress - Re(Psi)|| < 1e-6 gate rejected — egress IS Re; replaced with
  the meaningful contract + phase-recovery diagnostic.)
- G4 DEFAULT-OFF BYTE IDENTITY: flag OFF -> legacy LowRankCoupledTransition;
  contract asserts forward equality with control; full suite passes.
- G5 LATENCY: forward_complex at D=65,536 <= 1.0 ms on RTX 5090.
- WIRE: EFEPlanner(use_complex_transition=True) constructs the module;
  select_action + train_transition_step run at scale without device errors.
- G6 HONEST BOUNDARY: ARC-AGI-3 SOTA stays BLOCKED_NO_DEMONSTRATIONS
  (recorded, not faked).

## Kill criteria (pre-registered)
- G1 fails (native complex cannot reach <= 0.05 in <= 3 steps) -> the complex
  mechanism itself is falsified at this scale; module stays default-OFF;
  kill lesson; NO promotion.
- G2 fails exactly as expected -> NOT a kill of the mechanism; it is the
  recorded boundary: production real waves cannot be lifted losslessly.
  Report as OBSERVED boundary; next lever = Phase 8.12 complex-native encoder.
- G3/G4/G5/WIRE fail -> wiring defect; fix and rerun (infra class).
- NO promotion to main regardless; sealed phases stay sealed.

## Deviations from blueprint (documented)
1. Self-test step 3 (`wave_jepa.py --mode native_complex_test`) is the
   phantom-CLI family (3x confirmed non-existent). Replaced by contract
   tests + CUDA matrix runner (this phase's standard).
2. No vector L2 normalization in forward_complex (8.9 lesson #3); per-element
   unit modulus is preserved exactly by Hadamard with unit phasor.
3. Egress gate redefined (see G3) — the blueprint's literal gate is
   tautological.
4. Zone C provenance logging (Lever 8.11-D, P1) deferred: out of the
   transition gates; recorded for a later phase.
5. Branch base: fresh from main @ 2218ec4 (blueprint-compliant). The sealed
   8.9/8.10 modules live on their sealed branches and are NOT ported here;
   the complex module is standalone (blueprint design).

## Protocol deviation addendum (2026-08-15, docs-HENRI_V2_PHASE_8_11_VERIFICATION_AND_SEAL_PRO....pdf, SHA 7911b094...)
1. PHANTOM-CLI 5th confirmation: the protocol's `tests/test_complex_phase_transition.py`
   (Step 1) and `gpu_verification_suite.py --mode phase811` (Step 3) DO NOT EXIST in
   the worktree. Replaced with REAL artifacts mirroring the 8.10 precedent:
   - `HENRI V2/tests/contract/test_phase811_native_complex_transition.py` (9 tests:
     G1a identity, G1b native accept, G2 real-lift boundary EXPECTED FAIL, G3 egress,
     G4 default-OFF, G6 mutual exclusion, G7 fail-closed, WIRE select/train, WIRE roundtrip)
   - `HENRI V2/experiments/performance/phase811_native_complex_cuda_check.py`
     (D=65,536 RTX 5090 matrix; JEPA_DM_OUT=/tmp/p811_matrix_d65536.json)
2. G1b test-data fix (test-design, not mechanism): per pre-registration G1 uses
   NATIVE-DOMAIN per-element unit-modulus phasor pairs z_{t+1}=z_t*exp(j*delta) with
   ONE SHARED delta (single action rotation), separate alpha seeds for states; held-out
   REAL-egress Sagnac <= 0.05 in <= 3 closed-form calls. OBSERVED local d=512: PASS.
   (First draft fed real L2-normalized waves — that is the G2 boundary, not G1.)
3. G2 boundary asserted explicitly: real-wave lift->rotate->egress EXPECTED FAIL
   (kill = post >= pre - 0.02); NOT a mechanism kill; next lever = complex-native
   encoder at ingress.
4. Stable nonzero action wave used in G1b/G2 fits (all-zeros action triggers
   fingerprint insertion on every call, spreading the fit across buffers).
5. Local results: contract 9/9 PASS; full suite 414 passed / 1 skipped (baseline
   unchanged). Committed as Step 2; remote CUDA matrix = Step 3.
