# Phase 8.13 — Amplitude-Preserving Complex Ingress (pre-registered, 2026-08-15)

Blueprint: `docs-HENRI_V2_PHASE_8_12_POSTMORTEM_AND_PHASE_8_13....pdf.pdf`
(local `C:/tmp/p813_postmortem.pdf`, SHA-256 `5e435cd95ccd8f25a8e95146efa3c68d25626348bdacd56932ef0cdef265afbf`)
Branch base: `feat/amplitude-aware-complex-transition` from sealed 8.11 tip
`4db6916` (per blueprint §3.1 Step 1). main UNTOUCHED @ `2218ec4`.

## Mechanism (from 8.12 evidence, NOT speculation)
8.12 FALSIFIED unit-modulus ingress: per-element unit-modulus superposition
collapses distinct grids onto the shared carrier subspace (cos 0.9999).
The discriminative channel is AMPLITUDE (foreground occupancy), which
unit-modulus normalization discards. Phase 8.13 restores amplitude:
`z = sum_i c_i * exp(j*(x_i*omega_x + y_i*theta_y))`, background masked
(a=0), un-normalized complex wave in C^D. The 8.11 NativeComplexWaveTransition
is amplitude-INVARIANT (forward_complex = Hadamard by unit phasor;
update_phase_complex = torch.angle residual), so amplitude waves feed it
natively — NO acos lift.

Key math: a whole-grid translation (dx,dy) factors out of the superposition:
z_{t+1} = exp(j*(dx*omega_x + dy*theta_y)) ⊙ z_t  (per-element diagonal
phase rotation) — EXACTLY the operator 8.11 fits (Fourier Convolution
Theorem, Plate 1995). So translations are exact 1-step fit, any grid size.

## Gates (pre-registered)
- G1 DISTINCTNESS: distinct-grid complex cosine similarity < 0.0100 at
  D=65,536 (target 0.00036 class from 8.12 G3 legacy control).
  G1_LOCAL (mechanism floor, d=512): max distinct cos < 0.150 — strict
  0.0100 gate is D-dependent (E|cos| ~ 1/sqrt(D): 0.0039 at D=65,536 but
  ~0.044 at d=512); dimension-aware gate split, no post-hoc tuning.
- G2 FIT: step-3 online phase fit loss L_sagnac = mean|angle residual| on
  held-out translation pairs <= 0.0500 (exact 1-step expected, ~1e-7).
  Shared delta per action; stable fixed NONZERO action wave (8.11 lessons).
- G3 LATENCY: ingress forward + forward_complex + egress <= 1.0 ms at
  D=65,536 (50 iters, 30x30 grid, up to 900 fg pixels).
- G4 TASK PROGRESS: live level progress > 0.0 on 20 ARC environments —
  PRE-REGISTERED EXPECTED BLOCKED (standing BLOCKED_NO_DEMONSTRATIONS
  20/20; arcade examples: None observed lf52/tn36/sc25). Runner probes
  demo ingress and records BLOCKED_NO_DEMONSTRATIONS — never fabricates.
- G5 DEFAULT-OFF: production EFEPlanner/LowRankCoupledTransition path
  byte-identical; new module additive-only; no planner modification.
- DONE marker rc=1 if ANY arm fails (aggregated, learned lesson).

## Deviations (documented, pre-commit)
- D1 (phantom CLI #8): `o_vsa_ingress_tokenizer.py --mode
  amplitude_ingress_test` does not exist (no --mode arg). Replaced by
  dedicated CUDA runner + contract tests (8.11/8.12 precedent).
- D2 (phantom CLI #9): `production_arc_run.py --mode phase813_benchmark`
  does not exist (only --envs/--steps). Replaced by dedicated runner;
  production runner NOT modified.
- D3 (VALID path): `scripts/agentic_event_store.py` EXISTS at repo root
  (8.13-D ledger target valid; NOT under HENRI V2/). Lever 8.13-D is
  P1/DEFERRED this phase (no schema change without approval); G1-G3 scope.
- D4 (carrier init): blueprint sketch uses randn(D)*pi / randn(D)*e. Per
  8.12 evidence (full-band unit-modulus fails gates) and reproducibility,
  carriers = deterministic INCOMMENSURATE frequencies:
  omega_x[d] = 2*pi*(d+1)*sqrt(2), theta_y[d] = 2*pi*(d+1)*sqrt(3)
  (irrational multiples; same class as 8.8 band-limited fix). The
  amplitude channel carries occupancy; phase decorrelation ~1/sqrt(D).
- D5 (G4 scope): live 20-env ARC evaluation requires demos (blocked);
  runner records the preflight probe result only. No pseudo-demos.

## Kill criteria
- G1 fails at D=65,536 (max distinct cos >= 0.0100) -> mechanism KILL,
  seal branch, no promotion.
- G2 fails (fit loss > 0.05 held-out) -> transition-coupling KILL.
- G3 fails (>1.0 ms) -> perf KILL (non-mechanism).
- G4 = BLOCKED_NO_DEMONSTRATIONS recorded (expected, not a kill).
- G5 regression -> immediate stop, revert.

## LOCAL OBSERVATION (d=512, contract suite 9/9, 2026-08-15) — G1 KILL ESTABLISHED
Exact math, D-independent:
1. COLOR-BLIND: amplitude-weighted color is cosine-scale-invariant:
   cos(3z, 6z) = 1.0 EXACTLY for same-position grids (any carriers).
   ARC color is a core object property -> blueprint G1 < 0.0100 unreachable.
2. SHARED-SUPPORT COHERENCE: position-carrier superposition adds shared
   cells coherently: B-ring (8px) vs C-line (3px, 3 shared cells) ->
   cos = 6/sqrt(96) = 0.6113 (D cancels; finite-D correction O(1/sqrt(D))).
3. LEGACY CONTROL DOMINANCE (decisive): 7.3/7.4 encoder (incommensurate +
   bg_mask) phase-encodes color (pc term): 0.0000 (color pair), 0.0033
   (shared pair) — strictly better on BOTH hard pairs. The discriminative
   channel is COLOR PHASE, not amplitude. Amplitude occupancy alone
   (8.13-A) is a strict subset of what 7.3/7.4 already achieves.
SURVIVING (not promoted): G1X translation exactness (~1e-7) holds for any
position-carrier superposition; G2 fit (8.11 transition amplitude-invariant)
holds on amplitude waves. Both are properties of the 8.11 transition, not
of the amplitude ingress lever.

## VERDICT — SEALED KILL (OBSERVED 2026-08-15)
G1 KILL confirmed at D=65,536 (see runner evidence). G3 legacy control
dominance confirmed at scale. G2/G1X survive as transition properties.
G4 = BLOCKED_NO_DEMONSTRATIONS (expected). G5 default-OFF. main untouched.
NO promotion. Redirect: COLOR-PHASE ingress (7.3/7.4 pc term) wired to the
8.11 complex transition at the planner boundary — the amplitude channel
was the 8.12 error; the color-PHASE channel is the verified one.
