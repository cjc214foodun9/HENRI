# Phase 8.12 — Complex-Native Ingress Encoder (pre-registered, 2026-08-15)

Blueprint: `HENRI V2 Structural Analysis & Phase 8.12 Architecture.pdf`
(local `/tmp/p812_arch.pdf`, SHA-256 `5fc4f87d5dae4fc474e42d28edd25f046e26b884db1539502307ecbfa802ab62`).
Base branch: `feat/complex-native-ingress` from **sealed 8.11 tip `4db6916`**
(NOT main `2218ec4` — see deviation D1).

## Problem
Phase 8.11 (sealed `4db6916`) proved the internal complex wave transition is
EXACT at D=65,536 (G1b post=-1.19e-07, G5 0.983ms). But G2@scale was VACUOUS:
the production ingress (`HENRIVisionEncoder.encode_spatial_grid`, real-valued
per-block L2-normalized waves) collapses adjacent spatial states onto
collinear vectors (carrier dominance, K1 lesson). The world model has a
flawless internal simulator fed corrupt inputs.

## Mechanism (Lever 8.12-A)
`ComplexNativeIngress` maps raw grids s_t DIRECTLY to per-element
unit-modulus complex phasors in C^65,536 using incommensurate spatial
carriers:
- Carrier frequencies f_x, f_y drawn from `2*pi*rand(D)` (incommensurate,
  no integer-frequency aliasing).
- Grid binding: `Psi[x,y] = exp(j * (f_x * x + f_y * y + theta_v))` per
  colored cell, superposed, then per-ELEMENT unit-modulus renormalization
  (FHRR convention — NO vector L2 norm).
- Preserves spatial topology: adjacent cells share incommensurate carrier
  phase structure -> spatial cosine >= 0.85 gate (G1).
- Lie equivariance under spatial shift: a translation of the grid rotates
  the phasor phases by the carrier phase at the shift vector (mod 2pi),
  which is EXACTLY the algebra the Phase 8.11 transition implements.

## Gates (pre-registered)
- G1 (8.12-A): adjacent spatial state cosine >= 0.85 at D=65,536, plus
  distinct-state cosine < 0.95 (non-degenerate).
- G2 (8.12-B): held-out trajectory Sagnac L < 0.10 on LIVE env trajectories
  (translation pairs through ComplexNativeIngress + NativeComplexWaveTransition).
- G3 (latency): ingress + transition forward cycle <= 2.0 ms at D=65,536.
- G4 (default-OFF): flag OFF -> byte-identical legacy path.
- G6 (8.12-C): ARC-AGI-3 progress — **BLOCKED_NO_DEMONSTRATIONS** (no
  authorized demos; 20/20 envs), pre-registered as not-testable this phase.

## Kill criteria
- G1 fails (cos < 0.85 or degeneracy) -> kill, seal, record.
- G2 fails (L >= 0.10) -> kill (the complex-native ingress does not fix
  transfer); seal with lesson.
- G3 fails (> 2.0 ms) -> kill.
- G4 fails (default path changed) -> kill.

## Deviations from PDF (documented)
- D1: branch base = `4db6916` (sealed 8.11) NOT `2218ec4` (main): Lever
  8.12-B requires `NativeComplexWaveTransition`, absent on main. Main stays
  untouched.
- D2: PDF Step-2 CLI `o_vsa_ingress_tokenizer.py --mode complex_native_test`
  and Step-3 `wave_jepa.py --mode complex_e2e_test` are PHANTOM-CLIs (no
  `--mode` arg exists; 6th/7th confirmation). Real artifacts:
  `tests/contract/test_phase812_complex_native_ingress.py` and
  `experiments/performance/phase812_complex_ingress_cuda_check.py`.
- D3: new module `complex_native_ingress.py` (PDF names the class
  `ComplexNativeIngress`; the legacy `O_VSA_IngressTokenizer` has no such
  method — additive, not a modification).
- D4: default-OFF env flag `HENRI_ARC_COMPLEX_INGRESS` (production real
  path untouched).
- D5: `efe_planner.py` NOT modified this phase. The planner's production
  interface is real->real (it lifts via `_phasor`); consuming complex
  ingress waves requires a new planner branch (load-bearing). The wire to
  `NativeComplexWaveTransition` is executed at the TRANSITION level
  (`forward_complex` on ingress output) in the CUDA runner. If complex-
  domain G2 passes, Phase 8.13 adds the planner branch; if G2 fails, no
  planner surgery was wasted on a dead ingress.
- D6: carriers are BAND-LIMITED (s=0.10, E[cos]=sinc(s*pi)~0.984) NOT
  full-band U(0,2pi): full-band fails the blueprint's OWN G1 (adjacent
  cosine >= 0.85); E[cos(fx)] = 0 for U(0,2pi). 8.8 class lesson #1
  applied BEFORE coding.

## Evidence class
G1/G2/G3 OBSERVED on remote RTX 5090 (CUDA). G4 local contract. G6 BLOCKED.
