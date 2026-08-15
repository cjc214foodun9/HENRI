# Phase 8.9 — Frequency-Domain Diagonal Phase Rotators (PRE-REGISTRATION)

Source PDF: `docs-HENRI_V2_PHASE_8_8_POSTMORTEM_AND_PHASE_8_9_S....pdf.pdf`
Raw SHA-256: `ccacd14543c347e8b3d5df32ef1cd1ee4fc9dfe7c227626d777203b97fb6ddc3`
Extracted TEXT SHA-256: `f94a3cb1533abd376f7518fe8ea67eea9be0df301f0cf7d708ae301f953c23b0`
Author: Aletheia, Systems Architect. 7 pages. Git provenance: 8.8 sealed @ `6bde0a4`, main @ `2218ec4`.

## Protocol audit (premise discipline — phantom-CLI family)
- Step 1 (branch init from main @ 2218ec4): REAL.
- Step 2 `python "HENRI V2/wave_jepa.py" --mode diagonal_phase_test`: PHANTOM.
  `wave_jepa.py` has 0 argparse/`--mode` hits (5,326-char toy module; dead code since 8.7).
- Step 3 `python "HENRI V2/physical_world_model_benchmarks.py" --mode spatial_diagonal_test`:
  PHANTOM. File exists only at `HENRI V2/experiments/exploratory/`, 0 `--mode` argparse hits.
- Step 4 pytest: REAL (full suite run at every phase).
- Levers implemented fresh in a new default-OFF module (never imported by production);
  `LowRankCoupledTransition` @ `efe_planner.py:70` untouched. NO promotion.

## Levers (P0, per blueprint §3.1)
- 8.9-A: `FrequencyDomainDiagonalTransition` — action-conditioned diagonal phase
  rotator: Psi_{t+1} = Normalize(Psi_t * exp(j*Theta_a)), Theta_a in [-pi, pi]^D,
  init zero (identity), O(D) Hadamard, footprint NUM_ACTIONS x D float32 phases.
- 8.9-B: closed-form Wirtinger phase residual update:
  Theta_a <- Theta_a + lr * arg(Psi_{t+1}*conj(Psi_t)*exp(-j*Theta_a)).
- 8.9-C: train diagonal transition on CC-OS spatial carrier wavefronts; held-out
  Sagnac < 0.05 across 32 eval trajectories.
- 8.9-D (P1): Zone C phase codebook logging — NOT in scope this phase (P1; requires
  Zone C hypertable + agentic_event_store provenance chain; defer).

## Pre-registered deviations from the blueprint sketch (same class as 8.8 sketch-fix)
1. **Update rate**: sketch default `lr=1e-2` CANNOT meet its own gate 3
   (phase recovery < 1e-4 within <= 10 steps): at 1e-2/step, moving error from
   pi to 1e-4 needs ~810 steps (ln(pi/1e-4)/ln(1/0.99) ~ 8.3/0.01005). The exact
   residual update converges in ONE step at `lr=1.0`. Pre-registered: `lr=1.0`
   for 8.9-B/F2; note the sketch's 1e-2 is a tuning default, not a mechanism.
2. **Carrier state form**: 8.8-A produced REAL cosine carriers (Re[analytic]).
   A complex diagonal phase rotator does not act cleanly on real waves (real-part
   projection breaks the rotation). The blueprint's own eq defines the
   frequency-domain state as COMPLEX: Psi = F{s_t} in S^{D-1}. Pre-registered:
   8.9-C uses the ANALYTIC COMPLEX extension of the same 8.8-A carriers
   (Psi_d = exp(j*(r*Omega_d + c*Theta_d))), the canonical FHRR phasor form
   (Plate 1995: operate directly in the frequency domain). Real carriers remain
   Re[Psi]; 8.9-A forward is verified against the exact analytic translation.
3. **Footprint**: float32 phase params (16 x 65536 x 4B = 4.19 MB); the sketch's
   8.38 MB assumes complex64 storage. Polar construction is on-the-fly O(D).
4. **No vector-level Normalize in forward** (found during contract verification):
   the sketch's `Normalize(next_wave)` (vector L2) CONTRADICTS its own Sagnac
   formula 1 - |<pred, actual>|/D. Unit-L2 vectors floor Sagnac at 1 - 1/D ~=
   0.99998 even at perfect prediction -> gate G1 < 0.05 unreachable. FHRR phasor
   convention (per-element unit modulus |z_d|=1) keeps <a,a>/D = 1 -> Sagnac = 0
   at perfect; pure Hadamard rotation preserves per-element modulus, so no
   vector normalization is applied. Same self-gate-failure family as 8.8 sketch.

## Falsifiable gates (pre-registered; identical to blueprint §3.1)
- G1: held-out Sagnac L < 0.05 across 32 held-out eval trajectories (breaking the
  1.1673 low-rank wall).
- G2: end-to-end forward transition cycle <= 1.0 ms at D=65,536 on CUDA.
- G3: phase parameter recovery error ||Theta_learned - Theta_true||_inf < 1e-4
  within <= 10 online steps.
- Kill discipline: any arm nonzero rc -> BLOCKED_INFRASTRUCTURE; DONE marker only
  if ALL arms rc=0. Smoke-before-matrix at the SAME SHA. GPU-exclusive.
  Paired/seeded trajectory draws (one deterministic generator per (seed, step),
  cloned to train/eval arms). Evidence hashed before parse. Seal on feature
  branch; main untouched; NO promotion.

## Arms (CUDA matrix)
- F0 module self-test: identity init -> forward = state; unit modulus; complex
  dtype; deterministic; no [D,D] allocation (in-place Hadamard, 0 bytes alloc gate
  per 8.9-A).
- F1 8.9-A forward: apply learned rotator for known (dx,dy); predicted vs
  analytically-translated cos >= 0.9999; latency <= 1.0 ms (G2).
- F2 8.9-B update: recover Theta_a from a 1-action trajectory; ||err||_inf < 1e-4
  in <= 10 steps (G3).
- F3 8.9-C held-out: train 8 action rotators on train seeds; held-out Sagnac
  < 0.05 across 32 trajectories (G1).
- F4 end-to-end latency: full forward+update cycle <= 1.0 ms (G2).

## Environment
- Remote RTX 5090; D=65,536; /workspace/p89-wt worktree; checkpoint overlay
  SYMLINK (disk ~97-98% full); prod env /workspace/zonec_prod.env; detached
  setsid nohup; JEPA_DM_OUT manifest; DONE_MARKER aggregated.

## Known risks (pre-registered)
- Numerical: polar(ones, theta) per forward is a transcendental call on 512 KB —
  measure (F4). If latency gate fails, precompute phasor buffer per action
  (8.38 MB) — but that is a measured decision, not a pre-emptive change.
- Sagnac floor: float32 complex multiply gives |<p,a>|/D = 1 - 1e-7 -> Sagnac
  ~1e-7 << 0.05; gate is trivially satisfiable IF the learned rotator is exact —
  the honest test is F2/F3 phase recovery + held-out generalization.
- The module is DIAGNOSTIC ONLY. It is not wired into EFEPlanner. Production
  path byte-identical when flag OFF (no flag exists in production; module never
  imported).
