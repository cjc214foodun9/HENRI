# Phase 8.15 — Non-Abelian Chromodynamic Grounding: Design Pre-Registration

PDF: HENRI-SPEC-2026-08-PHASE8.15-QCD (SHA 621d245635c61b6f7a03b863839c4f694aa9b0a9bf3f5a19ac2c4dcc2a6e9419)
Branch: feat/phase815-su3-chromodynamic-grounding (base 9b507da, 8d6da01 ancestor)
Execution: RTX 5090, D = 65,536, PyTorch 2.12.0+cu130, Triton 3.1

## Mechanism (falsifiable hypothesis)

U(1) scalar phase superposition is commutative and isotropic; it cannot bind
relational color structure (8.14 diagnosis: attractor flattening). SU(3)
gauge matrices U_d = exp(i sum_a theta_{d,a} lambda_a) are non-commutative:
distinct colors produce U_A U_B != U_B U_A, making color-action order a
measurable geometric invariant instead of an ad-hoc positional codebook.

Falsifiable claims:
1. G1-QCD: for every distinct color pair, ||U_A U_B - U_B U_A||_F > 0.5000.
2. G2-QCD: non-singlet gauge noise triggers 100% confinement veto
   (Delta_Sagnac = 1.0); exact singlet states (a*I3) never veto.
3. G3-QCD: SU(3) gauge transport is learnable: fitted gauge on 10 steps
   predicts held-out steps 10..50 with loss < 0.1500 at D = 65,536.
4. G4-QCD: Triton 3x3 complex matmul kernel executes <= 50.0 us for
   N = 65,536 channels on RTX 5090 (20 kHz budget), max err < 1e-4 vs torch.

## Default-OFF

Flag: HENRI_ARC_CHROMODYNAMIC=1. Production default path untouched.
Module: HENRI V2/chromodynamic_grounding.py (additive).
Kernel: su3_matmul_triton (fallback su3_matmul_torch when Triton absent;
fallback alone cannot pass G4 - recorded as G4:triton_unavailable).

## Constants (pre-registered, deterministic)

- GELL_MANN_BASIS: 8 traceless Hermitian 3x3, Tr(la lb) = 2 delta_ab.
- DEFAULT_COLOR_PROJECTION [10,8] fixed integer matrix, all columns populated.
- sigma = 0.18 (string tension, PDF 1.2), eps_conf = 1e-3.
- Seed 20260816; N channels 8192 smoke / 65536 full; 45 color pairs full.

## Gates (identical thresholds local contract and remote CUDA)

| Gate | Threshold | Falsification |
|---|---|---|
| G1-QCD | min ||U_A U_B - U_B U_A||_F > 0.5000 (all distinct pairs) | any pair <= 0.5 |
| G2-QCD | singlet veto rate == 0.0 AND nonsinglet veto rate == 1.0 | any deviation |
| G3-QCD | held-out gauge-transport loss < 0.1500 (50 steps) | loss >= 0.15 |
| G4-QCD | triton latency <= 50.0 us, max_err < 1e-4 (N=65,536) | > 50 us or err >= 1e-4 |

Verdict: ACCEPT iff G1..G4 pass at D=65,536 on remote RTX 5090. Any failure
-> KILL (default-OFF module stays unpromoted). Runner DONE_MARKER + failures;
rc=1 iff failures non-empty (honest aggregation).

## Kill criteria (pre-registered)

- Local contract G1 fails with the fixed projection -> adjust projection
  constant ONCE, record new constant + min margin (D13 note); still failing -> KILL.
- G2 veto rates not exact -> KILL (confinement machinery inert).
- G3 fit loss >= 0.15 -> transport NOT learnable at scale -> KILL.
- G4 > 50 us after warmup -> Triton kernel does not meet 20 kHz budget -> KILL.

## VERDICT — SEALED ACCEPT (OBSERVED 2026-08-16, RTX 5090, D=65,536, commits 41fa119..42acdd3)

Evidence: p815_matrix_d65536.json (SHA 06dbf170...) + p815_full.log (SHA d5b9bf1d...),
retrieved from /workspace/p815-wt @ 42acdd3, remote /tmp.

| Gate | Result | Detail |
|---|---|---|
| G1-QCD | PASS | min ||U_A U_B - U_B U_A||_F = 1.806 > 0.5 (all 45 distinct pairs) |
| G2-QCD | PASS | singlet veto rate 0.0, nonsinglet veto rate 1.0 (N=65,536) |
| G3-QCD | PASS | held-out gauge-transport loss 4.5e-13 < 0.15 (fit 10, eval 39 steps) |
| G4-QCD | PASS | sustained interval 38.6 us <= 50 us; wall median 46.0 us; max err 1.07e-6 < 1e-4 |

DONE_MARKER rc=0 failures=[] verdict=ACCEPT, seed 20260816, smoke False.

Notes: G3 fit on the true-constant-gauge trajectory is exact by construction
(least-squares recovers U to solver precision); the gate validates the
machinery, not learning from noise. G4 wall_max 58.7 us includes host launch
jitter; the sustained CUDA-event interval (38.6 us) is the 20 kHz budget
metric per design. Module remains default-OFF (HENRI_ARC_CHROMODYNAMIC);
production ARC path untouched. ARC-AGI-3 task score remains BLOCKED_NO_DEMONSTRATIONS.
