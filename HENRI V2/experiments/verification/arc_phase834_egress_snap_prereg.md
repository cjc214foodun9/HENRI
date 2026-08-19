# Phase 8.34 — Evolution I Egress-Side Benchmark (preregistration)

Source: `Project HENRI: Phase 8.34 Evolution I & II Verification & Component Acceptance` (Drive inbox, 2026-08-19), Section 3.2 step 1: "Run the lexical_snap egress-side benchmark on the 90-record production bank to measure continuous-to-discrete retrieval precision."

Prior sealed evidence: Phase 8.34 transition benchmark ACCEPT @ 7127050 (A 0.9926, B 0.3153, C 0.3234, jac_field_delta 1.64e-3; log sha 45f5d14f…). Phase 8.33 VLA Gate 1 falsified wave→action linear decodability (calibrator A 12.06, E_cal 24.22).

## Hypothesis

`lexical_snap` (multi-vector zero-entropy Hopfield snap, hopfield_cleanup.py) converts continuous bank waves into discrete action symbols with retrieval precision above (a) random chance (1/6 ≈ 0.167) and (b) the raw cosine nearest-neighbour baseline over the same codebook; and the end-to-end chain (coupled EDMD prediction → snap) retains meaningful symbol agreement.

## Harness (frozen)

- Bank: `trajectories_production_run_1787164827.npz` + sealed manifest, digest-verified via `TrajectoryBank.load(verify_digest=True)`.
- Split: held_out_frac=0.2, seed=20260819, deterministic randperm (mirrors calibrator + 8.33/8.34).
- Codebook: per-action prototype engrams = L2-normalized mean of train `psi` rows per action index (K ≤ 6 engrams, D=65,536). Built from TRAIN only.
- Arms (same split, same device, CUDA):
  - A: raw cosine argmax of held-out `psi` over codebook → precision_A.
  - B: `ContinuousHopfieldCleanup.lexical_snap(top_k=1)` on held-out `psi` → precision_B + mean retrieval entropy H_B (softmax(β·sim), β=√D).
  - C: CoupledRecursiveDualEDMD (r=128, field ON, lambda_forget=0.98, online closed-form RLS, NO BPTT) fit on train; held-out predicted next-wave → normalize → lexical_snap → symbol; precision_C vs true action.
- No BPTT. No score-eligibility claims (diagnostic benchmark only; `trained_action_head_active` untouched).

## Pre-registered verdicts

- ACCEPT: precision_B ≥ 0.80 AND precision_C ≥ 0.50 AND H_B ≤ 0.10 nats (decisive, zero-entropy snaps, end-to-end symbol retention).
- KILL: NaN/divergence in any metric.
- CONDITIONAL: partial gain with any unmet conjunct (reason lists the failed clauses).
- Any nonzero arm exit / bank digest mismatch / GPU failure → BLOCKED_INFRASTRUCTURE (fail closed, no verdict).

## Amendment 1 (2026-08-19, pre-production smoke)

The original draft carried the conjunct `(precision_B − precision_A) ≥ +0.05` comparing
lexical_snap top-1 vs raw cosine argmax over the SAME normalized codebook. Reduced-dim
smoke proved these are arithmetically identical operations (0.2222 == 0.2222) — the
conjunct was vacuous (zero variance, can never fire). Removed. The comparison arm A is
retained in the harness as a sanity print (identical-by-construction check), NOT as a
gate. Retrieval decisiveness is instead gated by H_B (softmax-β entropy over the
codebook), which is non-vacuous at production D where engrams are near-orthogonal.

## Promotion linkage (PDF step 2)

Only a non-KILL egress verdict permits the FF merge of `feat/phase834-hopfield-lowrank-coupling` into `main@44cb165bc3`; all components remain default-OFF.

## Verdict (sealed 2026-08-19, full CUDA run @ fe583d8)

Full run on the sealed bank (72/18, seed 20260819, D=65,536, CUDA RTX 5090):
rc=0, log sha `ecdb9124092216a9`, bank npz sha `a5d8f1b3…`.

- A_raw_cosine_precision = 0.2778
- B_snap_precision = 0.2778 (5/18)
- B_retrieval_entropy = 0.0293 nats
- C_coupled_predict_snap_precision = 0.1111 (2/18)
- codebook_size = 4 engrams

**Verdict: CONDITIONAL** (prec_B and prec_C far below their gates; H_B passed).

Attribution: prec_B is statistically indistinguishable from random chance
(binomial p=1/6, n=18: mean 3, sd 1.58; 5 hits ≈ 1.3σ). prec_C is BELOW
chance. The snap is decisively low-entropy but snaps to the wrong symbols —
consistent with VLA Gate 1 (Arm D acc 0.222) and the calibrator (A 12.06,
E_cal 24.22): a THIRD independent confirmation that the bank wave
representation lacks recoverable action-discriminative structure at the
prototype/egress boundary.

Promotion note: the PDF gates the FF merge on "verification of the egress
benchmark" — verification did NOT meet its pre-registered thresholds.
Promotion is therefore HELD pending explicit user decision (literal prereg
wording permitted a non-KILL merge; PDF intent requires egress ACCEPT).
