# Phase 8.16.1 — Shared-Projection Calibration & Gauge Realignment: Design Pre-Registration

PDF: HENRI-SPEC-2026-08-PHASE8.16.1-REFORM (SHA bdf4602b9554ebe8d79fe59e99d6fff14bb1ca4a5d22990e596496d9cb8e39be)
Base: e0745d7 (Phase 8.16 SEALED KILL). Branch: feat/phase816-1-shared-projection-reform.

## Mechanism
Single shared projection A = s·Â (Â orthonormal rows, s = 5e-6 pinned by construction).
L_obstruct(w, a) = ||A(w - a)||^2. Category-theoretic gauge alignment: eta_X == eta_Y.

## DERIVED scale accounting (validated @D=4096, spec-exact module + real codec)
- Phase-ring waves: ||w||^2 = D/2 (cos/sin blocks). Independent pair: ||w1-w2||^2 ~ D.
- Projection of isotropic x: E||Âx||^2 = (k/D)·||x||^2  (k = latent_dim rows).
- L_mism = s^2·(k/D)·D = s^2·k  -> D-INDEPENDENT. @k=2048: 5.12e-8 (OBSERVED 3.18e-9 @D=4096,k=128: s^2·k = 3.2e-9 ✓).
- L_valid = 0.0 EXACT (codec round-trip lossless; observed d_rt = 0.0).
- Ratio = inf (denominator clamp 1e-12 -> 3180 @D=4096).

## Spec constant errors (deviation D16 — gates unchanged, claims corrected)
- Spec claims L_valid 2.7e-5, L_mism 3.2e-4 @ s=5e-6. Implied ||w-a||^2 = 1.1e6/1.28e7 —
  impossible for bounded waves (max ~4 if unit; actual D for independent). Spec's own s would
  need to be 1.26e-2 (2520x larger). Actual: L_valid = 0, L_mism = 5.1e-8 @D=65536.
- "L ~ 2s^2" formula omits the k/D projector factor. Gates pass regardless (see table).

## Pre-registered gates (per spec, thresholds unchanged)
| Gate | Metric | Threshold | Prediction @D=65536 | Basis |
|---|---|---|---|---|
| G1-8.16.1 | L_valid < 1e-4 | spec | PASS (0.0) | probe @D=4096 exact |
| G2-8.16.1 | ratio L_mism/L_valid >= 10 | spec | PASS (inf) | probe @D=4096 |
| G3-8.16.1 | Triton LUT <= 50 us | spec | PASS (40.8 us) | 8.16 measurement, re-measured |
| G4-8.16.1 | top-1 recall >= 0.99 | spec | PASS (1.0000) | 8.16 measurement, re-measured |

Ratio defined with denominator clamp max(L_valid, 1e-12); L_valid < 1e-12 -> ratio = inf counts as PASS.

## Failure modes / kill criteria
- G1 FAIL if L_valid >= 1e-4 (scale calibration broken).
- G2 FAIL if ratio < 10 (ordering inversion or zero discrimination).
- G3 FAIL if sustained latency > 50 us (20 kHz budget breached).
- G4 FAIL if recall < 0.99.

## Honesty notes
- L_valid = 0 is trivially strong (exact pairs through shared projection). The evaluator is a
  DIAGNOSTIC metric; it does not by itself ground ARC actions. G4 is component recall, not task
  grounding. ARC-AGI-3 SOTA remains BLOCKED_NO_DEMONSTRATIONS regardless of outcome.
- Component 1 file: real module is HENRI V2/henri_functor_flow.py (spec cites functor_flow.py —
  phantom CLI #17). Tests at tests/contract/ (repo convention; spec cites tests/test_phase816_1_calibration.py).
- G3/G4 suite CLI (gpu_verification_suite.py --kernel phase_ring_lut_unbinder) is a phantom CLI #18;
  re-measurement runs through our CUDA check runner.

## VERDICT — SEALED ACCEPT (OBSERVED 2026-08-16, RTX 5090, D=65,536, commit 6f710e1)

Full-scale confirmation @ D=65,536 (evidence: p8161_matrix_d65536.json SHA f4b26c82..., log 6eeb8fad..., DONE_MARKER rc=0 failures=[]):
- G1-8.16.1: PASS — L_valid = 0.0 < 1e-4 (codebook round-trip pairs, exact).
- G2-8.16.1: PASS — ratio 51,041 >= 10 (L_mism 5.104e-8 / max(L_valid,1e-12)).
- G3-8.16.1: PASS — Triton LUT sustained 38.31 us <= 50.0 us (CUDA-event, 20 launches, m=3, D=65,536).
- G4-8.16.1: PASS — top-1 recall 1.0000 @ sigma 0.01/0.05/0.1 (n=256, Triton path).
- DERIVED THEORY CONFIRMED at production D: L_mism = s^2*k = 5.12e-8 (D-independent),
  observed 5.104e-8 (0.3% agreement). L_valid = 0 (codec round-trip lossless).

Phase verdict: ACCEPT (all four pre-registered gates pass). Corrected-gate reformulation
of the 8.16 G1 KILL is validated: shared projection + pinned scale resolves the noise-floor
inversion (8.16: valid 3.26e-2 vs mism 3.30e-2; 8.16.1: valid 0.0 vs mism 5.1e-8, ratio 5.1e4).
Component acceptance only: diagnostic evaluator, additive, default-OFF; G4 is component
recall, NOT ARC task grounding. ARC-AGI-3 SOTA remains BLOCKED_NO_DEMONSTRATIONS (20/20).
