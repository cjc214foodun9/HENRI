# Phase 8.16 — Diagrammatic FunctorFlow Egress: Design Pre-Registration

PDF: HENRI-SPEC-2026-08-PHASE8.16-EGRESS (SHA 2ec601785c2c50f66655ee91e22012ce4442afc2221d83f111809b432fd65eec)
Branch: feat/phase816-lobstruct-egress-unbinder (from 27a7442 = Phase 8.15 tip, per spec §4)
Status: EXECUTING — G1 FALSIFIED as specified (cheap probe), G2/G3 verified, G4 blocked

## 1. Audit results (OBSERVED)

| Spec component | Cited file | Reality |
|---|---|---|
| C1 DiagrammaticEgressEvaluator | `HENRI V2/functor_flow.py` | MISSING file (phantom — real: `henri_functor_flow.py`); class MISSING → NEW implementation |
| C2 HENRINeuralEgressUnbinder | `HENRI V2/henri_decoder.py` | EXISTS @:58 (accepts vocab_size=256) → REUSE |
| C3 Triton phase-ring LUT kernel | `HENRI V2/qfhrr_kernels.py` | EXISTS (`qfhrr_similarity_triton`, int32 LUT @:47) → REUSE |
| G3 CLI | `gpu_verification_suite.py --kernel phase_ring_lut_unbinder` | PHANTOM CLI #13 |
| G1 CLI | `functor_flow.py --mode verify_lobstruct` | PHANTOM CLI #14 |
| G2 CLI | `henri_decoder.py --mode test_unbinder_recall` | PHANTOM CLI #15 |
| G4 CLI | `production_arc_run.py --mode phase816_benchmark` | PHANTOM CLI #16 (runner has only --envs/--steps) |

## 2. G1 mathematical falsification (pre-registered kill criterion)

Spec code: `l = mean((eta_X(wave_f) - eta_Y(ast_f))^2)` with two default-initialized
`Linear(dim, latent, bias=False)`.

Derived noise floor (default PyTorch init, bound = 1/sqrt(dim)):
  ||A||_F^2 = latent/3,  scale_eff = ||A||_F^2/dim = latent/(3*dim)
  L(w, a) = E[||(A-B)w||^2]/latent = 2*scale_eff*dim/latent = 2/3   (D-INDEPENDENT)

Gate threshold 1e-4 is ~6,700x below the metric's own noise floor at ANY scale.
Probe @D=4096 (OBSERVED): L_valid 3.2e-2, L_mism 2.0e-2 → ordering INVERTED
(valid pairs score WORSE than mismatched); no calibration procedure exists in the
spec; my calibrate() cannot converge A→B (held-out valid 3.0e-2 @ lambda=1.0).

KILL CRITERION (pre-registered): G1-EGRESS fails if L_valid >= 1e-4 with spec
defaults at any scale. FIRED. G1 = FALSIFIED as specified.

## 3. Corrected-gate proposal (requires USER APPROVAL — not auto-applied)

L_obstruct := mean(||A(w_f - a_f)||^2) with a SINGLE shared projection
(eta_X ≡ eta_Y = calibration limit), scale pinned by construction.

Pre-registered from measured codec statistics (NOT tuned): ||r||^2/D = 0.168
@D=4096; L_valid = s*||r||^2/latent, L_mism = 2*s*dim/latent.
Feasible band s in [1.56e-6, 1.86e-5] @D=65536/latent=2048 (factor 12).
Proposed s = 5e-6: L_valid 2.7e-5 (3.7x margin), L_mism 3.2e-4 (3.2x margin).

## 4. Gates for this phase (final)

- G1-EGRESS: FALSIFIED as specified (kill). Full-scale GPU confirm in runner.
- G2-EGRESS: top-1 recall >= 0.9900 on 256-symbol phase-ring codebook, held-out
  noisy realizations (sigma 0.01/0.05/0.1). Probe PASS @D=4096 (1.0000). Full D=65,536 on GPU.
- G3-EGRESS: Triton phase-ring LUT unbinding latency <= 50.0 us @D=65,536
  (CUDA-event sustained interval over 20 launches).
- G4-EGRESS: live ARC task grounding — STANDING BLOCKED_NO_DEMONSTRATIONS (20/20).

Phase verdict on completion: KILL (G1 falsified; G2/G3 component evidence;
G4 blocked). Corrected gate gated on user approval.
