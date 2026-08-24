# System-1 Stage-0b-rev / Stage-0c-rev — Identifiability Audit (2026-08-24)

**Verdict: IDENTIFIABILITY_BLOCKED** · Reference 3 (gpt-5.6-sol) binding
No Koopman adapter constructed. Stage-0c remains gated.

## Upload (proposal artifact, audited)
- `Stage-0c_Identifiability_Audit_and_Stage-0b-rev_Architecture.md`, 1,061 B,
  SHA-256 `dcce2be3a25e79fa8aefc9072b81eb11fe0dce130dd4964387f4ca1b3b6f717b`.
- Content: 14-line diagram — `x(4D) → RFF/Phase Modulation → Nonlinear RFF →
  Circular Conv Slot_i → Flat 6144D (PR ≥ 16) → EDMD UNBLOCKED`.
- C1–C3, B/S shapes, seed, dtype, normalization NOT specified in the upload —
  authored and sealed in `vla_stage0b_rev_contract.md` (sha `c2ca66e7…`).

## Feasibility (derived before code)
- NOT the degree-2 polynomial lift (which would be 15-dim → `FALSIFIED_BY_DIMENSION_BOUND`).
- RFF `φ=[cos(Wx̂+b); sin(Wx̂+b)]`, m=192 → span min(N, 384) = 171 > 16 → dimensionally feasible.

## Encoder contracts (frozen RFF encoder, C1–C9) — ALL PASS (OBSERVED)
- C1 default-OFF bypass byte-identical; C2 zero trainable state (class-source AST audit);
  C3/C5 cross-process output hash identical across separate runs:
  `a751cc77d61038ea190f5c3d670bee271b41bcdd8c416c8fe2719569549a69e0`.
- C4 `(1,16,384)` float32, per-slot unit-norm sphere max err 5.96e-08 ≤ 1e-6.
- C6 dedup (181 distinct obs): 100% pairs L2 > 1e-3, min 0.280, 0 collisions.
- C7 flat calib SVD rank(>1e-3) = 119 ≥ 16; min slot std 0.0212 ≥ 1e-3.
- Parameters frozen: `vla_stage0b_rev_params.npz`, full SHA-256
  `766e607ad0bc739ea0a139172dd34e16d01a268cca80e990af5aab01006cfcd7`,
  shapes W(192,4)/b(192,)/k(16,384), seed 20260824, calib stds nonzero.

## Stage-0c-rev audit — spectra (flat 6144-D, calib 171 = lexicographic first 10 eps; eval 133 untouched)
| Matrix | n | PR | r(>1e-3·s1) | r(>1e-6·s1) | κ16 |
|---|---|---|---|---|---|
| X0 | 102 | 7.03 | 89 | 102 | 5.49 |
| Y0 | 102 | 7.39 | 90 | 102 | 5.46 |
| X1 | 69 | 6.69 | 64 | 69 | 5.98 |
| Y1 | 69 | 6.80 | 64 | 69 | 5.89 |

## Gates (pre-registered) and disposition
- **G1 PR ≥ 16 per action: FAIL** (measured 6.69–7.39). No candidate r in
  {4,8,16,32} survives the combined gate → `IDENTIFIABILITY_BLOCKED`.
- **G2 κ16 ≤ 100 on X0/X1: PASS** (5.46–5.98, raw SVD, frozen calib transform).
- **G3 N_a ≥ 4r: satisfiable** for r=4 (16 ≤ 102, 16 ≤ 69) — moot under G1.
- G4 (eval reconstruction residual): NOT computed (frozen_r = 0).
- Audit JSON `vla_stage0c_rev_audit.json`, SHA-256 `5f44c3c8…`.

## What the RFF lift fixed (measured vs Stage-0b linear)
| Metric | Linear (Stage-0b) | RFF (Stage-0b-rev) |
|---|---|---|
| PR per action | 1.5–1.9 | 6.7–7.4 |
| κ16 | 1.9e3–3.3e3 | 5.5–6.0 |
| r(>1e-3) | ≤ 12 | 64–90 |
| r(>1e-6) | 44–56 | full n (102/69) |

## Corpus consult #24 (INFERRED, primary bank `ca4bb787…`)
- RFF span = min(N, 2m) supported; circular-conv binding preserves rank
  (Plate 1995 circulant operators) supported.
- PR≥16 on trajectory-correlated CartPole samples PREDICTED impossible;
  cited manifold saturation (stable rank ~2–4). **Live spectra override the
  corpus on the exact value (measured 7 vs predicted 2–4); the conclusion
  (gate fails) agrees.**

## Decision
- No adapter. No Stage-0c adapter pre-registration sealed. No re-tuning of
  W/σ (kill criteria — the frozen parameter spec is the experiment).
- **VLA gate stays 0/12.** Stage 1+ remains architecture map only.

## Next options (require user decision)
- (A) Stage-0b-rev2: multi-scale/multi-band RFF or explicitly higher-rank
  nonlinear observable — NEW carrier, NEW sealed pre-registration, NEW freeze.
  Corpus predicts manifold saturation persists (intrinsic dim ≤ 4); measured
  PR≈7 shows partial escape; the gate target would need its own justification.
- (B) Accept a re-derived gate (e.g. r=4 with PR floor ~7) under a NEW
  pre-registration — never relabel this audit's gates.
- (C) Richer substrate with more state DOF.
- (D) Hold.
