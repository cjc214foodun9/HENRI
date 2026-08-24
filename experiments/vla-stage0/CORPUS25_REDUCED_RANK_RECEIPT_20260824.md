# Corpus Consult #25 — Reduced-Rank Koopman on Full-Rank Low-PR Observables (INFERRED)

**Date:** 2026-08-24 · Notebook: HENRI philosophy (`ca4bb787…`) · Conversation `3179135d-9cca-42a0-a626-3b192ea55cc2`
**Question (one focused):** Does low participation ratio (PR~7) invalidate reduced-rank (r=4 or 8)
Koopman/EDMD fitting when the observable has full numerical rank and κ8<3? Which sources support
this and what are the stated limits? What baseline is non-vacuous for small one-step changes?

## Answer (INFERRED — corpus synthesis, NOT OBSERVED telemetry)
- **Low PR does NOT invalidate reduced-rank fitting — it justifies it.** Separation between
  numerical rank (thresholded) and effective/stable rank (physical invariants): trajectory-
  correlated spectra decay as a power law (exponent 1–2), so most high-dimensional phase space
  is "not effectively there" (Vanchurin). Featurizer audits (2606.25234) show stable rank
  saturates 2–4 even when blocks are allotted b=16. Fitting full rank forces inversion of
  near-zero eigenvalues → overfitting/collapse under active noise; a well-conditioned active
  subspace (κ8<3) is well-posed for projection.
- **Persistence/identity baseline pathology:** on slow systems (CartPole one-step changes small),
  identity achieves deceptively low error — vacuous without a causal baseline. Non-vacuous
  standard (2608.01615 protocol): composite normalized error C = δ_corr/δ_corr^MK +
  δ_vol/δ_vol^IID + δ_tail/δ_tail^MK < 1.0 vs training-free IID (spatial) and Markov-chain
  (temporal) controls.
- **Limits stated:** low-rank projection is for the stable physical manifold; it must be
  enforced with Stiefel compliance; residual normalization should be dimension-independent
  (RMS/√d) to avoid false vetoes at high D; learning gating on failure regimes.
- Sources: 8bbb7935, de214098, eeac77f1, 57d374a6, 3acc364a, 50001d2c, f4b94645, 3087c6b6,
  db783b33, b7bd4ca6, 833bbe68, 21e2cb54, 903a914b, 02199bb6.

## HENRI disposition (live code + measured telemetry OVERRIDE corpus on values)
- Corpus predicted stable-rank saturation 2–4; measured PR 6.7–7.4 (rev) → partial escape, same
  conclusion direction (reduced-rank defensible; full-rank fit ill-posed).
- Sealed Stage-0c-rev2 contract adopts: projected (coefficient-space) normalized Frobenius +
  persistence + calib-mean baselines + spectral radius + coefficient rollout (closest practical
  analog of the triage protocol for this substrate; the full 2608.01615 composite is not
  applicable to CartPole — no cross-sectional series structure).
