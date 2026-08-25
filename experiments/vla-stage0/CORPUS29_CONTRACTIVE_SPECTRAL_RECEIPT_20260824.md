# Corpus Consult #29 — r=8 Contractive Spectral Projection (INFERRED)

**Date:** 2026-08-24 · Notebook: HENRI philosophy (`ca4bb787…`) · Conversation `3179135d-9cca-42a0-a626-3b192ea55cc2`
**Question (one focused):** r=8 + contractive spectral projection + SSR_rollout5≤0.80 defensibility; clamping failure modes.

## Answer (INFERRED — corpus synthesis, NOT OBSERVED telemetry)
- **r=8 ≈ PR~7 is defensible and theoretically mandatory** under manifold dimensionality +
  power-law spectral decay (α∈[6,7]; stable rank of trajectory-correlated observables saturates
  2–4 regardless of granted block dimension, source 3ffca7fa/2606.25234). Fitting full rank in
  this regime fits null-space noise → representation collapse.
- **SSR_rollout5 ≤ 0.80 vs persistence-5 is defensible:** persistence assumes x_{t+5}≈x_t and
  accumulates massive error as the pole falls; the clamped operator under-predicts acceleration
  magnitude (saddle eigenvalue > 1 clamped to 1) but preserves velocity direction and cross-block
  couplings. Expecting ≤0.35 (rev3's absolute gate) is "mathematically impossible" under
  truncation + clamping due to irreducible projection error.
- **Clamping failure modes (documented, confirmed):** (1) defective/non-normal K → cond(U)→∞,
  snorm >> ρ, transient energy growth — clamp ρ does NOT bound snorm; (2) complex-conjugate pairs
  radially clamped to unit circle destroy damping (σ=0), asymmetric precision breaks conjugate
  symmetry → imaginary leakage; (3) saddle mode clamped to λ=1.0 is blind to falling acceleration.
  Corpus remedy: damped iteration map g_η = ηf + (1−η)z, η ∈ (0, η0), same fixed points, plus
  Cholesky/QR retractions for manifold compliance (NOT Newton–Schulz; √3 basin divergence).
- **Non-vacuous verification:** multi-axis composite C = δ_corr/δ_corr^MK + δ_vol/δ_vol^IID +
  δ_tail/δ_tail^MK < 1.0 against training-free baselines (source 50001d2c); identical scoring
  geometry for baselines and model; embargoed contiguous held-out segments.
- Sources: 31d8eac8, 67f32455, db783b33, 3ffca7fa, d8a89525, 8bbb7935, 903a914b, de214098,
  eeac77f1, ec42ce06, b155c153, 35bf7e14, ee3072de, 02199bb6, 56cfbe73, 8a590317, 42311872,
  57d374a6, 50001d2c, 21e2cb54.

## HENRI disposition (live telemetry OVERRIDES corpus values)
- Clamping-pathology predictions CONFIRMED: snorm_raw 1.042/1.016 > 1.0 with ρ 0.949/0.927 < 1.0
  (non-normal transient growth, cond(U) 2.46/2.22 healthy — no explosion at this scale).
- Contraction never fired (ρ<1.0 both actions) → the upload's "Unblocked by ρ ≤ 1.0" framing was
  vacuous in this carrier; the 5-step relative PASS (0.516 ≤ 0.80) is a persistence-anchored
  result, not a contraction result.
- Corpus's "≤0.35 impossible" for 5-step absolute at r=16 CONFIRMED (rev3: 0.555). Rev4 uses
  SSR vs persistence-5 (0.516) per the corpus's ratio-normalization guidance.
- The multi-axis composite C (2608.01615) remains not directly applicable to CartPole (no
  cross-sectional series); persistence-1/persistence-5/calib-mean + SSR adopted as closest
  practical protocol.
