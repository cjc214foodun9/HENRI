# Corpus Consult #26 — r=16 Over-Allocation vs PR~7; SSR Normalization; Action-Switch Rollout (INFERRED)

**Date:** 2026-08-24 · Notebook: HENRI philosophy (`ca4bb787…`) · Conversation `3179135d-9cca-42a0-a626-3b192ea55cc2`
**Question (one focused):** Is r=16 with SSR_eval≤0.40 and 5-step rollout≤0.35 defensible given
PR~7 and rev2 telemetry (projected one-step 0.15–0.18 at r=8, ρ 0.93–0.95, 5-step rollout 0.73–1.05)?
What SSR normalization is non-vacuous; what action-switch rollout protocol is correct?

## Answer (INFERRED — corpus synthesis, NOT OBSERVED telemetry)
- **r=16 > PR~7 predicted indefensible without aggressive regularization.** Corpus: strict
  separation between numerical rank and effective/stable rank; trajectory-correlated spectra decay
  as a power law; stable rank of featurizers saturates 2–4; fitting singular values 8–16 forces
  reconstruction of near-zero eigenvalues (null space populated by high-frequency phase noise and
  circular-convolution crosstalk) → overfitting → 5-step rollout error "physically impossible to
  satisfy ≤0.35 at r=16". Recommendation: r ≤ PR (e.g. 4 or 8).
- **SSR normalization:** raw unnormalized L2 in high ambient dimension accumulates off-manifold
  components (raw ~80 → rejects 93–100%); RMS normalization by √d_ambient restores the physical
  range [0.30, 0.35]. MSE/persistence identity baselines are vacuous on slow systems; the
  non-vacuous standard is the multi-axis composite C = δ_corr/δ_corr^MK + δ_vol/δ_vol^IID +
  δ_tail/δ_tail^MK < 1.0 against training-free IID and Markov-chain controls (source 50001d2c),
  with identical scoring geometry for baselines and model.
- **Action-switch rollout:** per-action bases require gauge transport — project the full latent
  state back to the common ambient representation, then project onto the target action's basis
  before applying that action's K (sources d8a89525, 02199bb6). Without it: coordinate shearing,
  high phase friction (Δ_Sagnac ≥ 0.35), false vetoes.
- **Stability:** Stiefel compliance via Cholesky/QR retractions (not Newton-Schulz; √3-basin
  divergence documented); ρ(K) < 1 does not guarantee rollout accuracy.
- Sources: de214098, db783b33, 774eefba, 57d374a6, eeac77f1, 903a914b, b7bd4ca6, f5a32da3,
  4c665d9f, e0e7d020, 50001d2c, d8a89525, 02199bb6, 21e2cb54, 3b49eb76, 97b163c4, 35bf7e14.

## HENRI disposition (live telemetry OVERRIDES corpus values)
- The rollout prediction was CONFIRMED: 5-step rollout 0.555 > 0.35 at r=16 (OBSERVED).
- The relative-transfer signal was positive (SSR_agg 0.369 ≤ 0.40 on fresh disjoint episodes) —
  corpus's blanket "indefensible" applied to absolute calibration, not relative skill.
- SSR normalized in PROJECTED coefficient space (not √D RMS of ambient residuals; √D applies to
  the Sagnac-veto channel, not EDMD coefficient-space SSR — convention authored in the sealed
  contract).
- Action-switch rollout implemented as full-state re-projection per Reference 3 + corpus.
- The full 2608.01615 composite is not applicable to CartPole (no cross-sectional series);
  persistence + calib-mean + SSR + rollout adopted as the closest practical protocol.
