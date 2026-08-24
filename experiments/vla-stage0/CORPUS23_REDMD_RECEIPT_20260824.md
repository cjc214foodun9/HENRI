# Corpus Consult #23 — R-EDMD Fast-Weight Dynamics on Discrete-Action CartPole (INFERRED)

**Date:** 2026-08-24 · Notebook: HENRI philosophy (`ca4bb787…`) · Conversation `3179135d-9cca-42a0-a626-3b192ea55cc2` · 10 turns
**Evidence class:** INFERRED (grounded synthesis from bank sources; NOT observed telemetry)

## Action-conditioned lift (Discrete(2))
- **REJECTED:** additive frozen action embeddings (Ψ_t + a_t, Ψ_t ⊗ a_t) — superposition crosstalk σ² ∝ M/D distorts phase, inflates Sagnac reflection, can trigger false vetoes.
- **APPROVED:** separate operators per action — W_task = [W_0 | W_1] column-block routing, or K_0, K_1 independent operators. Zero-entropy discrete routing selector.

## Recursive update equations
- A_{t+1} = λ A_t + Ψ_t Ψ_tᵀ (forgetting-factor covariance accumulation).
- Forgetting bound: **λ ∈ [0.95, 0.99]**.
- Retraction/alignment discipline: L⁻¹A semi-unitary alignment, QR retractions for hard-locking blocks (Stiefel compliance).
- Dimensional regularization: L2 norms accumulate across high D — dimension-normalize before threshold comparisons.

## Stability diagnostics
- Covariance condition-number monitoring; spectral/structural stability governs the fast-weight adapter.
- Sagnac delta bounded to physical range [0.30, 0.35] for veto-threshold interaction (HYPOTHESIS-level calibration, not measured on CartPole).
- Fast-weight reset / branch isolation: branch-local updates only; engram crystallization only on demonstrated success (Δν > 0); reset to canonical baseplate after commit (zero cross-task contamination).

## Baselines to distinguish learning from persistence (corpus-proposed)
1. Independent per-series Markov chains.
2. I.I.D. state resampling (destroys temporal/causal structure).
- Composite error C = δ_corr/δ_corr^MK + δ_vol/δ_vol^IID + δ_tail/δ_tail^MK; target C < 0.50; transition error below parameter-free TV floor ε̄/(1−ρ₀).
- These remain **HYPOTHESIS until calibrated on live telemetry** (per skill: corpus-proposed thresholds are INFERRED/HYPOTHESIS).

## Stated limitations / boundary
- Latent MCTS branch rewinding is NOT `reset(seed) + replay` in the corpus model — it uses pure-latent unrolling with isolated branch registers (Wave-JEPA T, visual W). Our Stage-0c uses REAL env transitions with deterministic replay; the corpus framing is architectural, not a claim about our wrapper.
- Corpus operates at D=65,536 in its examples; **live boundary is [1,16,384] (d_slot=384, flat 6,144)** — corpus-vs-code conflict recorded; live code + Reference 3 win.

## Source support
- Sources cited inside the consult include the bank's VSA/phase-space, Koopman/EDMD, Sagnac homodyne, and engram-crystallization documents (full source list preserved in the NotebookLM conversation; compact citations in the consult transcript `call_00_HPx8DZ3fMpKlvcEu2m6a1170.txt`).
- No consensus-from-models evidence; corpus synthesis is INFERRED, overridden by live measurement when they conflict.
