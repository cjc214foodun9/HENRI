# System-1 v0.4 / v0.4.1 Lineage — Design & Verdict Record

Date: 2026-08-23. Branch: `feature/v0.4-token-fsa-lineage`.

## Provenance anchors (sha256)

| Artifact | Sha256 |
|---|---|
| Upload `System-1_Kernel_v0.4_Engine.py` (inbox 09:43) | `d582406c43016466009a144deb5e493fc6b693a88333b184f9d5ad10207460f3` |
| Upload `System-1_v0.4_Token-FSA_Architecture_Specification.md` | `b42815d356d6b2177309db0723b4726f82d946ab2da29264b01cb1f8fb99cd9e` |
| Upload `System-1_Kernel_v0.4.1_Refactored_Energy_Engine.py` (inbox 15:46) | `92208ed0c2a64de56c61e112a1f7b13d6c3eec1f2cb010de3ef8824c2cbc10c4` |
| `system1_kernel_v04.py` (amended, eval-only stochastic amend) | `52af40fc4e26a5a468f0fffa84392980ef8d2a817a76ff838b625a5ade1625c8` |
| `train_system1_kernel_v04.py` (amended) | `5a27e8c210ecfc2d5620352aab409af274697444c35b0246dd24587c6d14e20d` |
| `contract_v04_stochastic.py` | `b258f89c37470ba04254bc12eca58dc1ec5c5c72cce6595a85d78f546aec15cf` |
| `system1_kernel_v041_energy_refactored.py` (kernel; arch = audited v0.4 kernel, batched-seeded fix) | recorded in run receipt at runtime |
| `train_system1_kernel_v041_brier.py` (binary-Brier energy trainer) | recorded in run receipt at runtime |
| Checkpoint `ckpt_v04/best_val.pt` (step 500, loaded by v0.4.1) | `085e3e71dda140af31c9baf067d8cafeae1806b92dc37ebb1d7609de9a083c13` |
| Stochastic eval dev-split result `eval_stochastic.json` | `a068f889c1a2a22d74b6d16b30176f4e43365f09065f80bcbbfba7a892e0dce3` |
| Split `smoke40_v04.json` (dev, disposable) | `887d0d6c5590871e885c21aadb199a48b6a89a3807b2a09ac1503b47f1b5d3b0` |

## Claim status (evidence-labelled)

### VERIFIED — deterministic single-decode capability baseline
- Heldout `heldout40_v04` (consumed once, single eval): **20/40 = 50.0%** exact
  sandbox pass, **100% AST validity**, no NaN, no abort (CUDA 5090, 3000 steps).
- Do NOT re-evaluate `heldout40_v04`. It is consumed.

### VERIFIED — stochastic evaluator engagement (dev split `smoke40_v04`, n=40)
- Seeded per-particle `decode_sample` + energy-weighted `decode_vote`,
  matched-budget arms (swarm B=128 / single 1×B / beam width 128 / greedy).
- `seed_replay_identical=true`, `mean_unique_programs=3.65`, ast_valid 1.0.
- All arms **19/40 identical**; `delta_vs_single=0.0`, McNemar p=1.0,
  `energy_assoc_spearman=0.0` → kill_fired=true, diagnostic_only=true.
- Instrument non-vacuous: diversity real, energy ranking invalid at this
  checkpoint.

### FALSIFIED (this checkpoint/evaluator) — swarm-superiority claim
- Energy-weighted vote ≈ greedy (winner mass 85–128/128; spearman 0.0).
- The energy head regressed shaped reward, not external outcome.
- Broader rejection requires fresh-split replicated controls (uniform-weight,
  shuffled-energy arms) — NOT run; the claim is retired for this line, the
  architecture is not globally condemned.

## v0.4.1 audit disposition (upload `92208ed0…`)

| Upload claim | Disposition |
|---|---|
| Binary-supervised Brier outcome head E_phi | `ALREADY_IMPLEMENTED` (live v0.4 `BrierOutcomeBaseline`, Linear→LN→GELU→Linear→Sigmoid). The CHANGE is the training objective. |
| Factorized dual-rate core | `ALREADY_IMPLEMENTED` (live v0.4 `FactorizedDualRateRecurrentCore`). |
| Cross-attention name decoder + Token-FSA | `ALREADY_IMPLEMENTED` (live v0.4, FSA-masked egress). Upload's egress skips the FSA → rejected as `CONFLICTS_WITH_LIVE_CODE` for egress. |
| Temperature-gated non-collapsing SMC engine | `BOUNDED_IMPLEMENTABLE` but DEFERRED: adaptive-tau particle resampling confounds the energy-only attribution experiment. |
| 25.86M params | `FALSIFIED` for this project: 32k-vocab artifact. Live vocab ≈ 90 → 2.14M < 30M rule. |
| `__main__` verification | `FALSIFIED` (mock loop: random input, no sandbox, no FSA, no labels). Rejected. |

## v0.4.1 experiment (pre-registered)

Mechanism: FREEZE the proven v0.4 decoder/FSA/core. Change ONLY energy
supervision: per-step seeded `decode_sample` of n_free candidates → REAL
sandbox binary labels → Brier loss `(E_phi(z) − y)²`. 1000 steps from pinned
`ckpt_v04/best_val.pt`. Evaluate once on fresh disposable split `dev41_v04`
(seed 42+66661, digest recorded at runtime). Never touches consumed splits.

Gates:
- PRIMARY: raw Spearman ρ(energy, outcome) > 0, both classes present, ≥20 pairs.
- PROMOTION: ρ>0 AND permutation p<0.05 AND Brier < constant baseline AND
  AUROC>0.5 AND energy-vote ≥ matched single (delta ≥ 0.10, McNemar p<0.05)
  AND ast_valid_rate ≥ 0.9.
- ABORT: NaN/Inf, missing outcome class over 200-step window, energy variance
  collapse (<1e-4 sustained).

Attribution boundary: if rank correlation remains zero, energy-weighted
voting is RETIRED for this System-1 line — decoder tuning is not reopened.
