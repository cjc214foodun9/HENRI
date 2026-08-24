# System-1 v0.6.0.1 — Pre-registration + Frozen Manifest

**Date:** 2026-08-24 (before any CUDA run)
**Reference 3 (gpt-5.6-sol):** ceiling correction locked — B13 already
52/52 on heldout53_v055; endpoint = **verifier-call reduction under EXACT
capability preservation**, NOT pass-rate gain. No heldout created/consumed;
disposable dev split only.

## 1. Mechanism (pre-registered)

    s_k      = cos(E_task(x), E_cand(c_k))            # candidate-specific
    z_k      = (s_k - mean_k) / std_k                 # within-task z-score
    score'_k = -k + beta * z_k                        # stable reorder

- E_task: mean-pooled signature latent `[1, slots, d] -> [1, d]` (pre-verifier).
- E_cand: mean-pooled backbone encode_tokens over the candidate's OWN code
  tokens (pre-verifier: code generated before any sandbox call).
- NO family id, canonical answer, verifier/outcome result, or task dict in
  either representation (C3 static audit).
- Default OFF; beta=0 => byte-identical order (C2, G1).
- Deterministic; task-local (no cross-task state); within-task z-score
  falls back to 0 if std < 1e-6.

## 2. Frozen manifest (full SHA-256, pre-run)

| Artifact | SHA-256 (prefix) |
|---|---|
| zone_c_bridge_v0601.py | c80d039d7429ea94f309da061086f0c65c1d4b1552e3900a13ca79e4438a5e34 |
| contract_v0601.py | d16993a06f9b9d1622299f64334b677d1e259ec97962dfc24fa0b3631ebc457b |
| eval_v0601_dev.py | 68816f6dc752a1b9655c7aa823d1411d7e9907f8bda9efe4f1f444dc52c35c45 |
| system1_kernel_v055_ast_skeleton.py | d9a976adff4146a11950c51218ca32af1cde4b3db59431ce370a45913ff8d870 |
| system1_kernel_v041_energy_refactored.py | 754cbe4910e35c8f018f9b2fc411b3798d7b0ca83e7a42e2d37522358aa6fa1a |
| system1_kernel_v042_cegis_beam.py | 057e0ec1f39c58cfee5043c601c0e767cdc9e0dd43d6c2bf352a9e185f09f94a |
| train_system1_kernel_v04.py | 35025cfa5d40819d815ca4ec7f49b5bce16ab707fc3effab593907232e1cf67f |
| train_v051_discriminator.py | dd596e93327fb994b83407d0b7156cc491ccf65a233a643f1cec4925c0e6d651 |
| v041_energy_checkpoint.pt | 11d56121e4b091e2162078eb4cae71ce213dacc01397d8f8209bc9e2152a8f4d |

## 3. Arms (matched; differ only in the intervention)

- **B13** frozen v0.5.5 carrier, uniform pool order (baseline).
- **R0** retrieval enabled, beta=0 -> MUST be byte-identical to B13.
- **R1** candidate-specific retrieval, pre-registered beta=0.15.

## 4. Gates + verdict chain (pre-registered)

- Split: dev9_v0601, seed 90909, **n=65 (13 families × 5 tasks — n must be
  divisible by 13 for exact stratification; n=60 crashed the stratifier at
  startup before any task ran; out dir empty → zero exposure, no quarantine)**.
  Disposable dev split; sealed `87390286…`/`a09bf275…` in consumed guard.

- G1 integrity: fresh disposable dev split; guard passed (sealed
  `87390286…` and `a09bf275…` in consumed digest list); R0 == B13.
- G2 outcome preservation: rate(R1) >= rate(B13) AND per-family support
  preserved (no family regresses).
- G3 cost reduction: paired bootstrap CI of (calls_B13 - calls_R1) 95% CI
  upper bound < 0.
- G5 variance: within-task sim variance > 0 on every nontrivial pool.

Verdicts:
- `CANDIDATE_RETRIEVAL_COST_PROMOTED` (G1+G2+G3+G5)
- `CANDIDATE_RETRIEVAL_EFFICACY_PROMOTED` (only under a harder disposable
  condition — FUTURE; not this run)
- `CANDIDATE_RETRIEVAL_NO_EFFECT` (zero discordance, no cost reduction)
- `CANDIDATE_RETRIEVAL_NO_IMPROVEMENT` (nonzero changes, no improvement)
- `CANDIDATE_RETRIEVAL_REGRESSION` (G2 or cost gate violation)
- `INVALID_VERIFIER_REPLAY` (guard match — raised at startup)

## 5. Smoke evidence (plumbing, NOT a verdict)

smoke601_disposable seed 93391 n=13 budget 8: `NO_IMPROVEMENT`; R1 reorders
admits 12/13 but calls unchanged (mean 4.15 all arms; CI [-2.0, 2.15]);
G1/G2/G5 TRUE. 13-task smoke cannot decide the cost endpoint (per
grammar-expansion safety: small smoke = plumbing evidence only).

## 6. Attribution boundary

- This run measures candidate-specific retrieval on a DISPOSABLE dev split.
- No claim about heldout capability (B13 remains the 52/52 carrier).
- The 5-stage VLA roadmap is a planning artifact, NOT evidence of VLA
  capability; each stage remains a separate isolated carrier.
