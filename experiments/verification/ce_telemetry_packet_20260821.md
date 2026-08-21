# CE Telemetry Module — Pre-Registered Packet (2026-08-21)

**Doc ID:** HENRI-CLASS47-CE-TELEMETRY-2026-08-21
**Branch:** `accuracy/fidelity-remediation` (HEAD `65aef1b` + this packet's commits)
**Governing source:** Levin "Inspiration Across Substrates" (k9lGNyPHurQ) — functional agency ratchet; Pigozzi & Levin arXiv:2605.06746 — CE on latent representations predicts final reward.

## 1. Mechanism

A diagnostic module `causal_emergence_telemetry.py` that measures Causal Emergence (CE) on the LIVE wave trajectory of the production planner, per Hoel's formulation:

- `EI(X) = I(X_t; X_{t+1})` under uniform do-intervention on X_t (empirical TPM from observed transitions, support-restricted, Laplace-smoothed — labeled DERIVED approximation, not a true intervention).
- Micro state: PCA-reduce each wave to r=4 coordinates (thin SVD over the window), sign-quantize each coordinate (K=2 bins) → micro state index (16 states). AMENDED 2026-08-21 (T2 smoke): r=8/K=3 gave 6,561 states over a T=64 window — nearly every transition unique, micro EI collapsed to ~1e-06, and k-means on one-hot time rows manufactured spurious macro determinism from white noise (noise CE 0.080 ≥ grid CE 0.068). Non-discriminative control = estimator failure; fixed pre-CUDA.
- Macro state: Hoel causal coarse-graining — cluster the VISITED states by their empirical TPM rows (k-means on rows), macro TPM = count-weighted row means. For pure noise all rows are the same random row → one cluster → EI_macro ≈ 0 → CE ≈ 0. For structured data, distinguishable rows → CE > 0 possible.
- **Null-surrogate correction (AMENDED 2026-08-21 pre-CUDA, T2 v3):** at T=64 / 16 states, each TPM row has ~4 samples; single-surrogate subtraction still leaves ±0.04 residual (observed noise corrected CE +0.042, grid corrected CE −0.061). Final estimator: `ce_null` = mean over **8 seeded shuffles** of the same window (marginals preserved, coupling destroyed), `ce_bits = ce_raw − ce_null`. Under iid noise `ce_bits ≈ 0`; negative values under the null are EXPECTED (corrected CE is null-centered) — gate S therefore bounds `|CE| ≤ 3`, not CE ≥ 0. Discrimination must be tested on a TEMPORALLY COUPLED control (alternating clusters with within-cluster jitter: macro deterministic, micro noisy → CE > 0), because iid frame sequences have no causal structure for any estimator to find.
- `CE = EI(macro) − EI(micro)` in bits.

## 2. Physical/mathematical hypothesis (falsifiable)

H1: On the production wave trajectory, CE is finite and > 0 after training windows (macro structure exists).
H2 (Levin ratchet): CE rises across SGLD-creep training windows (ΔCE > +0.01 bits).
H3 (forgetfulness resistance): erasing a suffix of the engram window does not drop CE below the pre-training level minus 0.02 bits.

## 3. Data path

`production_arc_run.py --ce-telemetry` (default-OFF) → per-step POST-RELAXATION planner wave (real `[num_blocks, 8]` float32, the wave used for action selection; raw observations would measure the environment, not the network) → sliding windows of T=64 steps → CE report appended to telemetry JSONL. No gradient, no action-policy influence. Diagnostic sidecar only.

## 4. Resource limits

PCA thin SVD on [64, D] at D=65,536: ~33 MB peak, < 50 ms on RTX 5090. Module adds < 1% step overhead. All computation on the device already in use.

## 5. Expected benefit

First-ever CE telemetry on the live path; enables the Pigozzi-Levin alignment check (CE ↔ external task outcome) without touching the representation. Functional use: analyzer correlates CE windows with external outcomes; the module never becomes an objective (internal-coherence trap banned by architecture).

## 6. Failure mode

Constant trajectory, NaN, unsupported short windows → return None with reason; never crash the runner, never emit a score.

## 7. Cheapest kill experiment

T1 calibration on analytic Markov chains (CPU unit tests). Exact Hoel EI under uniform do-intervention, EI = (1/n) Σ_i KL(TPM[i,:] ‖ p̄), p̄ = column mean:
- Deterministic 2-state identity chain (N=2000, 1000 transitions per row): EI = 1.000 ± 0.02 bits (analytic 1.0; Laplace smoothing bias ≈ 1/count ≈ 0.0014 bits at N=2000). CORRECTION 2026-08-21 pre-result: at N=200 (100/row) the empirical EI is ≈ 0.92 — bias 1/count is not negligible below N≈1000; the calibration therefore uses N=2000.
- Random (coin-flip) 2-state chain: EI < 0.005 bits (analytic 0).
- Lumpable 3→2 chain (a→{a,b} 50/50, b→{a,b} 50/50, c→c; macro A={a,b}, B={c}): EI_micro = log2(3) − 2/3 ≈ 0.918 bits, EI_macro = 1.000 bit, CE = +0.082 ± 0.01 bits. (AMENDED 2026-08-21 pre-result: original 0.531 was an arithmetic error; derivation above is exact.)

## 8. Pre-registered acceptance/rejection

| Gate | Condition | Rejection |
|---|---|---|
| T1 | Calibration within tolerance (above) | Estimator broken → fix before proceeding |
| T2 | CE on real waves @ D=4096 CPU smoke; support ≥ 8; **corrected CE: coupled-trajectory control > 0.01 AND > noise; noise |CE| < 0.02 @ T=256** (null-surrogate-discriminative) | BLOCKED_INFRASTRUCTURE (no verdict) |
| T3 | ΔCE(trained − untrained) > +0.01 bits @ D=65,536 CUDA | RATCHET_NOT_OBSERVED — seal negative measurement finding |
| T4 | CE(after engram erase) ≥ CE(untrained) − 0.02 bits | ASYMMETRY_NOT_OBSERVED — seal |
| S | **\|CE\| ≤ 3 bits, no NaN** (corrected CE is null-centered; negative under noise expected) | BLOCKED_INFRASTRUCTURE |

The first production run is a MEASUREMENT run. Positive T3/T4 = ratchet observed on this path (conditional, needs repeat). Negative T3/T4 = measurement finding sealed with evidence; module stays default-OFF diagnostic; no claim of mechanism falsification beyond "not observed on the live path".

## 9. Scope

In scope: module + contract tests + packet + default-OFF wiring in `production_arc_run.py` + offline CLI. Out of scope: any CE-based objective, any representation change, any score-gating use, Phase A/B simplification (separate approval gate, item 6 of the active task list).
