# Carrier F8.1 — Action-Conditioned Transition Derivative Ingress: Pre-Registration

**Directive:** HENRI-DIR-2026-08-F8-1-DERIVATIVE-INGRESS-ORDER
**Directive bytes/hash:** 14,638 B / SHA-256 `673d9afb69890373…` (direct read, 2026-08-31)
**Ratified seal:** `F8_ENV_DECOMPOSITION_SEALED #e66dd9bb` (ledger record 1,049, verified)
**Branch:** `carrier/f8-1-derivative-probe` (from `299507c`)
**Ledger target:** record 1,052+ (live head 1,051 `a8addc15…` at authoring — directive's "1,050+" stale, disclosed)

## 1. Hypothesis

The transition derivative bank
`ΔΨ_t = IFFT(FFT(Ψ_{t+1}) ⊙ conj(FFT(Ψ_t)))` (circular cross-correlation, real domain)
contains a decodable action channel: `I(ΔΨ_t; a_t) ≫ I(Ψ_t; a_t)`.
Falsification: if no probe family reaches the pre-registered gates, the passive
real-wave embedding family is uncoupled from discrete motor policy
(terminates passive-embedding paradigms; mandates active policy gradients).

## 2. OBSERVED bank schema (pre-registration probe, 2026-08-31, vast-5090)

| Artifact | Value |
|---|---|
| npz | `trajectories_production_run_f3v2.npz` SHA-256 `9e3c01b4…` |
| jsonl | `trajectories_production_run_f3v2.jsonl` SHA-256 `1ca089b2…` |
| psi | real float16 `[1536, 65536]` |
| next_wave | real float16 `[1536, 65536]` (authoritative stored successor) |
| actions_onehot | uint8 `[1536, 7]` |
| envs | 12 (ar25, bp35, cd82, ft09, g50t, ka59, lp85, sb26, sc25, sk48, tr87, wa30) |
| valid pairs | 1,524 (excluded 12 = 11 env boundaries + last row); directive N_valid = 1,524 ✓ |
| step-contiguity | 1,521/1,524 contiguous; 3 valid pairs span step gaps (disclosed) |
| no-change | 365/1,524 (24.0%) have next_wave == psi (stationarity/no-op) — reported as diagnostic |
| action entropy | H = 1.7991 nats ≥ 1.70 ✓ (G4 data side PASS, OBSERVED) |
| min N_a | 53 ≥ 30 ✓ (G4 data side PASS, OBSERVED) |
| majority | 0.2356 |

## 3. Pairing semantics (deviation disclosures)

1. **Successor source:** directive prose assumes Ψ_{t+1} = literal row t+1.
   OBSERVED: stored `next_wave[t]` is the causally paired successor and equals
   row t+1 in only 1225/1524 pairs. The loader uses stored `next_wave`; the
   299 differing pairs are disclosed, not hidden.
2. **Cancellation claim:** the directive's "Ψ ⊛ Ψ^† ≈ I for unit-norm phase
   vectors on S^{D-1}" holds for complex unit-modulus vectors. The bank is
   REAL float16 (F8 amendment `85da8bf4` precedent). Circular autocorrelation
   of real vectors ≠ identity. This claim is NOT assumed; gates stand
   empirically on the real-domain derivative.

## 4. Protocol (fixed by directive)

- Derivative: `ΔΨ_t = irfft(rfft(next_wave[t]) * conj(rfft(psi[t])), n=65536)`,
  float32; paired with `a_t = argmax(actions_onehot[t])`; valid mask =
  env-contiguous pairs. N = 1,524.
- Folds: 10-fold stratified by class, seed `20260905` (sealed
  `stratified_folds` from F8 carrier), disjoint and complete.
- Probes (directive suite): P1 min-norm LS (thin SVD, λ=1e-6); P2 L2-logistic
  (Adam, λ=1e-3, early stop); P3 3-layer MLP 65k→1024→256→7 (GELU/LayerNorm/
  dropout, early stop); P4 cosine k-NN k=3 (F8 acc_max family).
- No per-row normalization for P1–P3 (raw derivative); P4 uses
  `1 − |(1/D)⟨·,·⟩|` per F8 convention.
- In-sample: fit LS/logistic/MLP on the full 1,524; G1 uses max train acc
  over those three families. kNN train acc excluded from G1 (self-neighbor
  hit inflates it; reported as diagnostic only).
- Static baseline (G3): same four families, same folds, on `psi[t]` for the
  same 1,524 valid rows; G3 uses best(ΔΨ CV) − best(static CV).

## 5. Gates & kills (directive §3, verbatim thresholds)

| Gate | Criterion | Kill |
|---|---|---|
| G1 | in-sample train acc ≥ 0.9000 | K1 (derivative inseparability) |
| G2 | 10-fold CV top-1 acc ≥ 0.6500 | K2 (generalization deficit) |
| G3 | Acc_CV(ΔΨ) − Acc_CV(Ψ_static) ≥ +0.2000 | K3 (zero velocity advantage) |
| G4 | H(A) ≥ 1.70 nats, min N_a ≥ 30 | K4 (degenerate bank) |

## 6. Ternary verdict (pre-registered)

- ALL G1–G4 pass → `F8.1_TRANSITION_DERIVATIVE_VERIFIED` (authorizes F9).
- G1 or G3 fails (and G4 passed) → `F8.1_REPRESENTATION_FAMILY_FALSIFIED`.
- Otherwise (G2 fails with G1/G3 passing, or G4 fails) →
  `F8.1_INDETERMINATE` (partial signal; generalization or data caveat).

## 7. Default-OFF

`HENRI_F8_1_PROBE=1` required; fail-closed `RuntimeError` otherwise.
Diagnostic only — no production path is trained or modified.

## 8. Artifacts

- Receipt: `f8_1_gates_receipt.json` (schema `f8-1-derivative-probe.v1`) at
  `/tmp/henri_f8_1_derivative/` on vast-5090.
- Scorecard: per-probe train/CV acc, static baseline per probe, G1–G4 values,
  no-change fraction, differing-pairs count, env boundaries, H, min N_a,
  majority, folds seed, git SHA, bank hashes.
