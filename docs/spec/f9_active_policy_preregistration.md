# Carrier F9 — Active End-to-End Policy Gradient Optimization (Latent Wave Policy Flow): Pre-Registration

**Directive:** HENRI-DIR-2026-08-F8-1-POSTMORTEM-POLICY-GRADIENT-ORDER
**Directive bytes/hash:** 16,670 B / SHA-256 `80fb89f44fd5a7e3…` (direct read, 2026-08-31)
**Ratified seal:** `F8.1_REPRESENTATION_FAMILY_FALSIFIED #9a9b2464` (ledger 1,053, verified)
**Evaluated commit (base):** `0efd391` on `carrier/f8-1-derivative-probe`; F9 branch `carrier/f9-active-policy-gradients`
**Ledger target:** record 1,057+ (live head 1,056 `d413c2f3…` at authoring — directive's "1,054+" stale, disclosed)

## 1. Hypothesis

A parameterized wave ingress `Ψ_θ(s)` + differentiable egress prototype head
`M`, trained end-to-end by task cross-entropy backpropagation, deforms the
phase manifold so that discrete actions become linearly separable — exceeding
every passive-embedding ceiling (F4–F8.1, max 0.4617) by at least +0.25 macro
P@1 under grouped 4-fold held-out evaluation.

Falsification: if the active objective cannot exceed the passive baseline
under grouped generalization, the representation family itself is not the
binding constraint and no passive/offline adaptation of this family can carry
policy information.

## 2. Data (OBSERVED, bank hashes pinned 2026-08-31)

- Same 1,524 valid transition pairs as F8.1 (12 envs; `psi[t]` real float32
  [N, 65536], `next_wave[t]`, action `a_t`; env-boundary mask; hashes
  npz `9e3c01b4…`, jsonl `1ca089b2…`).
- Input `x_t = psi[t]` (the FPB embedding from the F3 v2 bank).
- Supervised label: `a_t`. Transition target: `next_wave[t]`.

## 3. Architecture (directive §3, real-domain adaptation)

| Module | Shape | Trainable | Retraction |
|---|---|---|---|
| W_in (Tier 1 ingress) | [65536, 128] | yes | QR per step (Stiefel) |
| D_a (Tier 2 Lie generators) | [7, 8192, 8, 8] skew-symmetric per block | yes | skew-symmetrize `(D − Dᵀ)/2` per step |
| M (Tier 3 egress prototypes) | [7, 65536], rows unit-normalized | yes | row normalize per step |

Forward (classification, directive Tier-3 formula adapted to REAL domain):
```
Ψ̃ = W_in @ (W_inᵀ @ x)          # rank-128 projection (directive: W_in · FPB(s))
Ψ  = Ψ̃ / ‖Ψ̃‖₂                   # S^(D-1)
logit_a = β · ⟨Ψ, M_a⟩ / sqrt(D) # β = 8.0 (HENRI Hopfield convention)
z = softmax(logits)
```
Tier-2 generators train through a transition-consistency auxiliary:
```
L_trans = mean_t ‖ exp4(D_{a_t}) · Ψ_t_blocks − next_t_blocks ‖₂²
```
with `exp4` = order-4 Taylor matrix exponential per 8×8 block (disclosed
approximation of scaling-and-squaring; cheap, differentiable; norms are
small). Combined objective `L_total = L_CE + 1.0 · L_trans`; **G1–G3 are
measured on L_CE / P@1 only** (transition residual reported as diagnostic).

### Deviations disclosed
1. Bank is REAL float16 (F8 amendment precedent); the directive's `Re(·)` is
   dropped (real domain) and `so(8)^8192` generators act per 8-block of the
   reshaped real wave.
2. `exp4` Taylor approximation replaces exact scaling-and-squaring (bounded
   ‖D‖, disclosed; exact expm infeasible at B×8192×8×8 per step).
3. Footprint: directive's "<5 MiB at r=128" is arithmetically inconsistent
   (65536×128 = 8.39M params ≥ 16.8 MB fp16). Actual fp32 ≈ 50 MB total
   (W_in 33.6 MB, D_a 14.7 MB, M 1.8 MB) — trivial vs 32 GB VRAM. Disclosed;
   gate is functional, not byte-count.
4. CE gradients flow into W_in and M; transition gradients into D_a
   (directive prose "CE → AdamW(W_in, D_a, M)" partially deviated: D_a is
   trained by the transition objective, not CE — disclosed).

## 4. Protocol

- **Grouped 4-fold CV:** 12 environments → 4 folds × 3 envs. An environment
  is NEVER split across folds (grouped = env-level generalization; stronger
  than F8.1 row-level folds).
- Optimizer: AdamW, lr 1e-4, batch 256, epochs 40 (fixed; no early-stop on
  held-out — fold selection would consume the split), seed 20260906.
- Metric: **Macro P@1 = mean over the 7 action classes of per-class
  accuracy, averaged over the 4 folds** (pin: class-balanced, then
  fold-averaged).
- Baseline (G3): F8.1 passive best CV = **0.4617** (pinned from sealed
  receipt `0566ba27…`). Target = 0.7117.

## 5. Gates & kills (directive §4, verbatim thresholds)

| Gate | Criterion | Kill |
|---|---|---|
| G1 | L_CE(train) ≤ 0.3500 (P@1_train ≥ 0.90) | K1 (optimization stagnation) |
| G2 | 4-fold grouped macro P@1 ≥ 0.7000 | K2 (generalization deficit) |
| G3 | macro P@1_active − 0.4617 ≥ +0.2500 (≥ 0.7117) | K3 (zero active advantage) |
| G4 | max ‖W_inᵀ W_in − I_128‖_F ≤ 1e-4 at end of training | K4 (manifold breach) |

## 6. Ternary verdict (pre-registered)

- ALL G1–G4 pass → `F9_ACTIVE_POLICY_VERIFIED` (authorizes carrier F10 /
  production wiring).
- G1 or G4 fails → `F9_OPTIMIZATION_FAILED` (K1/K4; harness or manifold
  defect, not a representation verdict).
- G2 or G3 fails with G1/G4 passing → `F9_ACTIVE_LOSS_NO_GAIN` (objective
  engaged, no generalization advantage — terminates offline active
  adaptation of this family).

## 7. Default-OFF

`HENRI_F9_ACTIVE=1` required; fail-closed `RuntimeError` otherwise.
Diagnostic only — no production path is trained or modified.

## 8. Artifacts

- Receipt: `f9_gates_receipt.json` (schema `f9-active-policy.v1`) at
  `/tmp/henri_f9_active/` on vast-5090.
- Scorecard: L_CE train/final, P@1_train, per-fold macro P@1, global macro
  P@1, G3 margin, final Gram error (max + mean), L_trans residual,
  per-fold env groups, seed, git SHA, bank hashes.
