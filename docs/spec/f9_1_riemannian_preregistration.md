# Carrier F9.1 — Riemannian Optimization & Manifold-Preserving Gradient Alignment: Pre-Registration

**Directive:** HENRI-DIR-2026-08-F9-POSTMORTEM-RIEMANNIAN-ORDER (18,721 B / `a198b34ae46f…`)
**Companion synthesis:** HENRI-SYNTH-2026-08-TIMESFM3-TRANSLATION (15,251 B / `6587cf0c877c…`)
**Ratified seal:** `F9_GATES_VERDICT #be40617c` (F9_OPTIMIZATION_FAILED, ledger 1,058, verified)
**Evaluated commit (base):** `d406ad4` on `carrier/f9-active-policy-gradients`; F9.1 branch `carrier/f9-1-riemannian-optimizer`
**Ledger target:** record 1,061+ (live head 1,060 `4d5088be…` at authoring — directive's "1,059+" stale, disclosed)

## 1. Hypothesis

F9's in-sample stagnation (L_CE ≈ ln 7, P@1_train 0.458) was an OPTIMIZATION
defect, not a representation falsification: Euclidean AdamW momentum buffers
are destroyed by per-step hard QR retraction (m_t ∉ T_{W_{t+1}} St(r,D)),
and η=1e-4 through the 65,536→128 bottleneck attenuated gradients
(‖∇W_in L‖_F < 1e-6 by epoch 3, per directive §2.1 — disclosed as directive
prose; not re-measured). F9.1 replaces the constrained Stiefel path with an
UNCONSTRAINED pre-activation adapter (LayerNorm + residual skip + L2 sphere
projection), scales the bottleneck to 512, lifts peak lr to 1e-2 under cosine
annealing with warmup, clips gradients to ‖g‖₂ ≤ 1.0, and trains the joint
objective L = L_CE + 0.1·L_transition.

Falsification: if the active objective still cannot exceed the passive
baseline under correct optimization dynamics, the offline active-adaptation
line for this representation family is closed.

## 2. Data (OBSERVED, pinned)

Same 12-env bank, N=1536 rows (all rows; env-boundary rows are valid for
classification), hashes npz `9e3c01b4…`, jsonl `1ca089b2…`. Grouped 4-fold
env-level CV (an env never splits across folds), seed 20260907.
Passive baseline pin: F8.1 best CV **0.4617** (sealed receipt `0566ba27…`).

## 3. Architecture (directive §3, faithful)

| Module | Shape | Constraint |
|---|---|---|
| W_down | [512, 65536] | unconstrained (NO QR) |
| W_skip (residual Π_fast) | [512, 65536] | unconstrained |
| LayerNorm | [512] | h = LayerNorm(W_down x + W_skip x) |
| W_up | [65536, 512] | unconstrained |
| Ψ | S^(D-1) | exact L2 row normalize (not a retraction) |
| D_a (so(8) generators) | [7, 8192, 8, 8] skew | exp4 Taylor (F9 precedent, disclosed) |
| M (egress prototypes) | [7, 65536] | row-normalized |

Forward: h = LayerNorm(W_down x + W_skip x); Ψ = normalize(W_up h);
logits = β·⟨Ψ, M_a⟩/√D (β=8.0); z = softmax.
Loss: L_total = CE(logits, a) + 0.1 · ‖exp4(D_a)·Ψ_blocks − next_blocks‖²
(transition targets row-normalized for scale comparability).

Optimizer: AdamW with gradient clipping ‖g‖₂ ≤ 1.0; cosine anneal
η_max 1e-2 → η_min 1e-5 with linear warmup (first 2 of 40 epochs), batch 256,
epochs 40, seed 20260907 (per fold: seed + fold index).

### TimesFM-3 synthesis dispositions (companion doc §4 — NOT fabrications)
| §4 item | Disposition |
|---|---|
| 1. Ingress p=32 patching + lookahead goal concat | **NOT_APPLICABLE**: bank inputs are already-embedded waves [N,65536] (no raw series to patch); no future-covariate/goal source exists in the bank — a synthetic goal would be a mock loop. The doc's LayerNorm+up-proj+L2-normalize ingress pattern IS implemented (Tier-1 confluence). |
| 2. Single-pass K=8 horizon unroll | **NOT_APPLICABLE**: F9.1 is a classification CV (no held-out future horizon target in bank). Noted for a future forecasting carrier. |
| 3. Zone C dual-stream aggregates | **BLOCKED**: requires ZONE_C_ENV=prod DSN (env-only, never defaulted); out of scope for this CV gauntlet. Documented as future work. |

## 4. Gates & kills (directive §4, verbatim thresholds)

| Gate | Criterion | Kill |
|---|---|---|
| G1 | L_CE(train) ≤ 0.5000 AND P@1_train ≥ 0.8500 (mean over folds) | K1 (optimization stagnation — active line terminated) |
| G2 | grouped 4-fold macro P@1 ≥ 0.6500 | K2 (generalization deficit) |
| G3 | macro P@1 − 0.4617 ≥ +0.2000 (≥ 0.6617) | K3 (zero active advantage) |
| G4 | min single-fold macro P@1 ≥ 0.5000 | K4 (localized fold collapse) |

## 5. Ternary verdict (pre-registered)

- ALL pass → `F9_1_RIEMANNIAN_VERIFIED` (authorizes F10 / production wiring).
- G1 fails → `F9_1_OPTIMIZATION_FAILED` (optimization line terminated).
- Else → `F9_1_ACTIVE_NO_GAIN` (objective engaged, no generalization advantage).

## 6. Disclosures

1. Directive §2.1's "‖∇W_in L‖_F < 1e-6 by epoch 3" and its forensic fold
   env names (cn04/dc22; fold 3 g50t/bp35/lp85) do NOT match the sealed F9
   receipt (actual folds: lp85/tr87/wa30, ar25/bp35/sb26, ft09/sc25/sk48,
   cd82/g50t/ka59). Design intent honored; forensic prose not re-sealed.
2. exp4 Taylor (F9 precedent) for D_a; exact expm infeasible per step at
   B×8192×8×8.
3. G1 uses mean-over-folds final train CE/P@1 (pre-registered; more honest
   than F9's min/max, disclosed).

## 7. Default-OFF & artifacts

`HENRI_F9_1_ACTIVE=1` required; fail-closed otherwise. Diagnostic only.
Receipt `f9_1_gates_receipt.json` (schema `f9-1-riemannian.v1`) at
`/tmp/henri_f9_1_riemannian/` on vast-5090.
