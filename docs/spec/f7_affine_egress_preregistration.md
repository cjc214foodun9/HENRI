# Carrier F7: Per-Task Non-Unitary Affine Operator & Family-Conditioned Supervised Egress — Pre-Registration

**Document ID:** HENRI-SPEC-2026-08-F7-AFFINE-EGRESS
**Branch:** carrier/f7-affine-egress
**Status:** PRE-REGISTERED
**Directive:** HENRI-DIR-2026-08-F6-POSTMORTEM-AFFINE-DIRECTIVE (SHA-256 `3ea072eafa961430707e8be8d465aedf7cc163fc4b51fe8b6a625d4bd98eaff3`)
**Base commit:** `752d915` (F6 sealed tip, `F6_GATES_VERDICT=FALSIFIED_NO_GEOMETRY` `#8a0fd516`, ledger 1,020)

## 0. Executive summary, reconciliation, and disclosed corrections

The directive RATIFIES the sealed F6 verdict and pre-registers Carrier F7. All F6 receipts quoted in the directive match live state (verified): event `8a0fd516`, ledger 1,020, arms A 0.4360 / B 0.4352 / C 0.4414 / D 0.1597, G1 26.6028, G2 0.7763, G3 0.4360, G4 +0.0008, receipts `b1767340…`/`e84d167b…`/`82971003…`. `F6_GATES_VERDICT` was sealed `#8a0fd516` by this arbiter and the chain is intact (1,021 records, head `05bf87bf…`).

**Disclosed corrections (pre-seal, same pattern as F6's spectral-NS correction):**

1. **Dense-to-dual ridge form.** The directive's pipeline formula `A^(e) = Y^T Ψ (Ψ^T Ψ + λI)^{-1}` requires a `[D,D]` solve (infeasible at D=65,536: 17.2 GB fp32). The mathematically identical dual form is `A^(e) = Y^T (Ψ Ψ^T + λ I)^{-1} Ψ` — Gram `[M,M]` solve (M ≤ 20), then `[7,M] @ [M,D]`. Same minimizer of `‖Y − AΨ‖² + λ‖A‖²` (Woodbury identity). Contract C1 proves dense/dual equivalence at toy scale.
2. **SVD singular-vector convention.** The directive's §3.1 expression `Y^T V_r (Σ_r²+λI)^{-1} Σ_r U_r^T` swaps left/right singular vectors under the standard `Ψ = UΣVᵀ` convention; the correct dual-ridge factor form is `Y^T U_r (Σ_r²+λI)^{-1} Σ_r V_r^T` (verified algebraically, Appendix A). The implemented operator is the standard ridge minimizer, matching the pipeline diagram's intent.
3. **Tier 2 estimator definition.** The directive specifies `z_calibrated = z_raw^(e) + Σ_family z_raw^(e)` but gives no estimator for `Σ_family ∈ R^{7×7}`. This spec defines it (below); arm B isolates its contribution so its value is measured, not assumed.
4. **Demo count M = 20** (F6 harness convention `FROZEN_DEMO_K=20`; directive's "M ≤ 10" is the rank-truncation rationale — the useful rank is `≤ min(M, |A|) = 7` regardless, since `rank(Y) ≤ 7`).

## 1. Academic foundation

For environment e, the true map `f_e: X_e → A` is many-to-one (multiple distinct states → same optimal action). Unitary operators preserve inner products and cannot collapse distinct states into a shared attractor — measured in F6 (G1 = 26.60 on real demos; kernel healthy at 2.66e-6 on synthetic unitary fixtures). The F7 operator is an affine contraction on the state space:

```
z^(e) = A^(e) Ψ_X + b^(e),    A^(e) ∈ R^{7×D}, b^(e) ∈ R^7
A^(e) = Y_demo^T (Ψ_demo Ψ_demo^T + λ I)^{-1} Ψ_demo        (dual ridge)
b^(e) = ȳ_demo − A^(e) Ψ̄_demo                               (affine centering)
```

Singular values of `A^(e)` decay smoothly (rank ≤ min(M, 7) ≤ 7); `b^(e)` removes the global DC/background carrier. This is the exact minimum-surprise solution for the demo pairs; it is the falsification direction named in the F6 verdict (per-task non-unitary affine operator) and now directive-authorized.

**Tier 2 estimator (disclosed):** `Σ_family` is the pooled [7,7] covariance of demo readout vectors `z_raw` across the fold's TRAIN environments, ridge-corrected (`+εI`, ε = 1e-3):

```
Σ_family = (1/(K−1)) Σ_k (z_k − z̄)(z_k − z̄)^T + ε I,   K = total train-env demo rows
```

It is computed from train-env demos ONLY (causal: no heldout leakage), and mixes the heldout env's logits `z_cal = (I + Σ_family) z_raw` before argmax. Arm B (Tier-1 only) measures whether this cross-env pooling changes outcomes.

## 2. Architecture and data path

```
Ψ_X (S^{D-1}, D=65,536)  --FPB codec-->  demo Ψ per env
        +  per-env demo pairs (X_i, a_i), M=20
        |--> dual ridge solve (GPU): G = ΨΨ^T + λI [M,M]; A = Y^T G^{-1} Ψ [7,D]; b = ȳ − A Ψ̄
        |--> z_raw = A Ψ_X + b                          (Tier 1)
        |--> z_cal = z_raw + Σ_family z_raw             (Tier 2, train-env pooled)
        |--> a* = argmax_k z_cal                        (Tier 3)
```

- λ = 1e-3 (fixed, pre-registered). Singular values via thin SVD of the Gram solve; no `[D,D]` tensor is ever formed (C8 memory gate).
- Solve latency target < 2 ms on RTX 5090 (directive §3.1).
- Default-OFF flag `HENRI_F7_AFFINE=1`; additive branch in `arc_task_functor.py` alongside (not replacing) the F6 branch. Default path byte-identical (C5).

## 3. Pre-registered gates (directive §4, verbatim)

| Gate | Requirement | Failure action |
|---|---|---|
| G1 | In-context demo training P@1 ≥ 0.9900 (arm A) | KILL K1 (affine inversion numerical error) |
| G2 | Grouped held-out macro P@1 ≥ 0.7000 across the 12-env bank (arm A) | KILL K2 (affine operator inexpressibility) |
| G3 | Margin `P@1_F7 − P@1_F6 ≥ +0.2500` (arm A vs F6 circulant control arm C on the same new split) | KILL K3 (zero affine gain over circulant baseline) |
| G4 | Min fold accuracy `min_f P@1(fold_f) ≥ 0.6000` (arm A) | KILL K4 (localized fold collapse) |

Verdict taxonomy: `F7_AFFINE_PROMOTED` / `FALSIFIED_AT_SCALE` / `FALSIFIED_NO_GAIN` / `BLOCKED_INFRASTRUCTURE` (any nonzero arm exit → BLOCKED_INFRASTRUCTURE, never per-arm science).

## 4. Split and arms

- NEW sealed split: seed 20260902, schema `f7-split-seal.v1`, grouped 4-fold env-disjoint seeded-permutation rule (same as F4/F5/F6). The F6 split (seed 20260901) is CONSUMED (F6 gauntlet ran on it); loader REFUSES f6/f5/f4/f3 schemas (consumed-guard, C7).
- Arms (multi-arm kill matrix): A = F7 full affine (Tier 1 + 2 + 3); B = F7 Tier-1 only (isolates Σ_family); C = F6 circulant control (reuses `f6_adaptive_functor.compile_adaptive_functor` on the NEW split — G3 baseline, fresh numbers, never the consumed split's); D = identity / no-supervision floor.
- The 12-env trajectory bank is the same authorized F3 v2 capture (npz `9e3c01b4…`, jsonl `1ca089b2…`, manifest authorized, N=1,536).

## 5. Contract tests (RED first, `tests/contract/test_f7_affine_egress.py`)

- C1 dual-vs-dense ridge equivalence at toy D (≤1e-5 rel).
- C2 affine centering: `b` removes DC; zero-mean-Ψ demo ⇒ b = ȳ.
- C3 demo reconstruction: synthetic linear map Y = A*Ψ + b*, G1-equivalent recon ≥ 0.99.
- C4 per-task discrimination: per-env affine margin > 0.30 over a pooled global affine on heldout rows (the occlusion claim).
- C5 default-OFF differential: flag unset reproduces captured pre-wiring baseline constants; flag set changes operator (byte-differential).
- C6 rank cap: effective rank ≤ min(M, |A|) = 7; no dense `[D,D]` allocation (C8 memory).
- C7 consumed-guard: loader refuses `f6-split-seal.v1` receipts.
- C8 memory: no tensor larger than `[M, D]` or `[|A|, D]` allocated in the solve path.

## 6. Execution order (directive §5)

1. Seal `F7_PREREG_SEALED` + commit spec + push.
2. RED contract tests; implement `f7_affine_egress.py` + `f7_split_seal.py` + `f7_affine_egress_gates.py` + additive `arc_task_functor.py` branch (`HENRI_F7_AFFINE=1`, default-OFF).
3. GREEN (F7 + F6 + F5 + arc regression).
4. Seal `F7_IMPL_COMMITTED` + commit + push.
5. Remote CUDA gauntlet: bank hash verify → detached worktree at exact SHA → fresh split seal (seed 20260902) → arms A–D → gates G1–G4 → seal `F7_GATES_VERDICT` → receipts to `telemetry_logs/f7/`.
6. Final report + promotion review gate (promotion is a SEPARATE approval; the verdict is sealed before requesting it).

## Appendix A — dual-ridge factor form (correction 2)

Ridge minimizer: `A = argmin ‖Y − AΨ‖²_F + λ‖A‖²_F ⇒ A = Yᵀ(ΨΨᵀ + λI)⁻¹Ψ`. Thin SVD `Ψ = UΣVᵀ` ([M,r],[r,r],[D,r]): `(ΨΨᵀ+λI)⁻¹ = U(Σ²+λI)⁻¹Uᵀ`, so `A = YᵀU(Σ²+λI)⁻¹ΣVᵀ`. The directive's §3.1 `YᵀV(Σ²+λI)⁻¹ΣUᵀ` is not the ridge solution under standard SVD orientation; C1 verifies the implemented form against the dense solve.

## Appendix B — real-domain wave decision (correction 5, pre-impl)

The directive's pipeline diagram routes FPB-complex waves (`Ψ ∈ ℂ^{65,536}`) into the affine solve while §3.1 and Tier 1 specify `A^(e) ∈ ℝ^{7×65536}` and `z ∈ ℝ^7`. A complex-Ψ ridge would yield a complex `A` and complex logits, contradicting the real contract. The consistent reading, adopted here: **Ψ_X is the real flattened bank wave** (the stored `psi` array, `[N, 65536]` real, same family F6's harness consumed), and the FPB codec remains the representation stage for the arm-C circulant control exactly as in F6. The affine solve is then a real dual ridge:

```
G = Ψ Ψᵀ + λI  [M,M];  A = Yᵀ G⁻¹ Ψ  [7,D] real;  b = ȳ − A Ψ̄  [7] real
```

This satisfies the directive's real-matrix contract exactly and keeps the F7 operator in the same wave family as the bank and the F6 control.

## Appendix C — arc_task_functor wiring disclosure (pre-impl, amended)

`compile_task_functor(demo_pairs, tokenizer, device, task_id, hold_out_index)` encodes grid pairs into COMPLEX waves (`_to_complex(encode(grid))`). The F7 branch (`HENRI_F7_AFFINE=1`) fits the REAL dual ridge on `_to_real(wx)`/`_to_real(wy)` per Appendix B via the **implicit affine** form (C8: no `[D,D]` tensor is ever formed):

```
Xc = X − x̄;  G = Xc Xcᵀ + λI  [M,M];  GinvX = G⁻¹ Xc  [M,D]   (the only factor)
A x = Ycᵀ (GinvX (x − x̄)) + b,   b = ȳ − Ycᵀ (GinvX x̄)
```

The held-out check uses the affine's own metric: `held_out_cos = |⟨norm(A·hold_x + b), norm(hold_y)⟩|` in the real domain (identity_cos unchanged: the no-supervision floor). The emitted `w_task` is `_to_complex(norm(A·hold_x + b))` — the affine's predicted goal anchor — so the digest provably differs from the legacy operator (C5). `res.provenance["egress"] = {"schema_id": "f7-affine-egress.v1", "implicit": true, "factor_sha256": …}`. The legacy path stays byte-identical when BOTH flags are unset (C5); F7 branch precedence over F6 (mutually exclusive default-off flags).



