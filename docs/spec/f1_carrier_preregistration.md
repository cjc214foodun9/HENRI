# F1 Carrier Differential Gauntlet — Pre-Registration

**Prereg ID:** F1-PREREG-2026-08-28
**Spec:** `docs/spec/f1_structured_carrier_spec.md` (SPEC-2026-08-28-F1-CARRIER)
**Directive:** HENRI-DIR-2026-08-M2-POSTMORTEM-F1-STRATEGY (inbox SHA-256 `1d2de7cc…92b40a`)
**Sealed baseline:** M2 `M2_HORIZON_COHERENCE_FALSIFIED` (#b80bbd8f, ledger idx 959, carrier_sha `3702af9`)
**Status:** STAGED — sealed before touching active model code. No production-path change is authorized by this document.

---

## 1. Objective

Prove or falsify, on the remote RTX 5090 CUDA target only, that a
zero-trainable, block-wise Lie displacement carrier
(`Ψ' = exp(θᵃMₐ)Ψ`, Mₐ = adjoint su(3) generators, R ∈ Ad(SU(3)) ⊂ SO(8))
preserves phase coherence across multi-step open-loop rollouts where the
rank-128 EDMD path disperses (M2 Δ̄ ≈ 0.98–1.01).

## 2. Bounded experiment scope (frozen)

| Parameter | Value |
|---|---|
| Environment cohort | ka59-38d34dbb, sk48-d8078629, sc25-635fd71a, g50t-5849a774, sb26-7fbdac44, vc33-5430563c (same 6 as M2) |
| Seeds | 20260905, 20260906, 20260907 (fresh disjoint from any prior cell) |
| Cells | 18 (6 env × 3 seeds), independent single-arm runs |
| Step budget | 60 interaction steps per cell |
| Horizon | k ∈ {1..8}, pending-buffer causal rollout (M2 `henri_m2_coherence.py` contract) |
| Flag | `HENRI_F1_CARRIER=1` (default-OFF; λ = 0.0 ⇒ byte-identical to baseline path) |
| Data source | live ARC-AGI-3 arcade environments via `production_arc_run.py --envs <n> --steps 60`; telemetry-only, no policy influence |
| Device | remote CUDA only; local CPU runs are shape/differential smoke, never the gate |
| Output root | run-scoped dir per cell, aggregate.log + per-cell JSONL (M2 layout) |

## 3. Pre-registered verdict gates (acceptance criteria)

| Gate | Criterion | Verdict class |
|---|---|---|
| G1 liveness | 18/18 cells RC = 0 | infra |
| G2 engagement | ≥ 0.95 engaged rows AND nonzero deltas for all k ∈ {1..8} | mechanism engaged |
| G3 primary coherence | cohort Δ̄(k) ≤ 0.35 for k ∈ {1..4} | **F1_CARRIER_VERIFIED** |
| G4 secondary | cohort Δ̄(k) ≤ 0.35 for k ∈ {5..8} | diagnostic only (not an acceptance gate) |
| G5 baseline beat | Δ̄(1) < 0.5 (M2 measured 0.9832) | required |
| G6 identity differential | flag-absent path byte-identical (λ=0 arm) | required |
| G7 kernel truth | implementation marker = TRITON on every cell | required |

Verdict taxonomy (M2-compatible): `F1_CARRIER_VERIFIED` |
`F1_CARRIER_FALSIFIED` | `FALSIFIED_NO_TRANSFER` (K3) |
`FALSIFIED_NO_ENGAGEMENT` (K5) | `BLOCKED_INFRA`.

## 4. Kill experiments (frozen)

- **K1 norm/orth:** 8-step unroll orth error ≤ 1e-4 AND per-block norm drift
  ≤ 1e-4. Failure ⇒ kernel defect ⇒ `BLOCKED_INFRA`, NOT a mechanism verdict.
- **K2 identity:** θ = 0 ⇒ outputs byte-identical to baseline. Proves
  default-OFF differential.
- **K3 fit transfer:** θ fit on calibration episodes (closed-form C5),
  evaluated on disjoint eval episodes; one-step Δ̄ < 0.5. Failure ⇒
  `FALSIFIED_NO_TRANSFER`.
- **K4 shuffle control:** per-block θ assignment permuted across blocks; gate
  Δ̄_shuffled − Δ̄_fit ≥ 0.15. Failure to separate ⇒ fit not causal ⇒
  `FALSIFIED`.
- **K5 engagement:** ≥ 95% of main telemetry rows carry finite horizon deltas
  for k ∈ {1..8} (M2 STEP-2 standard).

## 5. Telemetry contract (per step)

`f1_engaged`, `f1_theta_norm` (per-block ‖θ‖ mean/max), `f1_orth_err`,
`f1_impl` (`TRITON`|`TORCH_REF`), `f1_lambda`, plus the existing
`m2_sagnac_by_horizon`, `m2_engaged`, `sagnac_mean`, `active_temperature`.
`f1_drift_slope` (least-squares slope of cohort Δ̄(k) over k ∈ {1..8};
DIAGNOSTIC ONLY, no gate — the direct observable for the directive's
"eliminate the phase drift" clause).

## 6. Reduction rules

Cohort-wide per-horizon means over ALL cells (M2 reducer semantics:
`m2_coherence_reduce.py`), engagement denominator = main telemetry rows only
(ARC_ACTION_PAYLOAD event rows excluded). Per-env means reported as
diagnostics. Degraded horizon = argmax_k Δ̄(k) reported on FALSIFIED.

## 7. What this pre-registration does NOT authorize

- No change to `main` or the default production path.
- No training, SGLD creep, or trainable parameters (F1.1–F1.3 zero-trainable).
- No re-use of any consumed M2/R2 split or cell telemetry as fresh evidence.
- No relaxation of M2's original 0.15/8-horizon gate for M2 itself; the F1.3
  0.35/k∈{1..4} gate is a NEW carrier's pre-registered criterion, disclosed in
  spec §8.
- No egress/M3 claims. A passing F1 gauntlet unblocks M3 pre-registration
  only; it is not M3.

## 8. Escalation

`BLOCKED_INFRA`/harness defects → Contract C to /henri-architecture (≤ 2
targeted AST-diff iterations, then Sol). `FALSIFIED_*` verdicts → Contract C
to /henri-research for re-derived invariants (coupled-θ variant or different
generator family). Do NOT tune λ or θ scale to pass a failed gate.
