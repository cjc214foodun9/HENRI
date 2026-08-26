# Gate 2 — Procrustes Goal Adapter: Held-Out Validation (Preregistration, Carrier C)

Source: `HENRI_SOTA_Benchmark_Wiring_Evaluation.md` (PROPOSAL) Phase 2.
Prior disposition: Gate 2 `POTENTIALLY_ALREADY_IMPLEMENTED` + `GATE_INVALID_AS_WRITTEN`.
Branch: `feat/temporal-navigation-t0`. This document is the frozen contract.

## 1. Legacy gate rejected (self-referential)

The wiring doc proposed: falsify if `|<Psi_goal, Psi_Y>| < 0.20` vs demonstration outputs.
This gate is invalid as written:

- `Psi_goal` is built FROM the same demonstration outputs it is compared against
  (calibration on the evaluated demos -> trivially high inner product, no held-out test).
- The raw inner product is scale-dependent and uncalibrated (no margin analysis,
  no negative control, no conditioning check).

Replacement: three controls with pre-registered margins, all exercised through the
LIVE `HenriTaskOperator` / `HenriGoalAdapter` (henri_goal_adapter.py:126-144) — no
reimplementation. Production wiring remains default-OFF (`HENRI_GOAL_ADAPTER=1` +
`LAMBDA_GOAL>0` at production_arc_run.py:1138); this carrier only VALIDATES.

## 2. Hypothesis (falsifiable)

The per-block orthogonal Procrustes compiler recovers an underlying orthogonal
mapping from demonstration pairs: held-out reconstruction cosine >= 0.95
(calibration-only fit), a known per-block orthogonal transform is recovered with
cosine >= 0.95, and a deranged (shuffled) pair correspondence destroys the mapping
(cosine <= 0.60). Margins frozen before measurement.

## 3. Fixture (frozen)

- Seed: 20260826. num_blocks = 8192, block_dim = 8 (live planner wave shape).
- Demos: m_total = 14, calibration = first 10 (pinned order), held-out = last 4.
  Episode/task-disjoint by demo index.
- X: per-block unit rows (row norm 1), random. Y_known = X @ O_k (known per-block
  orthogonal O_k from QR), per-block unit rows. O_k is a DIAGNOSTIC reference,
  never the fitted operator.
- Shuffle control: derangement pi(i) = (i+1) mod 10 (no fixed points) on the
  calibration X side; fit on (X[pi], Y) and evaluate on held-out.

## 4. Controls and margins (frozen before measurement)

| Control | Procedure | Gate |
|---|---|---|
| C1 known-transform positive | fit on (X, Y_known), reconstruct held-out | recon_cos >= 0.95, orth_err <= 1e-4 |
| C2 held-out reconstruction | fit on calibration only, reconstruct held-out | recon_cos >= 0.95 |
| C3 shuffled-pair negative | fit on deranged pairs, reconstruct held-out | recon_cos <= 0.60 |

Diagnostics (reported, not gates): per-block SVD condition number of M_k (median,
p95), min singular value p5, orthogonality error max, shape checks.

## 5. Verdict precedence (fail-closed, total order)

```
BLOCKED_INFRA > FAIL_SHAPE > FAIL_ORTHOGONALITY > FAIL_KNOWN_TRANSFORM >
FAIL_SHUFFLE_CONTROL > FAIL_RECONSTRUCTION > GATE2_VALIDATION_PASS
```

- Any NaN -> `BLOCKED_INFRA` (never scientific KILL for infra).
- Shape mismatch (operator not [8192,8,8], wave not [8192,8]) -> `FAIL_SHAPE`.
- orth_err > 1e-4 -> `FAIL_ORTHOGONALITY`.
- C1 cosine < 0.95 -> `FAIL_KNOWN_TRANSFORM`.
- C3 cosine > 0.60 -> `FAIL_SHUFFLE_CONTROL` (compiler not discriminative / leakage).
- C2 cosine < 0.95 -> `FAIL_RECONSTRUCTION` (Gate 2 falsified).
- All pass -> `GATE2_VALIDATION_PASS` (validates the existing path; NOT a
  production enablement and NOT a benchmark score).

## 6. Kill experiments (pre-registered)

1. C2 cosine < 0.95 -> Procrustes goal adapter does not generalize held-out -> FAIL.
2. C3 cosine > 0.60 -> pair correspondence does not matter -> compiler is
   non-discriminative (mock-loop-like) -> FAIL.
3. C1 cosine < 0.95 -> known transform unrecoverable -> implementation defect -> FAIL.

## 7. Evidence labels

Metrics `OBSERVED` from the live Vast run of `promotion_gate2_validation.py`
(torch CPU, portable fixture); verdict per precedence. Corpus consult:
`BLOCKED_AUTH` (NotebookLM stale, retried 2026-08-26) — retry before execution.

## 8. Artifacts

- This preregistration.
- `promotion_gate2_contract.json` (machine-readable contract).
- `promotion_gate2_validation.py` (validation module, imports live adapter).
- `tests/contract/test_promotion_gate2_contract.py` (contract test).
- Receipt JSON written by the validation module (all controls + diagnostics).
