# Gate 1 — Few-Shot Scaling of Bounded Online Adaptation (Preregistration)

Source: `HENRI_SOTA_Benchmark_Wiring_Evaluation.md` (PROPOSAL, SHA-256 `85abb47a171993337a39e7e19590f498cbb465b8af5e0ccb12bbced5fb15bc51`) Phase 1.
Prior disposition: Gate 1 `REQUIRES_APPROVAL` + `BLOCKED_BY_PROVENANCE`. Resolved by Carrier A seal (event `4b28f89b-ef78-4a7c-bdf8-bbb6fcbbbe32`, commits `ca785c9` + `76fdf9d`) + this preregistration.
Branch: `feat/temporal-navigation-t0`. This document is the frozen contract. No result-dependent edits.

## 1. Hypothesis (falsifiable)

Bounded online adaptation (surprise-gated, Stiefel-retracted, episode-disjoint evaluation) on the T0 ledger stream produces positive action-conditioned information gain that grows with transition budget `n ∈ {1, 2, 5, 10, 32}`. Gain is attributed to action conditioning only if the real arm beats the shuffled-action control.

## 2. Data stream (T0)

- Flags: `HENRI_TEMPORAL_LEDGER=1`, `HENRI_LEDGER_PAYLOADS=1`. Rows are digest-only with payload refs; payloads are recovered via `LedgerPayloadStore.get_decoded(ref)` (the causal tensor source, never digests).
- Continuity: `record[t].obs_next == record[t+1].obs_t` within an episode; `reset()` splits episodes. Any violation -> `BLOCKED_INFRA`.
- Episode ordering rule (frozen): **lexicographic by `episode_id` string**, matching the sealed audit convention. Calibration = first `K_cal` episodes; evaluation = last `K_eval` episodes, `K_eval >= 2`. The exact ordered episode-ID list with record counts is emitted into the run manifest at execution time; the manifest is hash-pinned. The prereg fixes the RULE; the run fixes the IDs.
- Fail-closed: empty ledger, continuity violation, or payload decode failure -> `BLOCKED_INFRA`.

## 3. Arms (5) — one real, four controls

| ID | Arm | Update |
|---|---|---|
| R | real action-conditioned | per-action operator updated on the transition |
| S | shuffled-action | same update, action names permuted by a fixed derangement (no fixed points) |
| A | action-agnostic | single shared operator, action identity ignored |
| N | no-update | frozen baseline, no adaptation |
| P | persistence | `psi_hat_next = psi_t`, no parameters |

Equal budgets: every update arm sees exactly `n` transitions, one pass, same seed, same SGLD noise schedule (`sqrt(2*T*dt)`). Evaluation is pre-update for the evaluation episodes: metrics are computed with the adapted model and the evaluation episodes are NEVER adapted on. No selection on evaluation episodes.

## 4. Metrics and Delta-I (algebraic, frozen)

- Loss: `L_a(n) = mean over evaluation records of (1 - cos(psi_hat_next, psi_next))` in wave space. Aggregation unit = episode-level means; final value = mean of episode means.
- Improvement: `I_a(n) = L_N(n) - L_a(n)` (positive = better than frozen).
- Action-conditioned separation: `DeltaI(n) = I_R(n) - I_S(n) = L_S(n) - L_R(n)` (positive = real beats shuffled).
- CI: episode-level paired bootstrap, 10,000 resamples, percentile interval, paired across arms on the same evaluation episodes.
- Primary budget (frozen before measurement): **`n* = 32`**.
- Scaling trend gate (frozen): Spearman `rho(I_R(n), log n)` over the five budgets `>= 0.60` AND `I_R(32) - I_R(1) >= 0.02`.

## 5. Support requirements

- At `n* = 32`: every action observed in evaluation episodes must have `N_a >= 2` calibration transitions, else `BLOCKED_NO_EVAL_COVERAGE` (vocabulary inherited from the K2 runner).
- Low-shot budgets (1, 2, 5): `N_a < 2` is permitted; their `I` values are diagnostic-only (`ENGAGED_DIAGNOSTIC`), never verdict-forming.

## 6. Verdict precedence (fail-closed, total order)

```
BLOCKED_INFRA > BLOCKED_NO_EVAL_COVERAGE > FALSIFIED_NO_ENGAGEMENT > ENGAGED > ACTION_INFORMATION_GAIN > FEW_SHOT_SCALING
```

- `FEW_SHOT_SCALING` (Gate 1 ACCEPT): `DeltaI(32) > 0` with bootstrap lb > 0 AND trend gate passes.
- `ACTION_INFORMATION_GAIN` (PARTIAL): `DeltaI(32) > 0` with lb > 0 but trend gate fails.
- `ENGAGED`: updates executed and evaluation ran, but `DeltaI(32) <= 0` or CI includes 0 (Gate 1 FAIL).
- `FALSIFIED_NO_ENGAGEMENT`: update path never executed or zero evaluation coverage.
- `BLOCKED_*`: infra per section 2/5.
- Consequence: Gate 1 `FEW_SHOT_SCALING` unblocks Gate 2; otherwise Gate 2 stays blocked (sequential protocol).

## 7. Kill experiments (pre-registered)

1. `DeltaI(32) <= 0` with CI including 0 -> action-conditioned adaptation adds no information -> Gate 1 FAIL.
2. `I_R(32) <= 0` with lb > 0 -> adaptation worse than frozen -> FAIL (regression).
3. `DeltaI(32) < 0` with lb > 0 -> shuffled beats real -> FAIL + investigate action-binding leakage.

## 8. Flags and default-OFF

- New flag `HENRI_GATE1_ONLINE_ADAPTATION=1` (default 0). When off, the learner module is never imported (byte-identity kill test).
- `HENRI_FREEZE_LEARNING=1` remains the production default; the bounded experiment overrides it only inside its own process and restores it on exit.
- Zero-pretraining invariant preserved; no parameter change outside the experiment.

## 9. Evidence labels

All metrics `OBSERVED` from the live CUDA run; `DeltaI` `DERIVED`; verdict per precedence. Corpus consult: `BLOCKED_AUTH` (NotebookLM `nlm login --check` -> ClientAuthenticationError, 2026-08-26); to be retried before execution.

## 10. Artifacts

- This preregistration (doc).
- `experiments/verification/promotion_gate1_contract.json` (machine-readable contract).
- `tests/contract/test_promotion_gate1_contract.py` (contract test).
- Run manifest (execution-time, hash-pinned): ordered episode IDs, record counts, seeds, arm traces, telemetry.
