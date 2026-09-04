# Carrier P2-0 Results: Action→Outcome Coupling Measurement

**Carrier:** `P2_ACTION_OUTCOME_COUPLING` (31st carrier; measurement-resolved rerun)
**Prereg:** `docs/spec/p2_coupling_preregistration.md` (`00e1d97c…`, sealed `18932505` @1,217)
**Engine SHA:** M1 tip `3d519d2` (docs-only delta at branch tip `4a602e9`); remote worktree at `4a602e92`, byte-identical engine.

## Receipt (OBSERVED, remote CUDA RTX 5090)

`p2_gates_receipt.json` — SHA-256 `4c14d189…` (local) == remote; 1,800 steps, EXIT:0, wall 1,185.6 s, 12 envs × 150, seed 20260930.

| Metric | P1 run (broken meter) | P2-0 (M1 meter) | Meaning |
|---|---|---|---|
| `mean_delta_nu_wp` | 0.0 (structural) | **2.07e-4** | meter live; no systematic progress |
| `creeps` | 0 | **395** (21.9% steps) | state does move in goal-gauge |
| `waypoint_advances` | 0 | 0 | no accumulation to 0.60 threshold |
| `envs_solved` | 0 | 0 | LG2 binding gate fails |
| `p1_score_calls` / drops | 1800 / [+0.048…−0.027] | 1800 / same pattern | P1 goal engagement reproduced |
| `g4_affordance_mean` | 0.0026 | 0.0015 | near-zero residuals persist |
| LG3 kernel latency | 10.86 ms | 10.74 ms | perf flag fires on precedence (not seal basis) |

Preflight PG1/PG1a/PG2/PG3 all pass; per-action subset AUC 1.0; 0 infra errors.

## Verdict: `P2_NO_PROGRESS` (sealed)

Pre-registered condition met: Δν ≈ 0 (2e-4 ≪ 0.05) AND advances = 0.

## Bisection update (epistemic)

1. **Measurement: REPAIRED + live-confirmed** (M1): Δν ≠ 0.0, creeps 395. The line is now a valid evidence channel.
2. **Goal-term engagement: CONFIRMED again** (P1): ΔV(a) action-discriminating, 1,800/1,800 calls.
3. **Action→outcome coupling through goal metric: FALSIFIED at this level** — P2-0 hypothesis ("P1 progressed; the meter hid it") is FALSIFIED. With a working meter, actions change state (creeps) but alignment with the bank-terminal goal wave does not accumulate (mean ≈ 0; zero advances; zero solves).
4. **Sharpened inference:** the failure is NOT measurement (M1 fixed) and NOT goal-term reachability (P1 engaged). The residual suspects are the **goal semantics** (terminal-frame attractor vs intermediate/sub-goal waypoints — non-monotonic task topology) and/or **action semantics** (whether actions move the field toward task-relevant structure at all beyond local wiggle). This is the G8 candidate territory named by the M1 closeout doc.

## Status

- W0: **GATED** (unchanged; requires ≥1/12 solved).
- Task record: 29 task carriers, 28 falsifications, 0 solved. M1 (measurement) verified; P2-0 adds a measurement-resolved falsification of the coupling hypothesis, not a task-carrier falsification.
- Next phase (G8 goal-semantics re-anchoring): **REQUIRES_APPROVAL** — load-bearing design change (new goal-source mechanism), not self-started.
