# Phase 8.39 — Gate B Verdict: IDF-Only 50-Item HumanEval Sweep

Spec: `HENRI-SPEC-GATE-A-PRIME-IDF-2026` (Gate B section)
Status: **FALSIFIED at benchmark level** (solved = 2/50 ≤ 2/50 threshold). Kill fired.
Gate A′: PASS (ranks /23=3, /35=5) → Gate B launched per pre-registration.

## OBSERVED (RTX 5090, D=65,536, CUDA, commit `3eb4f3c`)
Scorecard: `experiments/verification/humaneval_gateb_idf_50_scorecard.json`
(remote `/root/telemetry_logs/humaneval_wave_ast_1787253967.json`, sha `64b0f489…`)

| Field | Value |
|---|---|
| benchmark | HumanEval |
| status | EVALUATED |
| item_count | 50 |
| **solved** | **2** |
| accuracy_attempted | 0.04 |
| expressible | 50 |
| not_expressible | 0 |
| infra_errors | 0 |
| ast_idf_only | true |
| ast_idf_items_reordered | 7 |
| ast_idf_codebook_sha256 | ccf64ceae9c5403b… |
| dataset_sha256 | b796127e… (canonical HumanEval) |
| avg_latency_ms_item | 907.095 |
| wall_clock_sec | 46.117 |
| egress_path | WAVE_AST_GRAMMAR_SANDBOX |

## Pass items (both = Gate A oracle items; identical to standing baseline)
- HumanEval/23 — `return len(string)` (attempt 12 of 12)
- HumanEval/35 — `return max(l)` (attempt 12 of 12)

## Verdict
- Gate B target: > 2/50 (> 4.0%) authentic passes. Falsification: ≤ 2/50.
- OBSERVED 2/50 = 4.0% → **FALSIFIED**.
- No improvement over the standing 2/50 baseline. IDF reordering moved 7 items'
  first candidate but produced no additional passes.
- IDF lever class CLOSED at benchmark level. No promotion, default-OFF preserved.

## Governance
- Gate A′ verdict event: `9ab71d91-e5ca-40f1-86cd-812eec04efb8` /
  AUDIT_HASH `51cd1d38b6141f447d2cd7791aa14dceeca81728a76ddd169cb49f2b5b5fa50d`.
- Gate B verdict event: emitted alongside this file (id in commit message).
