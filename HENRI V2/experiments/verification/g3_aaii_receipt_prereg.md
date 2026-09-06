# Carrier G3 — Sealed Honest Harness + AAII-Constituent Receipt (Scaffold)
## Pre-Registration (2026-09-05)

Branch: `carrier/g3-aaii-receipt` @ base `origin/main` (8eafe95d).
Worktree: `C:/Users/chan/henri-worktrees/carrier-g3-receipt`.
Scope: AAII v4.2 constituent subset **Terminal-Bench v2.1 (10%)**, **SciCode (10%)**,
**HLE (10%)** — scaffold phase only (≤16 rows, ≤600 s), per co-scientist rigor
Phase-1 scaffold → transition audit → Phase-3 full scale. No AAII submission
(AAII evaluates hosted endpoints only; this emits an HONEST local receipt).

## 1. Purpose

Produce a sealed, provenance-pinned receipt that states, per constituent:
source pin (repo/HF ID + revision), license, staged bytes + SHA-256, evaluator
contract, the HENRI path exercised, typed per-item status, and explicit
limitations. A negative/blocked verdict is a valid result. NO score claim is
made unless an item executed against a PUBLISHED checker with item-level
evidence.

## 2. Source pins (re-verified 2026-09-05 via GitHub + HuggingFace APIs)

| Constituent | Origin (pinned at immutable SHA) | License | Gated |
|---|---|---|---|
| Terminal-Bench v2.1 | GitHub harbor-framework/terminal-bench-2-1 (main → commit pin) — HF dataset holds ONLY metadata (LICENSE/README/eval.yaml/registry.json, 25 KB) | Apache-2.0 | no |
| SciCode | HF SciCode1/SciCode (rev pin) `problems_test.jsonl`; LICENSE from GitHub scicode-bench/SciCode | Apache-2.0 | no |
| HLE | HF cais/hle | MIT | **auto** → BLOCKED_GATED (terms acceptance required) |

## 3. Evaluator contract

- Terminal-Bench v2.1: task = terminal shell session; published task metadata
  (public). HENRI path attempted = wave→text diagnostic egress. Exec requires
  containerized shell harness — NOT present on Vast (no Docker CLI) →
  expected status `BLOCKED_INFRASTRUCTURE_PENDING_HARBOR_EXEC` (honest), never a
  fabricated score. Harbor harness pin (verified 2026-09-05):
  github.com/harbor-framework/harbor @ 5c364a538e0af19eb58a53fdb895d7c0f974cef5, Apache-2.0.
- SciCode: public full prompts + gold solutions + tests (`test_data.h5`/dataset
  cards). Deterministic grader: exec decoded Python against a published test
  case in the EXISTING HENRI REPL sandbox (python only) — no LLM judge.
- HLE: `cais/hle` gated (auto) → require user terms acceptance; scaffold emits
  `BLOCKED_GATED` with terms URL and no bytes staged. If user authorizes, add
  as separate bounded stage with its own hash; never bypass gate.

## 4. Scaffold bounds

- max items per constituent = 2 (total ≤ 6 rows), wall ≤ 600 s, one remote
  run, GPU-exclusive-free (scaffold is near-zero GPU).
- Data dir /root/g3_data; output /tmp/g3_scaffold.
- Deterministic: fixed item selection (first N in sorted pinned revision),
  fixed seed for any decode; hash everything.

## 5. Verdict classes (typed)

`STAGED_OK` (bytes pinned+hashed) / `STAGED_BLOCKED_GATED` / `STAGED_BLOCKED_INFRA` /
`EGRESS_DIAGNOSTIC` (wave→text via the sealed K2/U2 diagnostic method; DIAGNOSTIC_ONLY,
never score-eligible) / `EXEC_RESULT` (candidate executed against published test;
records pass/fail/timeout) / `MODALITY_HARNESS_BLOCKED`.

## 6. Honesty rules

1. No scorecard, average, or "accuracy" line unless every row is `EXEC_RESULT`
   with ≥1 published checker run and zero infra errors.
2. `HENRINeuralEgressUnbinder` decode is DIAGNOSTIC_ONLY (basis-dependent;
   K2/U2 BLOCKED_SEMANTIC_CAPACITY documented 2026-08-25). Outputs are labeled
   diagnostics, never model capability.
3. Any import/dep failure = `BLOCKED_INFRA` with the exact error class; never
   substitute placeholder output.
4. HLE gated = `BLOCKED_GATED`; do not fetch, hash, or quote test content.

## 7. Kill criteria

- If staging fails on all constituents (network/API), scaffold = FAIL → stop,
  no full-scale phase.
- If egress produces no decodable bytes on any item, receipt states
  `BLOCKED_EGRESS_CAPACITY` (already documented) and the full-scale phase stays
  gated: no release-level score attempt until a calibrated semantic egress
  exists (structural, not a scaffold defect).

## 8. Seals

No sealed artifact is modified. Scaffold adds `experiments/verification/
g3_aaii_scaffold.py` + this prereg, both default-OFF / no runner wiring.
