# Phase 8.39 Verdict — HumanEval Wave-AST Egress (SEALED)

**Status: EVALUATED (first verifiable external coding score)**
**Date:** 2026-08-20
**Branch:** `phase839/humaneval-wave-ast`
**Commit:** `79222a7`
**Hardware:** RTX 5090 (Vast 47411800), torch 2.12.0+cu130

## Result (OBSERVED)

| Metric | Value |
|---|---|
| Item count | 50 (official HumanEval.jsonl.gz, sha256 b796127e…) |
| Solved | **2 / 50** |
| Expressible | 50 / 50 |
| Infra errors | 0 |
| Checkpoint used | false |
| Egress path | WAVE_AST_GRAMMAR_SANDBOX |
| Accuracy (attempted) | 4.0% |
| Wall clock | 20.4 s |
| Avg item latency | 400 ms |

PASS items: `HumanEval/23` → `return len(string)`; `HumanEval/35` → `return max(l)`.

## Mechanism
Grammar enumeration (50–130 candidates/item) + transformation-relative wave ranking
(weak without in-context demos) + official test-code verification in a fail-closed
subprocess sandbox (`container-rlimit` mode; namespace mode unavailable on host).

## Context
- The archived `execute_authentic_coding_benchmark.py` token-decode path was
  **BLOCKED** (correctly fail-closed): `out-of-vocab token id 9237` — 10-token stub
  grammar map vs 32,000-logit head. Same mechanism falsified in MBPP run2.
- This run uses the proven egress (MBPP 11–17/500): grammar + ranking + sandbox.
- No checkpoint used: this is a grammar/sandbox score, not decoder-capability evidence.

## Gate
Kill criteria pre-registered: checkpoint LOAD required (n/a — not used); sandbox
launcher failures = EXECUTION_ERROR (0 occurred); any authentic pass = valid score.
**Gate PASS (2 authentic passes).**

## Scorecard
`experiments/verification/humaneval_839_scorecard.json`
sha256: `26484ee09ecfaad9a4ee4d239cf3bb2f36fb6ed7162b971dd30d69173f690173`

## Next
AA v4.1 adapter campaign (manifest drafted, all BLOCKED on primary-source
verification); extend grammar for HumanEval coverage (string ops, list indexing).
