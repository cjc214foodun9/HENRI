# HENRI Graph Workflow Profile

This is a static HENRI workflow policy. `SOUL.md` remains superior policy.
This file contains no session state, timestamps, run IDs, telemetry, or task text.

## Authority

1. Hermes system policy.
2. Global `SOUL.md`.
3. This workflow profile.
4. Selected HENRI skill adapters.
5. Verified repository and runtime evidence.
6. User task requirements.

## Routing

- Run deterministic collection before model inference.
- Use one worker by default.
- Use bounded multi-worker MoA only for independent or contradictory questions, load-bearing mathematics, persistent defects, or explicit user request.
- Load `henri-architecture` before HENRI wave, tensor, EDMD, planner, learning, constraint, or Zone C changes.
- Load `henri-agent-integration` before Hermes, graph, prompt, cache, retry, skill, or telemetry changes.
- Route architecture decisions to integration implementation. Do not recursively load skills.

## Evidence

- Use deterministic receipts for hashes, paths, schemas, tests, citations, diffs, queries, and telemetry reduction.
- Treat model agreement as advisory. It is not evidence.
- Pass compact receipts and artifact references. Do not pass raw logs, full ASTs, raw SQL rows, or complete source files unless the task explicitly permits a bounded excerpt.
- A claim without a receipt is `unverified` or `blocked`.
- A clean process exit is not a benchmark score or model-capability result.

## Budgets and retries

- Maximum graph depth: 2.
- Maximum model calls per turn: 4.
- Default workers: 1.
- Maximum high-risk workers: 3.
- Maximum leaf retries: 1.
- Maximum CoE repair cycles: 2.
- Retry only the failed leaf.
- A retry must change the input fingerprint and include a repair instruction.
- Never restart successful parent or sibling nodes after a leaf failure.
- On exhaustion, return `partial`, `blocked`, or `budget_exhausted` with unmet checks.

## Approval and completion

- Require human approval for load-bearing mathematics, schema changes, production experiments, destructive operations, and remote execution.
- Use ADS-STE100 Simplified Technical English.
- Preserve the claim chain: claim, assumption, evidence, mechanism, action, verification, uncertainty.
- Before completion, verify artifacts, execution, returned output, evidence labels, failures, and next action.
