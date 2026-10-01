---
name: henri-engineering
description: "Cache first. Use for source graphs and code review."
version: 0.1.0
category: henri-workflow
platforms: [windows, linux, macos]
---

# Bounded engineering graph workflow

## Cache-first execution contract

Keep the loaded bundle, system/tools/model roster, role rubric, and serializers fixed. Collect deterministic source inventories first. Append only changed paths, exact SHA/hash refs, test results, and bounded review findings. No full graph dumps or repeated fan-out. Project recall is a user-tail artifact, never a rewritten system prefix. Provider hits need usage telemetry; source-index reuse is not KV caching.

## When to use

Use during caller audits, implementation, recent-code simplification, independent review, and project handoff. This operates on engineering source, not Zone C waves, evaluation answers, or cited-paper truth.

## Procedure

1. Inspect Git SHA and dirty paths. Use a clean worktree. Resolve current ontology sources and prior decisions through henri-ontology; recall verified project records for that source SHA.
2. Use pinned CodeGraph to locate definition, callers, impact, and dynamic boundaries. Disable telemetry and update checks before first use. Index exact approved source into an outside-Git snapshot. Inspect source lines and real consumers after graph retrieval. AST extraction and heuristic edge labels do not prove execution or absence of dynamic callers.
3. Implement one approved mechanism with a discriminating test. HENRI model tests run on Vast CUDA/canonical CI; local tooling tests prove only their declared infrastructure scope.
4. Simplify only code changed in the task. Follow the actual language/project rules. Preserve outputs, APIs, defaults, safety gates, exceptions, device/layout semantics, and failure behavior. Prefer clear branches over nested ternaries; remove duplication only when tests establish equivalence. Attached model: opus is not a mandate to change the MoA roster. JS/React conventions do not govern Python.
5. Request independent review of the bounded diff, exact base/head or diff hash, approved requirements, threat scope, and real test receipts. Retain the supplied requesting-code-review template as guidance. Reviewer is read-only; no nested dispatch or self-approval. Resolve every Critical/Important issue or block delivery. At most two repair rounds; repeated same-class failures route to STRACE. Reviewer agreement is not experimental evidence.
6. Publish an explicit path allowlist, exact-SHA GitHub readback, and required run checks. Link the existing audit event, source artifact hashes, ontology records, and a reviewed public project-memory record. Do not create another audit chain, scheduler, or whole-repo upload.

## Project memory and Honcho

Use scripts/henri_project_memory.py through terminal for add/verify/query/remote-verify. Canonical reviewed records live in docs/project-memory/records; deterministic bounded file scanning supplies offline retrieval without another database. Each record pins Git-stored source bytes and commit, not working-tree filenames alone. Add requires --reviewed-public acknowledgement after content review. Pattern screening blocks tested spellings, not arbitrary sensitive prose; public review stays mandatory. No claim of complete secret or benchmark detection. Recall at another SHA is rejected unless separately requested as historical evidence. Never copy raw sessions, secrets, benchmark answers, or latent state into Git or Honcho.

Honcho is an optional projection, not the ontology or audit authority. The installed provider is offline and failed host compatibility at agent.turn_author; do not activate it or replace that import with a stub. Native saveMessages:false does not prevent memory-file migration or explicit tool writes. Before future activation require host compatibility, approved self-host endpoint, HENRI-only workspace, per-session strategy, tools-only recall, no raw writes/dialectic, and exact conclusion readback. See references/engineering-stack.md.

## Verification

Real source query returns a known symbol with matching bytes. Memory persists across processes, validates content/source hashes, rejects wrong SHA, path traversal, and sensitive data. Honcho sync must refuse while offline before network/client construction. Real bundle loader and frozen-contract validator pass. Read back external writes and audit chain; report unavailable services as BLOCKED. A larger graph or more memory is not measured optimization.
