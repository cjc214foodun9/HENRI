---
name: henri-holonic-graph
description: "Cache first. Use when linking HENRI stores."
category: henri-workflow
---

# Holonic graph and store boundaries

## Cache-first execution contract

Freeze the existing system prefix, tool/schema order, and model roster. Load the bundle once; append task state and new evidence last. Keep static instructions and custom JSON serialization stable. Use deterministic collection and bounded deltas before inference. No padding, empty warm-ups, or loss of correctness/security for hit rate. Measure real provider reads/writes, eligibility, and cost; prefix hashes are not hits. Jev memoization is application caching, not provider KV caching. Full protocol: henri-agent-integration/references/cache-maximization-playbook.md.

A holon is a bounded workflow inside a parent workflow. Skills define roles, not running agents/services. Parents receive contracts/artifact refs, not raw traces/private reasoning. Workflow: henri-soul. Approval/delivery: integration. Semantics: ontology.

## Stores

Drive = originals/revisions; Obsidian = readable source/decision/conflict projections; NotebookLM = cited retrieval/synthesis; typed ontology = terms/mappings/constraints/evidence; audit chain = hash-linked governance; graph schemas = control-plane envelopes, not A/B/C; STRACE = derived failure analysis; time-series = operational telemetry; optional TrustGraph = verified relationship/agent flows; Zone C = latent/reference artifacts, not engineering notes.

CodeGraph = disposable SHA-pinned static source index, with dynamic/heuristic boundaries. Project memory = reviewed operational records mirrored in GitHub; optional Honcho = derivative retrieval, offline until approved and verified. Neither is the cited ontology, audit authority, or Zone C.

Link with IDs/hashes, not schema merger. NotebookLM does not implement the engineering graph; Zone C is not its substitute.

## Path

Drive source → hash/export → receipt → Obsidian → typed mapping → cited corpus consult → live consumer audit → approval → patch → push → remote evidence → decision.

Service gaps remain explicit. Mounted access is not cloud receipt. NotebookLM import needs source readback. Note creation is not ontology validation. Governance must pass ledger integrity before promotion.

## Optional TrustGraph

Verify endpoint, collection, flow, permissions, processing IDs, query provenance, and completed trace/result IDs. Packages and source symbols do not prove execution. No guessed collections, fake payloads, or silent in-memory fallback.

Use two controlled documents before bulk ingest. Compare provenance/context/latency with existing path at equal budget. Retain simpler path without measured benefit. A graph agent cannot self-approve, mutate GitHub, or launch CUDA beyond scope.

No-agent collectors are durable; delegated children are temporary. External writes need exact readback. Record failures and route changes separately; retries do not erase failures.

Store contract history: references/stack-before-20261001.md (revalidate dated claims). Index: references/reference-index.md. Commands/gaps: henri-soul/references/scientisttwo-workflow.md.

Historical snapshot: `references/stack-before-20261001.md` is not current policy.
