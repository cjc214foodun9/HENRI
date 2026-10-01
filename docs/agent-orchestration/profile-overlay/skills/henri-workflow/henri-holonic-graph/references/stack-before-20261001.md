# Historical instruction snapshot

This is the previous entry file. It is not current policy. Use current SKILL.md for routing. Revalidate every dated runtime claim.

---
name: henri-holonic-graph
description: Use for HENRI TrustGraph holonic workflows. Keep graph context, audit events, time-series telemetry, and Zone C separate.
category: henri-workflow
---

# HENRI Holonic Graph Workflow

## Purpose

Use TrustGraph as an optional holonic context and agent-orchestration layer.
Use the local HENRI event store as the governance source of truth. Use the
agentic time-series database for operational events and compact telemetry. Use
Zone C only for HENRI latent artifacts and CUDA telemetry.

A holon is a bounded execution unit that is both a complete local workflow and
part of a parent workflow. The parent receives a contract envelope, not local
reasoning or raw tool output.

This skill defines an operating contract. It does not prove that a TrustGraph
service, an agentic time-series database, or a holonic HENRI consumer is live.

## Evidence boundary

Keep these claims separate:

| Claim | Required evidence |
|---|---|
| TrustGraph components installed | The Hermes interpreter imports the installed component packages and `importlib.metadata` reports versions. |
| TrustGraph service live | `tg-verify-system-status --skip-ui` returns a successful service result against the configured endpoint. |
| TrustGraph holonic execution live | A real `agent-orchestrator` or API flow starts, completes, and returns a trace or result with stable identifiers. |
| Agentic time-series database live | The configured adapter connects to the intended database and reports the schema, table, and write/read probe. |
| HENRI integration live | A controlled Drive source creates linked audit, graph, Obsidian, TrustGraph, and time-series records. |
| External task progress | A real task score, WIN, level completion, or other external result is recorded and causally linked. |

Package import, source checkout, CLI help, and a graph-shaped design do not
prove service integration.

## Holon topology

```text
HENRI root holon
├── research holon
│   ├── Drive source watcher
│   ├── source hashing and file-class adapter
│   ├── Obsidian projection
│   └── TrustGraph context-graph ingest
├── governance holon
│   ├── claim audit
│   ├── human approval
│   └── Hermes hash-chain seal
├── implementation holon
│   ├── repository audit
│   ├── bounded patch
│   └── changed-path manifest
├── execution holon
│   ├── commit and push
│   ├── HENRI CI / Vast CUDA run
│   └── returned run identifier
└── telemetry holon
    ├── deterministic time-series reduction
    ├── compact Obsidian projection
    └── outcome and rejection decision
```

Each child holon exposes:

```yaml
holon_id: stable identifier
parent_holon_id: stable identifier or null
contract_version: integer
input_refs: stable event, source, or run identifiers
output_refs: stable event, source, or run identifiers
status: observed | derived | inferred | hypothesis | falsified | blocked
failure_scope: local | parent | global
summary: bounded text or JSON
causal_parent_event_id: stable event identifier or null
audit_hash: Hermes audit-chain hash or null before sealing
```

A child failure stays local when the parent can select a valid alternative
without hiding the failure. The parent must record the failure and the route
change. Do not convert retry success into proof that the first attempt worked.

## Canonical data path

```text
Google Drive HENRI_Inbox
  → SOURCE_DISCOVERED
  → SOURCE_HASHED
  → file-class adapter
  → Hermes audit event
  → Obsidian Markdown projection
  → local event-store record
  → optional TrustGraph library upload and processing
  → optional GraphRAG or agent-orchestrator query
  → compact holon summary
  → agentic time-series event
  → approval / implementation / remote CUDA path
  → compact telemetry and external outcome events
```

The local event record must be sealed before the event is treated as governed.
The event log remains authoritative. Obsidian Markdown, Chroma, TrustGraph
context graphs, and time-series rollups are projections or separate evidence
layers.

## TrustGraph use

Use the installed TrustGraph CLI or Python API only after service preflight.
The checked source tree contains the following real interfaces:

- `trustgraph.api.Api` for REST and socket clients;
- `Api.library().add_document()` and `start_processing()` for document ingest;
- `Api.flow().graph_rag()` and `graph_rag_explain()` for graph retrieval;
- `trustgraph.agent.orchestrator.SupervisorPattern` for decompose, fan-out,
and synthesise execution;
- the `agent-orchestrator` service entry point in `trustgraph-flow`.

These source and package interfaces are not evidence that a local service is
running. Verify the endpoint, workspace, collection, flow, and returned IDs.
Never guess a collection or flow name in an evidence record.

For HENRI research, use one TrustGraph collection per approved domain. Keep
research claims separate from approvals, implementation records, execution
records, telemetry, and external outcomes. Use source identifiers and
explainability events to connect GraphRAG output to the original document.

Use the holonic execution path for bounded specialist work:

1. Root holon receives a research or audit question.
2. TrustGraph supervisor decomposes the question into independent goals.
3. Child holons retrieve or audit only their assigned boundary.
4. Parent holon receives compact findings with source and provenance IDs.
5. Parent holon rejects missing provenance or contradictory evidence.
6. Governance holon records approval before code or remote execution.

Do not allow a TrustGraph agent to mutate the repository, approve its own
patch, launch a HENRI run, or create an external outcome event without the
existing Hermes approval and remote-verification path.

## Agentic time-series database

Treat the agentic time-series database as an operational evidence store, not as
latent memory. Its minimum event contract is:

```text
time, event_id, holon_id, stream, event_type, run_id,
metric, value, unit, source_hash, parent_event_id,
causal_status, audit_hash, schema_version
```

Write only bounded events and compact measurements. Recommended streams are
`research`, `graph`, `approval`, `implementation`, `execution`, `telemetry`,
and `external_outcome`. Keep the following metrics separate:

- graph latency, token use, context size, provenance coverage;
- ingest count, bytes, processing status, and failure reason;
- GPU step latency, memory, fallback, candidate admission, and learning
  engagement;
- external score or outcome.

Use time-window queries and continuous aggregates only after the live schema
and adapter are verified. A time-series row proves that an observation was
stored. It does not prove that the observation changed a candidate ranking or
improved a task.

Do not write TrustGraph graph payloads or Obsidian notes into Zone C. Do not
write wave checkpoints or full latent engrams into the local vault or the
agentic time-series database. Link stores with stable IDs, hashes, and compact
artifact paths.

## Audit and causal rules

For every material transition, preserve:

```text
claim → assumption → evidence → mechanism → action → verification → uncertainty
```

Recommended event sequence:

```text
SOURCE_HASHED
→ TRUSTGRAPH_DOCUMENT_UPLOADED
→ TRUSTGRAPH_PROCESSING_STARTED
→ TRUSTGRAPH_PROCESSING_COMPLETED
→ GRAPH_PROVENANCE_OBSERVED
→ CLAIM_AUDITED
→ APPROVAL_REQUESTED
→ HUMAN_DECISION
→ PATCH_APPLIED
→ REMOTE_RUN_COMPLETED
→ TELEMETRY_REDUCED
→ OUTCOME_ACCEPTED or OUTCOME_REJECTED
```

A later observation must use a new event. Do not backfill an outcome into the
action that preceded it. Use explicit typed edges such as `SUPPORTS`,
`CONTRADICTS`, `DERIVED_FROM`, `REQUIRES_APPROVAL`, `VERIFIED_BY`,
`IMPLEMENTS`, `MEASURES`, and `SEPARATE_FROM`.

## Deterministic-first operation

Run deterministic collectors before model inference:

1. Verify the Hermes audit chain.
2. Verify local event payload hashes and projection.
3. Query the Obsidian event log by stream and time window.
4. Query the agentic time-series database by `holon_id`, `run_id`, and time.
5. If enabled, query TrustGraph for relationship-heavy retrieval and retain
   provenance identifiers.
6. Produce a compact context envelope.
7. Use a model only for synthesis or a bounded decision proposal.
8. Wait for human approval before load-bearing implementation.
9. Verify HENRI execution on canonical CI or Vast CUDA.
10. Reduce telemetry and record a separate outcome event.

Do not send raw TrustGraph traces, raw telemetry, or full session history into a
parent holon. Store them outside the model context and pass paths, hashes, and
bounded summaries.

## Controlled activation and kill experiment

Do not process the full Drive inbox on first activation. Use two uniquely named
controlled documents. For each document, require:

- source hash;
- sealed local event;
- Obsidian note;
- TrustGraph document and processing IDs, if TrustGraph is enabled;
- provenance identifiers from a real query;
- time-series write/read confirmation;
- intact Hermes audit chain;
- deterministic graph projection.

Compare the current local Chroma/event path with the TrustGraph holonic path on
a fixed research set and fixed context budget. Pre-register:

| Measure | Accept TrustGraph path when |
|---|---|
| Provenance coverage | Higher than baseline without unsupported source links. |
| Causal-link validity | Higher than baseline after human review. |
| Context size | Lower or equal at the same task scope. |
| Retrieval latency | Within the registered operating limit. |
| Human correction count | Lower than baseline. |
| External HENRI outcome | Improved only if a real task outcome is measured. |

Reject the addition if it only renames the local event store, rescales all
candidates, increases context cost without better evidence, or produces no
verified provenance. If the TrustGraph endpoint, time-series DSN, or Google
native export is unavailable, mark the path `BLOCKED` and retain the local
fallback without calling it holonic TrustGraph execution.

## Failure policy

- TrustGraph package present, endpoint absent: `BLOCKED`, use local governance
  graph only.
- TrustGraph processing fails: record a local failure event and do not silently
  mark the source processed.
- Time-series write fails: preserve the sealed local event and mark the
  measurement projection `BLOCKED`; do not use an in-memory substitute as
  production evidence.
- Obsidian projection fails: preserve the event and stop the semantic path.
- Audit seal fails: do not append the local governance event.
- CUDA or CI fails: record the real return code and artifact path; do not infer
  task progress from graph coherence.
- Provenance is missing: reject the finding for implementation decisions.

## Required final report

Report each layer with an evidence class:

```text
TrustGraph package:
TrustGraph service:
TrustGraph holonic execution:
Google Drive ingress:
Obsidian event store:
Agentic time-series database:
Hermes audit chain:
HENRI CI / Vast CUDA:
External outcome:
Next falsification:
```

Use `OBSERVED`, `DERIVED`, `INFERRED`, `HYPOTHESIS`, `FALSIFIED`, or `BLOCKED`.
Do not claim that a stable graph, valid audit chain, or successful CI run is an
external task outcome.
