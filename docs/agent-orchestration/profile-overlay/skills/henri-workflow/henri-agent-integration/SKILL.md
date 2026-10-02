---
name: henri-agent-integration
description: "Cache first. Use when dispatching HENRI work."
category: henri-workflow
---

# Root holon and integration

## Cache-first execution contract

Freeze the existing system prefix, tool/schema order, and model roster. Load the bundle once; append task state and new evidence last. Keep static instructions and custom JSON serialization stable. Use deterministic collection and bounded deltas before inference. No padding, empty warm-ups, or loss of correctness/security for hit rate. Measure real provider reads/writes, eligibility, and cost; prefix hashes are not hits. Jev memoization is application caching, not provider KV caching. Full protocol: henri-agent-integration/references/cache-maximization-playbook.md.

## Language and visual tracks

Use HENRI-STE-V1 with the cache contract. Use short active sentences for operational prose.
Target 20 words per instruction and 25 words per description. Use one task per sentence.
Preserve formal code, equations, identifiers, and quotes. Define technical nouns in the ontology.
Use editable diagrams and accessible HTML for substantive user reports and human decisions.
Visual prose and layout remain unrestricted. Keep immediate safety text and a text alternative.
Style findings do not prove compliance or permit execution. Read henri-soul/references/language-visual-protocol.md.

Own approval, audit linkage, repository delivery, remote execution, and returned evidence. Route source work to research, implementation to architecture, failure attribution to STRACE. Roles can run in the acting agent; wrappers do not create new agents.

## HOLONIC STATIC PREAMBLE — shared contract header

Byte-identical across the triad. Validate after edits with `henri-agent-integration/scripts/validate_holonic_contracts.py`.

- Roles: integration coordinates; research emits SpecContract A; architecture emits HarnessContract B; integration returns ExecutionFeedbackContract C.
- Schemas remain frozen in each skill's `references/holonic-contracts.md`. Graph task/result/state/receipt schemas are separate contracts.
- Evidence classes: OBSERVED, DERIVED, INFERRED, HYPOTHESIS, FALSIFIED, BLOCKED. Models advise; the acting agent executes and verifies.
- MoA authority: henri-moa-routing. Read the live preset; no fixed roster or uncalibrated consensus threshold here.
- Pass intent, constraints, structured outputs, paths, hashes, and bounded evidence. Do not pass raw traces or private reasoning.
- Keep the existing system prefix stable. Append task state. Apply instruction, roster, and plugin edits in a new session.

## Dispatch and approval

Use henri-soul/references/scientisttwo-workflow.md. Register scope, acceptance/rejection, model/GPU/time budget, seeds, data split. Approval is required for founding work, load-bearing math, schemas, production experiments, and destructive changes. Models cannot approve themselves.

Verify graph classes, schemas, and live consumers before calling the graph executable. Frozen A/B/C are not agentic_graph/schemas. GraphRuntime is a control plane, not proof of a scheduler. Optional TrustGraph needs endpoint/flow/result-ID evidence.

## Cryptographic governance

Use existing active-profile scripts/henri_audit.py:

```bash
python "$HERMES_HOME/scripts/henri_audit.py" verify
python "$HERMES_HOME/scripts/henri_audit.py" record henri-arbiter HUMAN_DECISION '<explicit-json-payload>'
python "$HERMES_HOME/scripts/henri_audit.py" verify
```

Read back the exact event hash/payload. Verify before and after append. Stop if integrity fails. Separate source import, approval, patch, push, run verdict, and final decision with causal artifact refs.

The ledger is SHA-256 hash-linked, not signed identity or externally anchored immutability. Its writer has no demonstrated concurrent-append lock. Serialize governed writes; post-append failure blocks downstream actions. Do not alter ledger code/history in an instruction task.

## GitHub sync

1. Fetch/inspect without changing dirty work. Use a clean worktree from approved base.
2. Stage an explicit allowlist. Exclude PDFs, raw telemetry, secrets, binaries, traces, and vendor checkouts.
3. Inspect diff/seals. Commit and push a review branch. Compare local HEAD with git ls-remote for the exact branch. Seal real SHA/push result.
4. Read checks/artifacts for the exact SHA. Cron last_status=ok is not CUDA PASS. Branch push is not main sync.
5. Promote only after required tests and approval. No force push, dirty-tree pull, blanket staging, or bypass of a failed gate.
6. Production sync requires local release SHA = GitHub SHA = tested remote SHA.

Verify branch support. henri-ci may follow main only. Do not merge to trigger a missing branch gate; report BLOCKED and queue an approved remote run.

## Local control-plane execution gate

Use henri-system1 for the new Jev/OpenShell consumers. Plugin henri-control-plane exposes typed advice and guarded execution; activation was checked in a fresh Hermes session. The active-profile scripts/henri_openshell.py obtains the complete effective policy and runs the operator-boundary prover before sandbox exec. Any error/unsupported/inconclusive/violation blocks. Kernel qualification and denial controls are separate evidence. Keep policy changes serialized: proof+exec is not atomic against an operator's concurrent mutation. No host-tool override and no Vast protection claim. Exact procedures: references/openshell-boundary.md.

## Stateful project handoff

Use henri-engineering for reviewed public project records pinned to Git-stored source bytes and exact commit. Local records are canonical; Honcho is an optional projection and remains offline. GitHub sync requires exact remote branch SHA plus content readback. Link ontology mappings and the existing audit event without copying raw ledger/chat/data. A per-repo Honcho session is not repository sync.

## Remote and measurement

Use henri-vast-lifecycle. Default: commit → push → henri-ci → CUDA artifacts → delivery. SSH/SCP only for CI failure, debugging, or approved queued experiment. Check current endpoint, interpreter, CUDA, workdir, and one active run per GPU.

Reduce telemetry before inference. List cron jobs before mutation; do not resume paused jobs without approval. Pin provider/model for agent jobs. Use context_from/workdir where supported. Use henri-co-scientist-rigor before empirical delivery. Return frozen feedback C, real commands/return codes, external outcome, limits, and next falsification. Coherence and graph receipts are not task progress.

References: `references/holonic-contracts.md`, `references/reference-index.md`.

Historical snapshot: `references/stack-before-20261001.md` is not current policy.
