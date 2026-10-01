# HENRI reviewed project memory

This directory stores public, content-addressed engineering observations, not raw sessions, cited ontology truth, or Zone C latent state. The existing governance ledger stays outside Git. Records link exact source commit + Git blob + SHA-256; a record hash proves linkage, not factual correctness or signer identity.

## Workflow

1. Resolve cited ontology and exact current source SHA.
2. Audit the source graph and live callers, then implement within approved scope.
3. Run discriminating checks and independent review; preserve failures and limits.
4. Add a bounded public record with `profile-overlay/scripts/henri_project_memory.py` or the active-profile installed script. Add requires --reviewed-public after content review. This is caller acknowledgement, not signer/approval proof. Pattern guards reject tested spellings only; arbitrary secret/answer text cannot be proven absent by regex. Read back and review the record before staging.
5. Commit named paths, push a review branch, and run `remote-verify` for its exact SHA and committed record objects. No pull/overwrite, background scheduler, auto-main merge, or Git credential is embedded.
6. Query by explicit source SHA. Reuse of an unrelated SHA refuses; it is history, not current context. Results append to the task tail.

Examples via Hermes terminal:

```bash
python "<installed-script>" query --repo "<clean-worktree>" --commit "<full-source-SHA>" --query GraphRuntime
python "<installed-script>" verify --repo "<clean-worktree>"
python "<installed-script>" remote-verify --repo "<clean-worktree>" --branch feat/engineering-project-memory
```

Local file scanning is deliberately simpler than another vector store. The generated CodeGraph DB and source snapshot stay outside Git. Static graph extraction may miss dynamic calls; heuristic edges cannot establish executed paths. This installation indexed 12 exact committed control-plane Python files, not the whole HENRI codebase.

## Honcho offline boundary

The user selected offline. The supplied plugin is installed at pinned upstream `32dfd0ba62ae0e8dad82d55fc81515e8c4a181a9` and remains inactive. Native doctor fails because `agent.turn_author` is absent. No SDK/server fake or compatibility stub was added. No raw conversation, automatic migration, or dialectic call ran.

`honcho-sync` is intentionally a refusal while offline, not a working remote projection. A future sync needs a compatible plugin/runtime, an approved self-host endpoint/auth, project-scoped peers, no automatic message/migration path, and exact conclusion create/list readback. Installation alone is not remote memory. Native `saveMessages:false` does not gate memory-file migration or explicit tool writes.

## Evidence and limits

See `evidence/` and the existing instruction-overlay verifier under `docs/agent-orchestration/`. Local infrastructure checks establish deterministic records, provenance, known-symbol retrieval, and refusal controls only. No GPU model performance, broad workflow speedup, routing accuracy, cloud Drive revision, Honcho server, or main promotion is established.

NotebookLM source readback and Obsidian/Drive mounted note links are mapped via existing ontology term/mapping/constraint/evidence families; they are not merged with project-record schema. A corpus synthesis that conflates cited ontology with runtime ontology-error signals is kept as a conflict, not adopted.
