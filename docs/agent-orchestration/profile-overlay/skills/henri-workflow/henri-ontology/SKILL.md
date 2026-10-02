---
name: henri-ontology
description: "Cache first. Use when grounding HENRI claims."
category: henri-workflow
---

# Evidence-grounded ontology

## Cache-first execution contract

Freeze the existing system prefix, tool/schema order, and model roster. Load the bundle once; append task state and new evidence last. Keep static instructions and custom JSON serialization stable. Use deterministic collection and bounded deltas before inference. No padding, empty warm-ups, or loss of correctness/security for hit rate. Measure real provider reads/writes, eligibility, and cost; prefix hashes are not hits. Jev memoization is application caching, not provider KV caching. Full protocol: henri-agent-integration/references/cache-maximization-playbook.md.

## Language and visual tracks

Use HENRI-STE-V1 with the cache contract. Use short active sentences for operational prose.
Target 20 words per instruction and 25 words per description. Use one task per sentence.
Preserve formal code, equations, identifiers, and quotes. Define technical nouns in the ontology.
Use editable diagrams and accessible HTML for substantive user reports and human decisions.
Visual prose and layout remain unrestricted. Keep immediate safety text and a text alternative.
Style findings do not prove compliance or permit execution. Read henri-soul/references/language-visual-protocol.md.

An ontology maps typed terms to sources, code, constraints, and observations. It is not ground truth or latent memory. Own semantic grounding across NotebookLM, Obsidian, Google Drive. EvoOntology arXiv:2609.15779 is a method antecedent; ScientistTwo arXiv:2609.19644v1 guides research gates. Neither proves HENRI gains.

## Source surfaces

| Surface | Role | Required resolution |
|---|---|---|
| Drive | original specs/papers/revisions | API file ID/revision/export hash, or exact mounted path/hash with cloud revision unverified |
| Obsidian | readable notes, contradictions, decisions | exact vault note, original source hash |
| NotebookLM MCP | cited retrieval and synthesis | bank/source ID, raw source content, version/hash when available |

Primary bank: ca4bb787-de9d-4ee0-89c9-bf71259cc86d. Resolve vault through OBSIDIAN_VAULT_PATH. A Drive mount is not API auth or cloud-sync proof. Google-native files need authorized export; .gdoc shortcuts are not document text.

## Grounding

1. Inspect typed manifest/prior decisions for the current mechanism.
2. Search relevant vault/Drive sources. Hash original bytes. Separate approved specifications from descriptive claims.
3. MCP server_info, then one focused notebook_query with citations. Repeat only on changed sources/decision.
4. Resolve cited IDs with source_get_content; check exact support. Synthesis remains INFERRED/HYPOTHESIS.
5. Code claims need definition/caller/consumer/execution evidence. Preserve conflicts, not just the newest narrative.
6. Send compact relevant records, refs, constraints, conflicts, and gaps to the root.

Evidence is claim-specific: code establishes implementation; runs establish behavior; primary text establishes what authors report. None establishes all three. Do not equate the engineering graph with Zone C latent/engram graphs.

## Store and evolution

Existing schema: references/ontology-schema.md. Every family needs probe_ref. Existing store C:/Users/chan/henri-telemetry/ontology/: objects.jsonl, candidates.jsonl, evolution.jsonl, manifest.md. Append-only; explicit supersedes. Schema/edge changes require approval.

Builder: propose → probe → verify → commit. No probe means no committed mapping. Preserve Evidence hashes.

Evolution: diagnose → attribute → one-level patch → paired-validation gate. STRACE provides reduced signatures. Identical budgets and pre-registered margin. Development feedback may guide revision; sealed test answers cannot. No valid held-out gate means benefit BLOCKED. Log rejected families.

## Typed advice boundary

Evidence/store classification is eligible for Jev advice through henri-system1. Do not let the label replace source resolution, schema validation, deterministic provenance, or approval. A source hash or probe must still support each committed mapping. Diagram provenance and policy proof are separate from latent ontology and internal coherence.

## Engineering graph and project memory

CodeGraph supplies SHA-pinned source relationships; dynamic and heuristic edges keep their limits. Project memory stores reviewed operational decisions, references, failures, and next gates; Honcho recall is optional derivative context. Neither replaces cited NotebookLM/Obsidian/Drive sources or proves execution. Use existing term/mapping/constraint/evidence families with probe refs; link SHA/blob/source IDs and existing audit event. No schema merger, raw session upload, benchmark answers, or Zone C payloads.

## Operations

Load notebooklm, obsidian, google-workspace only for their active operations; henri-holonic-graph for store boundaries. Use python "$HERMES_HOME/scripts/nlm_run.py" login --check on this host. After CLI reauth, MCP refresh_auth plus real source read verifies recovery. Do not upgrade MCP mid-session.

External imports/writes need exact-target readback. Exclude benchmark gold answers, latent tensors, and secrets. Revalidate dated counts and method ablation numbers before quoting.

References: references/builder-protocol.md, references/evolution-loop.md, references/ontology-schema.md, references/notebooklm-stack-ops.md, references/reference-index.md.

Historical snapshot: `references/stack-before-20261001.md` is not current policy.
