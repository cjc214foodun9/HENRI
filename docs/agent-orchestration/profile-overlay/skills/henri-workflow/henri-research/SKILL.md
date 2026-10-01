---
name: henri-research
description: "Cache first. Use when researching HENRI."
category: henri-workflow
---

# Research holon

## Cache-first execution contract

Freeze the existing system prefix, tool/schema order, and model roster. Load the bundle once; append task state and new evidence last. Keep static instructions and custom JSON serialization stable. Use deterministic collection and bounded deltas before inference. No padding, empty warm-ups, or loss of correctness/security for hit rate. Measure real provider reads/writes, eligibility, and cost; prefix hashes are not hits. Jev memoization is application caching, not provider KV caching. Full protocol: henri-agent-integration/references/cache-maximization-playbook.md.

Own source resolution and SpecContract A. Do not implement production code. Use henri-soul for workflow; henri-ontology for source grounding.

## HOLONIC STATIC PREAMBLE — shared contract header

Byte-identical across the triad. Validate after edits with `henri-agent-integration/scripts/validate_holonic_contracts.py`.

- Roles: integration coordinates; research emits SpecContract A; architecture emits HarnessContract B; integration returns ExecutionFeedbackContract C.
- Schemas remain frozen in each skill's `references/holonic-contracts.md`. Graph task/result/state/receipt schemas are separate contracts.
- Evidence classes: OBSERVED, DERIVED, INFERRED, HYPOTHESIS, FALSIFIED, BLOCKED. Models advise; the acting agent executes and verifies.
- MoA authority: henri-moa-routing. Read the live preset; no fixed roster or uncalibrated consensus threshold here.
- Pass intent, constraints, structured outputs, paths, hashes, and bounded evidence. Do not pass raw traces or private reasoning.
- Keep the existing system prefix stable. Append task state. Apply instruction, roster, and plugin edits in a new session.

## Procedure

1. Inspect the live question and repository. Resolve relevant vault, Drive, and NotebookLM evidence; do not query all banks by default.
2. Verify the baseline and its limits. Distinguish a binding approved specification from a claim of implementation.
3. For literature, use existing autoresearch when present in the selected checkout, then primary sources and current official docs. Failed retrieval is not proof of absence.
4. Return at most two initial hypotheses with mechanism, assumptions, data path, resource cap, failure mode, and cheapest discriminating experiment. Larger budgets need explicit scope.
5. Emit frozen SpecContract A plus resolved artifact refs. Preserve contradictions and killed hypothesis families.

## Autoresearch is retrieval

From a checkout containing the module, use HENRI V2 as workdir:

```bash
python -m agentic_graph.autoresearch_cli --probe
python -m agentic_graph.autoresearch_cli --query "<question>" --limit 8 --out "<outside-git-dir>" --receipts "<outside-git-dir>/receipts.jsonl"
```

Inspect binary version/hash and return codes. Limits: 3 strategies, 3 primitive calls, 90 seconds, result limit ≤25. Receipts establish fetched bytes, not scientific truth or capability. Capability needs executed mechanism/outcome evidence; lint or hash alone is insufficient. No-login retrieval does not imply authenticated experiment execution or agent spawning. Confirm a real harness before those tiers. Do not assume main contains carrier-only files.

Pin paper version, raw-byte SHA-256, page/section, and artifact. Resolve NotebookLM source IDs. Its synthesis is INFERRED/HYPOTHESIS, not runtime observation. Use official docs for changing APIs; browser/Firecrawl for difficult pages; Apify only through a verified installed interface.

References: `references/autoresearch-grounding.md`, `references/reference-index.md`. ScientistTwo workflow: henri-soul; rigor: henri-co-scientist-rigor.

Historical snapshot: `references/stack-before-20261001.md` is not current policy.
