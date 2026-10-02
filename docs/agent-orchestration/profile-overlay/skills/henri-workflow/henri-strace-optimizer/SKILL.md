---
name: henri-strace-optimizer
description: "Cache first. Use when HENRI failures repeat."
category: henri-workflow
---

# STRACE attribution holon

## Cache-first execution contract

Freeze the existing system prefix, tool/schema order, and model roster. Load the bundle once; append task state and new evidence last. Keep static instructions and custom JSON serialization stable. Use deterministic collection and bounded deltas before inference. No padding, empty warm-ups, or loss of correctness/security for hit rate. Measure real provider reads/writes, eligibility, and cost; prefix hashes are not hits. Jev memoization is application caching, not provider KV caching. Full protocol: henri-agent-integration/references/cache-maximization-playbook.md.

STRACE (arXiv:2607.07702) analyzes agent trajectories; it is not Linux strace. Use real records to localize failure before policy repair. Derived slices are not causal proof. Do not implement production code in this role.

Trigger: two same-class failures or two genuine repairs without external gain. No traces means BLOCKED. Reuse prior if topology unchanged. One trace/named manifestation can go directly to slice; existing attribution can go directly to proposal.

## Language and visual tracks

Use HENRI-STE-V1 with the cache contract. Use short active sentences for operational prose.
Target 20 words per instruction and 25 words per description. Use one task per sentence.
Preserve formal code, equations, identifiers, and quotes. Define technical nouns in the ontology.
Use editable diagrams and accessible HTML for substantive user reports and human decisions.
Visual prose and layout remain unrestricted. Keep immediate safety text and a text alternative.
Style findings do not prove compliance or permit execution. Read henri-soul/references/language-visual-protocol.md.

## Tools and stages

Inspect installed scripts/trace_reducer.py and scripts/causal_slice.py --help. Use real corpus and verified EDG. Output outside Git at C:/Users/chan/henri-telemetry/strace_output/<run_id>/.

1. Structure: refresh EDG only on topology change; verify nodes/edges against live callers.
2. Reduce: summaries, severity, patterns, selection. Report matched keys and coverage. Missing outcome/node mappings mean unknown severity.
3. Attribute: backward slice narrows suspects, not causes. fallback=true retains all positions and signals a mapping gap. Read bounded evidence positions; state competing causes and discriminating intervention.
4. Propose: ≤3 reusable rules for the root node, exact target, predicted effect, held-out A/B kill criterion. Human approval precedes skill/contract edits.

Input planning bounds: structure 2K, selection 4K, slice 6K/trace, proposal 8K tokens. Verify sizes when tokenizer exists. One MoA escalation/cycle. Stop when trace schema cannot support attribution.

One approved root-node patch, not global rewrite. Failed paired gate retains parent. Two failed A/Bs kill that rule family. Ontology uses reduced patterns, not automatic acceptance.

Vendor STRACE-main under agentic_graph is separate from installed scripts; presence proves neither integration nor license permission. Do not publish vendor checkout as a skill-edit side effect.

References: references/henri-edg.json, references/henri-edg.md, references/stage1-env-modeling.md through references/stage4-policy-synthesis.md; references/reference-index.md. Old numeric kill rules require calibration.

Historical snapshot: `references/stack-before-20261001.md` is not current policy.
