---
name: henri-co-scientist-rigor
description: "Cache first. Use before empirical HENRI claims."
category: henri-workflow
---

# Scientific integrity and review

## Cache-first execution contract

Freeze the existing system prefix, tool/schema order, and model roster. Load the bundle once; append task state and new evidence last. Keep static instructions and custom JSON serialization stable. Use deterministic collection and bounded deltas before inference. No padding, empty warm-ups, or loss of correctness/security for hit rate. Measure real provider reads/writes, eligibility, and cost; prefix hashes are not hits. Jev memoization is application caching, not provider KV caching. Full protocol: henri-agent-integration/references/cache-maximization-playbook.md.

Foundation: ScientistTwo arXiv:2609.19644v1 §3.2 baselines, §3.4 ablations, §§3.5–3.6 review experiments, Table 7/Appendix B integrity. Earlier co-scientist/TrueSkill proposals remain historical and conditional, not this paper's method.

## Language and visual tracks

Use HENRI-STE-V1 with the cache contract. Use short active sentences for operational prose.
Target 20 words per instruction and 25 words per description. Use one task per sentence.
Preserve formal code, equations, identifiers, and quotes. Define technical nouns in the ontology.
Use editable diagrams and accessible HTML for substantive user reports and human decisions.
Visual prose and layout remain unrestricted. Keep immediate safety text and a text alternative.
Style findings do not prove compliance or permit execution. Read henri-soul/references/language-visual-protocol.md.

## Four checks

| Check | Evidence |
|---|---|
| score reproduction | pinned-code real rerun; data/evaluator/device/run IDs; return code; metric artifact |
| specification compliance | original rules; unchanged evaluator/split/budget; no leakage or reward hacking |
| reference verification | resolved primary source/version/page supporting exact claim |
| method–code alignment | operation → caller/consumer → engaged computation and ablation |

Hash/schema/lint alone cannot pass these. Consensus and AI review scores are not task outcomes. Paper outcomes are author-reported. HENRI benefit remains HYPOTHESIS until matched evaluation.

## Logs and experiment ladder

Preserve command, SHA, input hashes, environment, run ID, stdout/stderr paths, return code, metric artifacts. Each empirical sentence maps to artifact/excerpt/key. Empty logs mean BLOCKED. Never use fake advisor logs or generated telemetry.

Reproduce baseline first. Small real-data scaffold runs on approved remote target. Before full scale check imports/device/consumer, subsample leftovers, dummy callbacks, hardcoded outcomes, evaluator changes, self-confirming fixtures. Matched arms share split, decoding, memory isolation, and budget.

## Ablation and review

Hold evaluator/readout fixed; change only proposed mechanism. Include negative control with signal on an axis the mechanism cannot preserve. For ranking claims measure engagement and decision changes. Gains caused by unrelated controls cannot support the mechanism.

Convert each material objection to executable check, owner, artifact, stop rule. Do not revise prose just to raise reviewer score. Code changes invalidate dependent results; rerun affected ablations. Keep last accepted candidate unless pre-registered paired gate passes. Record killed families.

Defaults: two engineering repairs, one ablation refinement, one review revision. Larger budgets require explicit scope/approval. STRACE precedes another repeated-failure patch.

Provenance/limits: henri-soul/references/scientisttwo-workflow.md. Retained details: references/reference-index.md. Historical Bayesian tournaments stay OFF unless separately approved and validated.

Historical snapshot: `references/stack-before-20261001.md` is not current policy.
