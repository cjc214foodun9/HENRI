---
name: henri-diagrams
description: "Cache first. Use when HENRI material has structure."
version: 0.1.0
category: henri-workflow
---

# HENRI diagram priority

## Cache-first execution contract

Load this policy once. Keep the existing prefix and schema order stable; append task state last. Do not paste XML, images, full graph dumps, or repeated render logs into model context. Reuse stable node IDs and cached source inventories. Hashes prove artifact identity, not provider cache hits. Protocol: henri-agent-integration/references/cache-maximization-playbook.md.

## Language and visual tracks

Use HENRI-STE-V1 with the cache contract. Use short active sentences for operational prose.
Target 20 words per instruction and 25 words per description. Use one task per sentence.
Preserve formal code, equations, identifiers, and quotes. Define technical nouns in the ontology.
Use editable diagrams and accessible HTML for substantive user reports and human decisions.
Visual prose and layout remain unrestricted. Keep immediate safety text and a text alternative.
Style findings do not prove compliance or permit execution. Read henri-soul/references/language-visual-protocol.md.

## When to use

Produce an editable diagram for architecture, workflow, trust boundaries, lineage, causal hypotheses, state machines, or module relations. Use measured plots for numeric results. Use a visual report for substantive HENRI status and HITL decisions. Use creative/claude-design for HTML, not the ambiguous bare name. Layout and explanatory prose are unrestricted; preserve source truth, safety text, and accessible alternatives. A short acknowledgment or viewer failure can use plain text. No decorative graph or forced design fan-out.

## Procedure

1. Invoke installed drawio-skill explicitly. Its upstream disable-model-invocation flag is respected; this policy supplies the explicit trigger. Use its scripts/diagramctl.py with python on native Windows, not missing python3.
2. Resolve sources and evidence classes before drawing. Use Diagram IR, stable IDs, and provenance. Label proposed, observed, and blocked paths. Import source when possible; source extraction is not runtime observation.
3. Build editable .drawio, validate structure with scripts/validate.py, and run semantic review/test where applicable. Preserve layout on local edits; do not prune unless requested.
4. Render a local self-contained Story HTML or native export. Native PNG/SVG/PDF requires a real draw.io executable; never claim an export without it. Inspect layout through browser DOM or vision where available. Deliver editable source and preview separately.
5. Keep secrets, raw model traces, benchmark answers, and latent tensors out of diagram properties. Numeric telemetry remains authoritative.

## Limits and verification

The upstream skill is installed from a pinned commit and its missing support files were restored from that same checkout. No new MCP or GUI binary is implied. Stop after two layout-repair rounds. Record exact source/output paths and hashes; check nodes/edges against the source and distinguish design from execution.

Full diagram policy: henri-soul/references/diagram-mandate.md. Artifact generation does not prove any diagrammed service is live.
