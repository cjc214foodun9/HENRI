# HENRI-STE-V1: language and visual protocol

## Cache first

Load this fixed policy once. Keep the system prefix, tool order, and model roster fixed for the session.
Append task state and new evidence last. Do not rewrite prior messages to change their style.
Apply disk edits in a new session. Do not add model calls for grammar repair.
Reuse exact source terms and compact role packets. Measure provider cache reads and costs; a hash does not establish a hit.

## Operational track

Use short sentences with one task or topic. Name the actor for a result or a failure.
Use the same term for the same meaning. Include the evidence class and the next gate.
Target 20 words per instruction. Target 25 words per description. Target six sentences per paragraph.
Target three words per noun group. Use a defined shorter term if a technical noun exceeds this target.
Use commands and simple present, past, or future verbs. Infinitives and participle adjectives can remain.
Avoid progressive and perfect verbs. Do not use passive procedures. Do not ban every word that ends in `ing`.
Use technical nouns and verbs from the ontology. Cite their definition, source, part of speech, and scope.
Do not claim that a HENRI term has official ASD approval.

## Formal track

Keep code, argv, schemas, equations, units, identifiers, hashes, paths, and source quotes exact.
Use unrestricted mathematical English for derivations. Do not change a technical object to fit a word target.
Do not compile natural-language commands into shell code. Schema, proof, permission, and human approval checks remain separate.
Keep frozen A/B/C headers and examples unchanged. Historical source records remain historical.

## Visual track and HITL

Use diagrams or HTML for substantive user reports, review choices, approvals, status, and structured explanations.
Use Draw.io for editable graphs. Use `creative/claude-design` for HTML; the bare name is ambiguous on this host.
Use measured plots for numerical evidence. Keep editable source and accessible text with each visual artifact.
Do not restrict layout, graph richness, equations, labels, or explanatory prose with STE targets in this track.
Use the smallest useful view. Do not add decoration, fake data, forced graph counts, or a design fan-out.
The visual exemption does not permit unsafe claims, hidden decisions, secrets, fabricated evidence, or inaccessible safety content.
Show evidence classes, source links, exact IDs, and limits. Preserve the user's decision and approval scope.
State urgent safety text in plain text first. Use plain text if the viewer fails or the user requests it.
A short acknowledgment can remain plain text. Do not delay a safety warning to render a diagram.
A diagram or button does not grant approval. Use the actual approval transport; do not fake a consent action.

## Safety markers

Use `WARNING:` for injury or death risk. State the condition and the risk.
Use `CAUTION:` for equipment damage, state damage, or data loss. State the condition and the risk.
Do not use `WARNING:` for a disk error alone. Do not infer OCSF severity, status, or task success from a label.
Keep safety text accessible even when the rest of a page uses unrestricted prose.

## Source authority

Official source: ASD-STE100 Issue 9, 2025-01-15. It replaces prior issues.
Primary rule locators: 2.1/2.2, 3.2–3.6, 5.1/5.2, 6.3/6.6, and 7.1–7.3.
The local PDF receipt pins SHA-256 `d1f4ea9e7cd6e46b47aa9057209f99e78c0e9cfc4e27a5b07895b05c1a166431`.
PDF pages: 63, 68–71, 87, 95, 103, 150, and 166.
The supplied document names Issue 8. Its approximation and safety mappings conflict with Issue 9.
Use `approximately` for an estimate. Do not substitute `about`, which has a different approved meaning.
No full official dictionary is bundled. The checker and skills adapt principles; they do not certify full compliance.
Do not reproduce or publish the standard PDF or its dictionary without the applicable rights.

## Deterministic advisory checker

Use the active-profile `scripts/henri_language.py` through `terminal`:

```bash
python "$HERMES_HOME/scripts/henri_language.py" --file "<text-file>" --track operational --mode procedure
python "$HERMES_HOME/scripts/henri_language.py" --file "<text-file>" --track visual
```

The checker reports selected word/verb patterns and heuristic counts. Formal and visual tracks skip style checks.
It does not rewrite text. It does not authorize execution, infer noun groups, or validate every meaning or part of speech.
False positives and false negatives remain possible. Review findings once; do not start a regeneration loop.
Malformed input returns BLOCKED. Style findings return ADVISORY and do not replace safety or correctness gates.
Jev supplies eligible typed advice. It does not validate grammar, hashes, schema, proof, permission, or final success.

## Real consumers and middleware boundary

The existing Hermes hook appends this policy to the user tail in a new process.
MoA reference packets carry the same fixed guide because references do not receive SOUL.
The OpenShell consumer assesses an optional operational description and keeps command argv exact.
It then uses the existing complete-policy proof and named sandbox checks. Every failed proof still blocks.
This is a host preflight, not native Supervisor middleware or a host-wide interception layer.

OpenShell v0.1.2 supports external gRPC middleware. It needs a reachable service, gateway registration, restart, and policy attachment.
The service must implement the pinned protocol and authentication rules. It cannot inspect every command or every transport.
No service attachment or gateway/policy expansion is part of this change. Native L7 enforcement remains BLOCKED_NOT_ATTACHED.
Do not copy a Python file into a Rust crate and call that an installed middleware.
Do not disable authentication or use `fail_open` to hide a failed prerequisite.

## Verify

Test positive and refusal cases on the real checker. Test exact formal/visual byte preservation.
Test the hook, MoA packet, and existing guarded command path. Inspect return codes and artifacts.
Keep language clarity, source truth, sandbox proof, external outcome, and provider cache evidence separate.
No measured language error reduction, 1:1 AST compilation, semantic-entropy elimination, or workflow-speed gain follows from installation.
