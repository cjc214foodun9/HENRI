---
name: henri-moa-routing
description: "Cache first. Use when routing MoA advice."
category: henri-workflow
---

# MoA routing authority

## Cache-first execution contract

Freeze the existing system prefix, tool/schema order, and model roster. Load the bundle once; append task state and new evidence last. Keep static instructions and custom JSON serialization stable. Use deterministic collection and bounded deltas before inference. No padding, empty warm-ups, or loss of correctness/security for hit rate. Measure real provider reads/writes, eligibility, and cost; prefix hashes are not hits. Jev memoization is application caching, not provider KV caching. Full protocol: henri-agent-integration/references/cache-maximization-playbook.md.

MoA combines tool-less reference advice with one acting aggregator. This skill owns roster and wire policy. Do not copy dated model lists into SOUL, bundle, triad.

## Resolve

Use hermes moa list or python -m hermes_cli.main moa list. Resolve selected preset; inspect preset/legacy blocks after approved config edits. Session model can differ from config. Returned provider/model/trace IDs establish fanout, not config alone.

Official docs: https://hermes-agent.nousresearch.com/docs/user-guide/features/mixture-of-agents . Current public docs use provider-owned output limits. This installed hermes_cli/moa_config.py still normalizes per-slot max_tokens; inspect its live request path before relying on a cap. Config presence and a short-output prompt do not prove enforcement. Record this version difference; do not change settings under an instruction task.

## Wire

References cannot execute tools or approve actions. Aggregator owns execution/readback. References run in parallel and cannot see each other's current output. Ref C cannot judge unseen A/B output; sequential review needs a second explicit packet.

The reference view drops the HENRI system prompt. Local agent/moa_loop.py _reference_messages includes flattened tool previews; current public docs describe a narrower view. Supply essential constraints in task text, not SOUL alone. No secrets. Text-only advice does not prove image inspection; use a real pixel tool.

Request advisory focus in the shared packet with exact live slot labels: A consumer/code/device defects; B mechanism/assumptions/alternatives; C falsification/missing evidence/kill check. Role adherence is not a per-slot system feature. Advisor execution claims require own-tool verification.

## Typed System 1 and kernel boundary

Use henri-system1 for every eligible finite model judgment before a generative escalation. Jev returns advice through the real Decisions API, not a chat slot. The new Hermes hook appends owner advice to the user tail and leaves the original prefix and roster untouched. The typed tool covers escalation, failure, evidence, store, and policy-risk advice. Missing confidence/unknown choices/errors escalate; no auto-approval. Use the OpenShell guarded tool for sandbox-designated commands; do not treat local containment as Vast coverage. Active parent sessions need a new process to load plugin registration.

## Surgical use

Solo tools for reads/edits/retrieval/routine debugging. Delegate independent questions. Use MoA for load-bearing derivation, cross-file audit, or two genuine repair failures. Limit explicit escalations to three/session and one/STRACE cycle. Already-selected MoA can fan out every user turn; this policy does not override runtime. No nested MoA, silent model change, or unmeasured entropy/consensus gate.

Prefer user_turn cadence. No mid-session roster toggles. Announce degraded refs; missing advice is not agreement.

## Packet

Question; advisory slot roles; bounded OBSERVED excerpts/commands/return codes/paths/hashes; constraints/allowed paths/approval; pinned baseline; one candidate and alternative explanation; accept/reject/conditional verdict, contradictions, kill check; budget/stop rule. For time-sensitive sources include a tool-observed current date and authenticated primary URL metadata in the task tail. Model cutoff assumptions do not establish that an identifier is future-dated.

Advice is INFERRED/HYPOTHESIS, not an execution receipt. Inspect assertion snippets before execution; HENRI tests stay CUDA/CI. Use henri-co-scientist-rigor before promotion.

Preserve existing prefix; append state. Measure usage/cost/latency from actual traces. Prefix hash is not a cache hit. Missing aggregator usage is BLOCKED, not zero. Smaller instructions do not establish savings or better outcomes.

Dated detail: references/moa-wire-optimization.md, references/vision-wire-reality.md, references/reference-index.md. Revalidate before use.

Historical snapshot: `references/stack-before-20261001.md` is not current policy.
