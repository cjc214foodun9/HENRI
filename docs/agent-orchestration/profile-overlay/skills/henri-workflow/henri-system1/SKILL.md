---
name: henri-system1
description: "Cache first. Use for typed judgments and sandbox runs."
version: 0.1.0
category: henri-workflow
---

# HENRI typed decisions and OpenShell execution

## Cache-first execution contract

Keep role/rubric/schema/model versions stable. Append only the bounded state and source hashes. Batch independent questions sharing state. Use exact-request application memoization, not a claimed Jev provider cache. Changed evidence invalidates reuse. Security, authorization, correctness, and human approval outrank caching. Full protocol: henri-agent-integration/references/cache-maximization-playbook.md.

## Eligible System 1 work

Every model-based judgment that fits an approved finite answer space uses Jev first: owner routing, MoA escalation advice, failure taxonomy, evidence-kind labeling, store triage, and policy-risk advice. Use the existing active-profile scripts/henri_system1.py caller and its named rubric registry. The enabled henri-control-plane plugin appends owner advice in a fresh Hermes HENRI turn; its typed tool handles the other rubric kinds. The acting agent consumes the result and performs the next authorized step; no automatic production dispatcher is implied. It is a cache-safe user-tail hook, not a generative MoA slot or a host-wide interception layer.

Do not send arithmetic, hashes, dates, schema validation, filesystem permission decisions, approval, policy proof verdicts, or task success to a probabilistic model. Keep them in code or with the human. Novel theory/code/prose and unsupported states escalate to the acting reasoner. Jev never replaces a generative MoA reference or aggregate model.

## Jev invocation

From Hermes terminal, use:

```bash
python "$HERMES_HOME/scripts/henri_system1.py" decide --state-file "<bounded-state.json>" --kind owner --out "<outside-git-receipt.json>"
```

Pin typesafe/jev-1.13 on POST https://openrouter.ai/api/alpha/decisions. Use Hermes credential resolution; never print/save the key. Return actual model/provider/generation ID, input/output hashes, answer, latency, usage cost, and application-hit status. Semantic answers remain INFERRED. Missing credentials, HTTP failures, malformed answers, unknown options, ambiguous/stale state, or low confidence produce escalation without execution.

Confidence is probability concentration, not correctness or security proof. Threshold 0.85 is a pilot routing rule only, not calibrated authority. Auto-action promotion requires separately approved labeled calibration. The observed first live call was about 401 ms, not the supplied <50 ms claim.

## OpenShell invocation

For new sandbox-designated commands, use the real active-profile scripts/henri_openshell.py gate. Verify current gateway, sandbox, complete effective policy, operator-owned boundary, and coverage before dispatch. Only prover result within_boundary permits a declared command. Every nonzero/unsupported/inconclusive/error result blocks; no structural-key fallback or unsandboxed retry.

OpenShell v0.1.2 runs in Ubuntu WSL for this pilot; Windows support is experimental. Use real policy schema version: 1 with filesystem_policy, landlock, process, network_policies, network_middlewares. Gateway schema is rooted at openshell version 2. No arbitrary syscall/rlimit fields from the attached blueprint. Local WSL containment is not Vast GPU containment.

No host secrets, repo write mounts, Docker socket, or production data enter the workload. Human approval governs policy expansion. CLI/prover installation alone is not a qualified sandbox. Do not alter the active Hermes terminal backend mid-session.

References: henri-moa-routing/references/jev-systemone-integration.md; henri-agent-integration/references/openshell-boundary.md. Real Jev, proof, execution, and telemetry receipts stay distinct.
