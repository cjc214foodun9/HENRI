---
name: henri-architecture
description: "Cache first. Use before HENRI code changes."
category: data-science
---

# Architecture and implementation holon

## Cache-first execution contract

Freeze the existing system prefix, tool/schema order, and model roster. Load the bundle once; append task state and new evidence last. Keep static instructions and custom JSON serialization stable. Use deterministic collection and bounded deltas before inference. No padding, empty warm-ups, or loss of correctness/security for hit rate. Measure real provider reads/writes, eligibility, and cost; prefix hashes are not hits. Jev memoization is application caching, not provider KV caching. Full protocol: henri-agent-integration/references/cache-maximization-playbook.md.

Consume frozen SpecContract A without changing its API. Own caller audit, design, and bounded implementation. Emit HarnessContract B and changed paths. Workflow owner: henri-soul.

## HOLONIC STATIC PREAMBLE — shared contract header

Byte-identical across the triad. Validate after edits with `henri-agent-integration/scripts/validate_holonic_contracts.py`.

- Roles: integration coordinates; research emits SpecContract A; architecture emits HarnessContract B; integration returns ExecutionFeedbackContract C.
- Schemas remain frozen in each skill's `references/holonic-contracts.md`. Graph task/result/state/receipt schemas are separate contracts.
- Evidence classes: OBSERVED, DERIVED, INFERRED, HYPOTHESIS, FALSIFIED, BLOCKED. Models advise; the acting agent executes and verifies.
- MoA authority: henri-moa-routing. Read the live preset; no fixed roster or uncalibrated consensus threshold here.
- Pass intent, constraints, structured outputs, paths, hashes, and bounded evidence. Do not pass raw traces or private reasoning.
- Keep the existing system prefix stable. Append task state. Apply instruction, roster, and plugin edits in a new session.

## Audit before code

Trace definition → caller → data/tensor path → computational consumer → external effect. Verify symbols, shapes, layout, dtype, device, normalization, causal timing, gradient path, and representation family. A stored flag or forwarded argument does not prove engagement. A penalty must change relative candidate scores, not a common scale.

Load applicable constraint references through `references/reference-index.md` before load-bearing edits. Revalidate dated numbers. Do not substitute analogy or a function name for a proof. The retained constraint snapshot is in `references/constraints-before-20261001.md`; it contains historical claims, not unconditional truths.

## Core invariants

- Stabilize the substrate first. One mechanism and one kill experiment per change.
- Preserve defaults; use named default-OFF flags for A/B experiments.
- Keep Cl(3,0) [8192,8], uint8 phase rings, and complex flat waves distinct. Declare adapters at every boundary; no silent flattening or value-range change.
- EDMD stays dual/thin-SVD. No production d² allocations. Clamp rank to sample and feature dimensions.
- Differentiate the intended objective through the live parameter path. Measure engagement and parameter change.
- Normalize dimension-dependent thresholds. Verify the live Sagnac domain; do not interchange normalized inner product and mean phase overlap.
- Use the production learner and consumer in tests. Include an independent negative control that must fail the asserted mechanism.
- No monotonicity demand on noisy learning. No ceiling/floor controls for ranking claims.
- Use zone_c_env.py; preserve environment and attribution gates. No silent surrogate after live persistence failure.
- Preserve the declared zero-pretraining/task-compilation contract. Disclose pretrained backbones and separate their contribution.
- Archive deprecated HENRI code under HENRI V2/_archive/ unless deletion is approved.

## Engineering graph and review

Use henri-engineering for a SHA-bound CodeGraph source inventory before caller audit; inspect its dynamic boundary and heuristic confidence. Simplify changed code only after behavior tests. Apply supplied review guidance to an independent bounded diff/spec/security review before commit; resolve blockers. CodeGraph is source evidence, not executed effect. Record public project-memory refs after exact artifact verification.

## Implement and verify

Write the discriminating test first. Use isolated worktrees and exact paths. HENRI tests run only on Vast CUDA or canonical CI. Local instruction/schema/prompt checks do not establish model performance.

Return code/data hashes, controls, baseline/candidate artifacts, failures, and unmet checks. Broken preflight/nonzero execution is infrastructure BLOCKED, not a scientific verdict. Two same-class genuine repair failures trigger STRACE before the next patch.

References: `references/holonic-contracts.md`, `references/reference-index.md`.

Historical snapshot: `references/stack-before-20261001.md` is not current policy.
