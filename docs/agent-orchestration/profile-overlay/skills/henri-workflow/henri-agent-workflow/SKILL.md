---
name: henri-agent-workflow
description: "Cache first. Use when finding HENRI tools."
category: henri-workflow
---

# HENRI owner map

## Cache-first execution contract

Freeze the existing system prefix, tool/schema order, and model roster. Load the bundle once; append task state and new evidence last. Keep static instructions and custom JSON serialization stable. Use deterministic collection and bounded deltas before inference. No padding, empty warm-ups, or loss of correctness/security for hit rate. Measure real provider reads/writes, eligibility, and cost; prefix hashes are not hits. Jev memoization is application caching, not provider KV caching. Full protocol: henri-agent-integration/references/cache-maximization-playbook.md.

Language follows HENRI-STE-V1 after cache rules. Use soft operational targets and preserve formal bytes.
Use diagrams and accessible HTML for substantive user reports. Visual layout and prose remain unrestricted.
Policy: henri-soul/references/language-visual-protocol.md.

Entry: henri-soul. The fixed bundle contains five HENRI core skills plus diagram priority, explicit upstream Draw.io, and System 1/OpenShell operations. See the actual bundle file for order.

| Policy | Owner |
|---|---|
| loop, gates, budgets | henri-soul |
| sources, autoresearch | henri-research |
| code, tensors, causal contracts | henri-architecture |
| approval, audit, GitHub, CI | henri-agent-integration |
| NotebookLM–Drive–Obsidian | henri-ontology |
| MoA wire, roster | henri-moa-routing |
| failure attribution | henri-strace-optimizer |
| reproducibility, ablation, review | henri-co-scientist-rigor |
| store boundaries, optional TrustGraph | henri-holonic-graph |
| GPU lifecycle | henri-vast-lifecycle |
| telemetry / profiling | henri-telemetry-analyzer / henri-kernel-profiler |

Load active specialists only. No recursive skill calls or automatic MoA trees. Preserve frozen A/B/C. Use existing no-agent collectors before synthesis: active-profile scripts henri_audit.py, henri_cache_audit.py, henri_experiment_digest.py, henri_sync_manifest.py, henri_telemetry_report.py. Verify paths and arguments. Their output proves only its measured scope.

Resolve membership with `python -m hermes_cli.main bundles show henri-bundle`. Validate reload in a separate process, not the live prompt. Retained procedures: `references/reference-index.md`.

Historical snapshot: `references/stack-before-20261001.md` is not current policy.
