# Historical instruction snapshot

This is the previous entry file. It is not current policy. Use current SKILL.md for routing. Revalidate every dated runtime claim.

---
name: henri-agent-workflow
description: "Index skill — HENRI skill map, bundle mechanics, deterministic reporting stack. Load with henri-soul."
category: henri-workflow
---

# HENRI Agent Workflow — Index

This skill is an INDEX. Policies live in the specialist skills; the session
entry point is `henri-soul`. Compaction 2026-09-22: long procedural bodies moved
into `references/` (Level 2) or into the owning skill; the pre-compaction body is
in `%LOCALAPPDATA%/hermes/backups/henri-skills-snapshot-20260922-190640.tar.gz`.

## Holonic triad (spec 2026-08-26)

- `/henri-research` — Exploration & Spec Holon. Emits `SpecContract` (A). Never emits code.
- `/henri-architecture` — Implementation Engine Holon. Emits `HarnessContract` (B) + commit.
- `/henri-agent-integration` — Root Holon & Infrastructure Conductor. Owns Vast.ai dispatch, MoA dispatch, Judge-C gating, remote verification, and the `ExecutionFeedbackContract` (C) escalation loop.

Contracts + the HOLONIC STATIC PREAMBLE are byte-identical across the three
skills. After any contract-adjacent edit run
`henri-agent-integration/scripts/validate_holonic_contracts.py` (identity +
payload + plugin-mirror validation; FAIL = rollback).

Master loop: `RESEARCH → SPEC → ARCHITECTURE (MoA) → HARNESS → REMOTE VERIFY →
FEEDBACK → ESCALATE`. Pass distilled payloads only; never raw traces upward.

## Skill map

| Skill | Use for |
|---|---|
| `henri-soul` | **session entry**: bootstrap, canonical loop, contracts, audit commands, diagram mandate |
| `henri-ontology` | NotebookLM as foundational ontology: typed records, browse/resolve, builder + evolution loop |
| `henri-research` | source order, orx autoresearch, vault ingestion, evidence discipline, conditional routing |
| `henri-architecture` | invariants, causal planning, tensor contracts, falsification checklist |
| `henri-agent-integration` | cache/token spine, release gate, CoE gates, remote verification, repo hygiene |
| `henri-strace-optimizer` | trace root-cause loop: EDG → trace reduction → causal slice → policy proposal |
| `henri-moa-routing` | MoA policy + wire mechanics (single MoA authority) |
| `henri-vast-lifecycle` | Vast.ai lifecycle + Mutagen sync (single Vast authority) |
| `henri-kernel-profiler` | remote nsys/ncu profiling |
| `henri-telemetry-analyzer` | W&B/JSONL run audit |
| `henri-holonic-graph` | TrustGraph holons, Drive→Obsidian provenance, time-series events |
| `henri-co-scientist-rigor` | E_log claim clipping, scaffold audits, UCB ranking |
| `notebooklm` | direct NotebookLM bank operations (MCP + `nlm_run.py` shim) |
| `context-handoff` | context ceiling migration |

## Bundle mechanics

- `/henri-bundle` loads the whole core in one message. File:
  `%LOCALAPPDATA%\hermes\skill-bundles\henri-bundle.yaml`; managed by
  `hermes bundles list|show|reload`.
- Add a member: edit the YAML, `hermes bundles reload`, then verify the member
  count with `hermes bundles list`. Keep each SKILL.md `description` inside the
  ~60-char trigger budget (first 57 chars must stand alone).
- Size ceiling: a SKILL.md near ~100,000 characters is at the wall (99k is the
  practical ceiling). Verify with `wc -c` after every patch to a large SKILL.md;
  when over, move the newest body section into `references/` behind a one-line
  pointer (`henri_skill_compaction.py` automates this pattern).

## Deterministic reporting stack (`%LOCALAPPDATA%\hermes\scripts\`)

Use these BEFORE any LLM reads telemetry. stdlib-only unless noted.

| Tool | Purpose |
|---|---|
| `henri_cache_audit.py [--newest N] [--json f]` | per-slot MoA prompt-cache economics: fresh/cache-read tokens, hit %, spend |
| `henri_experiment_digest.py --root <dir>/experiments/verification` | all scorecards in one table + PASS/FAIL vs pre-registered `accept_margin`; `--figure`, `--md`, `--json` |
| `henri_sync_manifest.py verify` | four-surface topology drift check (GitHub / local / Drive / Vast) |
| `henri_ast_dedup_probe.py` | AST name-signature collision rate before building any dedup gate |
| `henri_telemetry_report.py` | one unified HTML + MD report across layers, figures embedded |

Rendering needs matplotlib, which the Hermes interpreter lacks. Use the
provisioned venv: `%LOCALAPPDATA%\hermes\viz-venv\Scripts\python.exe` (uv;
matplotlib 3.11.2).

**Vision loop (verified 2026-09-11).** Workers render a figure; `vision_analyze`
reads it as a coarse filter; the numeric table stays authoritative. Vision never
overrides a number.

Measured baseline (comparison point): MoA prompt-cache hit **25.5%**, p50 prompt
~137k tokens, reference spend $19.49 over 8 traces / 285 slot calls. The trace
AGGREGATOR slot carries no `usage`/`cost_usd` field → aggregator cache and spend
are `UNVERIFIED`; never quote an aggregator cost figure.

Full doctrine: `henri-agent-integration/references/four-surface-sync-and-reporting.md`.

## Store boundaries (do not merge)

Vault (Obsidian event log) · TrustGraph (holonic context) · agentic time-series
(compact operational events) · Zone C (latent artifacts) · `strace_output/`
(derived analysis) · the ontology store (`henri-telemetry/ontology/`, typed
semantic records). Different schemas, owners, retention, and evidence meanings.
Protocols: `henri-holonic-graph`; vault/Zone C envelope:
`references/agentic-graph-vault-zonec.md`.

## Read-only Drive triage

Moved to `henri-research` (source order + evidence discipline + Drive-ingest
canonical paths). Use that skill's `references/` files; do not restate them here.

## Token & execution rules

1. **All HENRI tests and benchmarks run on the Vast CUDA target or `henri-ci`. Never substitute local CPU tests** for model/kernel verification.
2. Never dump raw logs; failure-first summaries with paths to the full artifact.
3. Vault first, then the NotebookLM ontology, then web/arXiv when freshness matters.
4. MoA is expensive: reserve for load-bearing derivation, cross-file audits, or a bug surviving two genuine fixes. Max 3 calls per session.
5. `execute_code` for multi-step logic; scripts for reduction.
6. Vast.ai runs go in background; pull telemetry before stop; egress gate before stop.
7. Reference/advisor output can fabricate environment facts (paths, versions, log lines, byte counts). Verify every environment claim with a direct disk/process/log read; where a reference and your own probe disagree, your probe wins.

## Remote execution invariant

- Remote venv: `/venv/main/bin/python3` (PyTorch + CUDA + `arc_agi`); NOT system python.
- Workspace root `/workspace/HENRI V2`; execution dir `/workspace/HENRI V2/HENRI V2`.
- Full procedure, SSH diagnosis order, cost guards, templates: `henri-vast-lifecycle`.

## Plugin restart requirement

`hermes plugins enable` writes to config, but Hermes loads plugins at startup.
`/reset` does NOT reload the plugin loader — restart Hermes entirely.

## MoA cost model (from session 2026-07-21)

A `/moa` call routes through 3 references + 1 aggregator in parallel (~4–5× a
single call; a broad "review the codebase" prompt burned $50 in one turn). Be
surgical: "fix this function given this error trace", not "evaluate this area".

## References (Level 2)

- `references/empirical-calibration-receipts.md` — honest calibration receipts: store per-task rows, self-check at T=1.0 with a provenance-justified tolerance, fit temperature by NLL (never ECE), gate on `ECE<=0.05 AND BSS>0`.
- `references/adversarial-control-design.md` — which controls discriminate and which only appear to; the one valid design (hold the readout FIXED, swap only the operator).
- `references/spec-provenance-and-phantom-citations.md` — fingerprint an attached spec before importing its numbers; confirm claimed edits by reading the TARGET, never the claim.
- `references/byte-pinned-artifacts-and-line-endings.md` — SHA-pinned artifact fails on a FRESH worktree → `core.autocrlf` checkout conversion, not a content regression; diagnosis order + `.gitattributes text eol=lf` fix.
- `references/digest-canonicalization-and-publication.md` — a published sha256 that does not reproduce is still a defect; cite `sha256(LF-normalized bytes)`.
- `references/intervention-validity-and-confirmation-bias.md` — prove a feedback/steering term CAN flip the scored decision before crediting it with a gain; emit decision-change rate beside every accuracy delta.
- `references/agentic-graph-vault-zonec.md` — event envelope, projection manifest, kill experiment, repository allowlist.
- `references/benchmark-reporting-protocol.md` — strict empirical benchmark reporting rules (no staging status).
- `references/moa-preset-config.md` — MoA preset config mechanics; `references/hermes-plugin-authoring.md` — plugin authoring + restart rules.
- `references/chromadb-time-filter-pitfall.md`, `references/subprocess-env-isolation.md`, `references/torch-version-pinning.md`, `references/vault-endpoint-reference.md` — environment traps (ChromaDB time filters, subprocess env leakage, torch pinning, vault endpoint).
