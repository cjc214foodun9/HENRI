# Historical instruction snapshot

This is the previous entry file. It is not current policy. Use current SKILL.md for routing. Revalidate every dated runtime claim.

---
name: henri-soul
description: HENRI V2 session operating system — load for grounded research, audit, implementation, GPU verification, telemetry, MoA, cron, and Hermes tooling.
category: henri-workflow
---

# HENRI SOUL — OPERATING SYSTEM

Use this skill as the HENRI session entry point. Load specialist skills only when their domain is active. Keep the system falsifiable, causal, and economical with context.

## 1. Session bootstrap

1. Read the repository `SOUL.md`, latest handoff, project rules, and current git state.
2. Inspect live code before trusting documentation or external design claims.
3. Query the local HENRI vault first for project-specific theory and telemetry.
4. Use web research when the question depends on current Hermes, Firecrawl, Apify, or external literature behaviour. The old “do not use the internet” rule is obsolete.
5. Record material decisions and evidence in the HENRI audit chain. Do not treat a chat transcript as the governance ledger.

Load when needed:
- `henri-architecture` — invariants and falsified mechanisms.
- `henri-ontology` — NotebookLM as the foundational ontology: typed records, browse/resolve, builder + evolution loop (EvoOntology method).
- `henri-moa-routing` — MoA routing, wire mechanics, and cost policy (the single MoA authority).
- `henri-vast-lifecycle` — Vast search, provision, Mutagen sync, execution, egress, termination (the single Vast authority; `vast_lifecycle` and `henri-vast-sync` are pointer stubs).
- `henri-kernel-profiler` — Nsight profiling.
- `henri-telemetry-analyzer` — W&B and JSONL audit.
- `henri-research` — arXiv, orx autoresearch, Firecrawl, and vault ingestion.
- `henri-holonic-graph` — TrustGraph holons, Drive-to-Obsidian provenance, agentic time-series events, and store boundaries.
- `context-handoff` — context ceiling migration.
- `notebooklm` — direct bank operations (MCP tools + the `nlm_run.py` shim) when the ontology skill needs the raw surfaces.

## 2. Canonical research-to-evidence loop

`RESEARCH → AUDIT → DESIGN → APPROVAL → IMPLEMENT → REMOTE VERIFY → MEASURE → INFER`

- **Research:** vault first, then the NotebookLM corpus (MCP bank `ca4bb787-de9d-4ee0-89c9-bf71259cc86d`) as the always-on consult subagent, then current web sources or arXiv when freshness or external evidence requires it.
- **Audit:** verify every cited file, API, dependency, tensor shape, and causal path against the live repository. Reject phantom APIs and diagnostic mock loops.
- **Design:** whiteboard the mechanism, assumptions, failure mode, and a cheap kill experiment before implementation.
- **Approval:** no code before approval for founding-exercise plans or load-bearing math changes.
- **Implement:** one bounded change at a time. New behaviour is flag-gated and OFF by default unless the experiment explicitly changes the default.
- **Remote verify:** all HENRI tests and production-scale benchmarks run on the Vast CUDA target or canonical CI. Never substitute local CPU tests.
- **Measure:** collect failure-filtered telemetry, not raw log dumps. Preserve run IDs and paths.
- **Infer:** distinguish internal coherence, observable frame change, and externally reported task outcome. Never call a self-consistency signal task progress.

## 2A. NotebookLM corpus consult (always-on subagent)

The NotebookLM MCP is the project's curated theory/philosophy bank. It is authenticated (`auth_status=configured`, MCP v0.9.6, verified 2026-08-03) and treated as an always-on consult subagent in every workflow phase:

- RESEARCH: before external search, query the corpus for prior treatment of the mechanism, its philosophy, and its stated assumptions.
- DESIGN: ask how the corpus frames the mechanism (physical interpretation, known limits, labeled falsifications).
- MEASURE: after any test execution, ask the corpus to interpret telemetry (Sagnac delta, Kuramoto r, EFE decomposition, veto counts, learning engagement).
- INFER: ask whether a proposed attribution contradicts any corpus statement; record conflicts.

Primary bank: `ca4bb787-de9d-4ee0-89c9-bf71259cc86d` (HENRI philosophy, 217 sources). Related banks and full MCP tool map: `henri-research` (corpus-consult section).

Evidence rules:

- One focused question per query; request citations ("which sources support this?").
- Label corpus answers `INFERRED`/`HYPOTHESIS`, never `OBSERVED` for telemetry the corpus did not generate.
- Live code and measured CUDA telemetry OVERRIDE corpus claims on conflict.
- Auth probe: `python "%LOCALAPPDATA%\hermes\scripts\nlm_run.py" login --check` (deterministic; `nlm.exe` itself is OS-blocked). If stale, re-run `... nlm_run.py login`.

Auth watchdog: cron job `henri-notebooklm-auth-watchdog` (no-agent, daily 09:00) runs the check via `C:\Users\chan\AppData\Local\hermes\scripts\henri_notebooklm_auth_watchdog.py`. Silent when valid; alerts with re-auth instructions when stale or broken. Do not create a second auth watchdog. Keep it a `.py` file: the cron scheduler runs `.sh` through bash, which strips the backslashes in Windows absolute paths and fails with exit 127 (observed 2026-08-03). The watchdog uses the CLI shim because `nlm.exe` is BLOCKED by Windows Application Control (WinError 4551); the working invocation is `python "%LOCALAPPDATA%\hermes\scripts\nlm_run.py" <args>`.

## 2B. Ontology layer (EvoOntology method)

NotebookLM is not only a corpus to consult — it is the **content layer** of the session ontology (method: EvoOntology, arXiv:2609.15779; skill `henri-ontology`). Load `henri-ontology` when you need any of:

- **browse(q)** — semantic probe of the banks (MCP `notebook_query` / `cross_notebook_query`); one question per call, citations requested.
- **resolve(I)** — turn a match into records (`source_get_content`, `note list`, `chat_get`); never cite an unresolved match.
- **typed store** — `C:\Users\chan\henri-telemetry\ontology\{objects,candidates,evolution}.jsonl`; every record needs a `probe_ref` (no probe, no commit).
- **evolution loop** — diagnose → attribute → patch → gate; the gate and the attribution are the load-bearing steps; rejected candidates are logged, never retried blind.

Evidence rule: browse answers are `INFERRED`/`HYPOTHESIS`; live code and measured telemetry override them.

## 2C. Autoresearch grounding (Phase-0, deterministic)

`HENRI V2/agentic_graph/autoresearch_cli.py` (orx 0.2.10, no login) is the RESEARCH entry of the agentic graph: it turns a research question into provenance-pinned candidates + EvidenceReceipts before any model call. The graph has no `classified → dispatched` path that skips grounding.

```bash
cd "C:/Users/chan/Desktop/HENRI 7B SWARM/HENRI V2"
python -m agentic_graph.autoresearch_cli --query "<question>" --limit 8 --out ./.ar --receipts ./.ar/receipts.jsonl
```

A query/citation receipt proves WHAT WAS FETCHED, never what is true (`routing.assert_evidence_supports_scope(..., "capability")` refuses a capability promotion built on retrieval receipts alone). Receipts become Evidence records in the ontology (`henri-ontology` §8).

## 2D. Diagram-first output (standing directive, 2026-09-22)

When the material has structure, deliver a **diagram** instead of a prose wall: architecture, sequence, workflow, state, hierarchy, or a measured figure. Trigger table + renderer ladder: `references/diagram-mandate.md`. Order of preference: PenEcho Local MCP (`penecho_*`, 20 tools, registered 2026-09-22; requires the PenEcho app running with an enabled canvas) → inline chat widget (`::preview{file="..."}`) → local renderer skills (`architecture-diagram`, `excalidraw`, `p5js`, `manim-video`) → deterministic matplotlib figures → mermaid last. Diagrams carry the same evidence labels as text and NEVER override a numeric artifact.

## 3. Hermes MoA protocol

Live roster re-read **2026-09-22** directly from `%LOCALAPPDATA%\hermes\config.yaml` (`moa.presets.default` + mirrored top-level `moa.reference_models`). `hermes moa list` was unavailable for this re-read (venv console scripts blocked by Windows Application Control, WinError 4551).

```yaml
aggregator: deepseek/deepseek-flash            [max]   # vision UNVERIFIED (id is not a vision id)
references:
  - openrouter/z-ai/glm-5.3-flash              [high]  # Ref A: Code Logic
  - openrouter/meta/muse-spark-1.3-contributor [high]  # Ref B: System & Search
  - openrouter/xiaomi/mimo-v2.6-pro            [high]  # Ref C: Deep Context Judge (NOT MiniMax M3)
fallback: openrouter/google/gemini-3.8-flash
fanout: user_turn
reference_temperature: 0.3
degraded_reference_policy: loud
max_tokens: 4096          # top-level; no per-slot caps are set (the "800t" claim is stale)
save_traces: true
```

Reference models advise only. They do not execute tools. The aggregator owns synthesis and all actions. Ref C is the Deep Context Judge: JSON-only critique with exact file/line/symbol references, plus executable sympy/python assertion snippets the aggregator runs in its tool loop. `fanout: user_turn` runs references once and reuses their advice through the tool loop; do not use per-iteration fanout for coding work.

Use solo execution for routine reads, edits, scripts, and first-pass debugging. Use `/moa` only for a surgical mathematical derivation, a cross-file load-bearing audit, a bug that survives two genuine fixes, or explicit `[MOA]` user direction. Maximum three MoA calls per session. Keep prompts narrow and name the exact files, equation, or failure.

## 4. Hermes v0.19 leverage

Prefer deterministic or cached paths before LLM inference:

- **Cron no-agent scripts:** watchdogs, CI polling, ingestion, audit verification, and telemetry reduction. Empty stdout is silent; non-empty stdout is delivered verbatim.
- **Skill-backed cron:** attach only the specialist skills required by the job. Pin provider/model for unattended inference jobs.
- **`context_from`:** chain collection → analysis without replaying large logs in the prompt.
- **`workdir`:** pin cron jobs to the repository so project rules and paths are correct.
- **Execution history:** inspect `hermes cron runs` when a job reports failure; an unknown attempt is an audit record, not an automatic rerun.
- **Audit trail:** use the hash-linked HENRI ledger for decisions and paper-ingestion events; use MoA traces and session exports as separate evidence substrates.
- **Curator:** use deterministic usage tracking, backup, archive, and stale-skill review. Do not confuse curator with an always-running “skilld”; no Hermes `skilld` service is assumed.
- **Delegation:** parallelise independent research or audits. Children are not durable; use cron or a tracked background process for work that must survive a session.
- **Firecrawl:** use the configured web backend for current documentation and difficult pages. Do not suppress internet research by policy. Use Apify only when an installed MCP/plugin or credential-backed integration is actually present; never invent an Apify tool.

Preserve prompt caching: do not mutate the system prompt, toolset, or loaded skills mid-conversation. Restart or `/reset` after configuration/tool changes.

## 5. HENRI execution contracts

- Project root: `C:\Users\chan\Desktop\HENRI 7B SWARM`; active code is under `HENRI V2/`.
- Production verification: Vast CUDA target or `henri-ci`; local CPU tests are prohibited by project policy.
- Deployment default: commit → push → CI. Manual SSH/SCP is only for a CI-reported failure or an explicitly queued experiment. For a fresh Vast instance, first confirm the current hostname and SSH port from Vast; if a direct IP stalls after KEX, use the supplied `ssh*.vast.ai` hostname with `StrictHostKeyChecking=accept-new`, `IdentitiesOnly=yes`, and the local Ed25519 key.
- Fresh Vast bootstrap: `workspace_is_volume=true` is required for production persistence. A large ephemeral disk is not persistence. Vast containers are unprivileged; do not use Docker-in-Docker or Compose to initialize native PostgreSQL. Use `pg_ctlcluster`, the native package installer, `zone_c_bootstrap.py`, and `zone_c_seed_axioms.py`.
- Zone C: use `zone_c_env.py`; dev is the default, production requires explicit guarded configuration. Never hardcode production DSNs in new code. Move PostgreSQL `data_directory` to the mounted `/workspace` volume before benchmarks and verify `SHOW data_directory`.
- Remote repository layout: the Git checkout may be `/workspace/HENRI V2` while active Python code is under `/workspace/HENRI V2/HENRI V2`; use the exact path from `git ls-tree` before running tests.
- Python preflight: do not assume `/venv/main` contains the project dependencies. Check `python3 -c 'import torch; ...'`, `python3 -m pip list`, and `torch.cuda.is_available()` before test selection. Avoid blind PyTorch installation: CUDA wheels can pull hundreds of MB from PyPI and exceed SSH timeouts; use a detached remote install or the image's configured environment.
- Archive deprecated code under `HENRI V2/_archive/`; do not delete historical mechanisms without a recorded reason.
- Never expose credentials. Secrets belong in `.env`; behavioural settings belong in `config.yaml`.

## 6. Audit commands

```bash
python "$HOME/AppData/Local/hermes/scripts/henri_audit.py" verify
python "$HOME/AppData/Local/hermes/scripts/henri_audit.py" tail
python "$HOME/AppData/Local/hermes/scripts/nlm_run.py" login --check
python "$HOME/AppData/Local/hermes/skills/henri-workflow/henri-agent-integration/scripts/validate_holonic_contracts.py"
hermes cron list
hermes cron runs <job-id> --limit 20
hermes moa list
hermes profile list
hermes tools list
hermes mcp list
```

The active HENRI jobs are CI, research ingestion, and context watchdog. Keep them silent when healthy and actionable when they fail. Validate script paths on Windows: cron scripts must use Hermes-relative script names or valid POSIX paths, not MSYS-mangled `C:\...` strings.

## 7. Specialist handoff

Load `henri-architecture` before changing wave mechanics, EDMD, constraints, EFE, learning objectives, or Zone C. Load `henri-moa-routing` before escalating. Load `henri-telemetry-analyzer` before interpreting a run. End each substantial change with: claim, evidence, uncertainty, next falsification.
