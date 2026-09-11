# Research Queue — Agentic-Engineering Literature (registered 2026-09-11)

**Status:** `QUEUED` (not executed) · **Route:** vault → arXiv → NotebookLM MCP → SpecContract
**Owner holon:** `/henri-research` (Exploration & Spec). Prereg discipline per `henri-research`.

## Correction of record

STRACE (the trace-attribution pipeline vendored at
`HENRI V2/agentic_graph/STRACE-main`, ported as the `henri-strace-optimizer`
skill) is **Microsoft Research**, arXiv:2607.07702 — not Google. The skill header
already states this. The request's phrasing "strace from google" is corrected
here so the queue targets the right provenance.

Two further corrections, verified this session:

- The `strace` skill's own frontmatter and `henri-strace-optimizer` agree the
  source repo has **no LICENSE file**; the skill paraphrases and cites rather
  than copying. Preserve that posture when quoting the paper.
- The `holon-graph-shacl-main` tree is **CC BY 4.0** (Kurt Cagle, HGA v0.9.0,
  `https://ontologist.io/ns/holon#`). Attribution is required if any of its
  ontology text ships in-tree and is redistributed.

## Queue

Search queries, not IDs. Record an arXiv ID only after retrieval verifies it —
never invent an identifier.

| # | Topic | Query | Feeds |
|---|---|---|---|
| 1 | Prompt-cache / prefix-reuse economics | `KV cache reuse prompt caching inference cost LLM agents` | `henri-moa-routing` cache spine; the 25.5% measured hit rate |
| 2 | Multi-agent aggregation & routing | `mixture of agents LLM aggregation routing debate` | MoA persona topology, Judge-C gating |
| 3 | Trace failure attribution | `LLM agent trajectory failure attribution root cause` | cross-check against STRACE's 4-stage pipeline |
| 4 | Long-horizon context management | `long-horizon LLM agent context management memory state` | `context_packer` 4-layer design; 137k-token p50 prompt |
| 5 | Vision-in-the-loop verification | `multimodal LLM chart figure understanding evaluation` | digest-figure → `vision_analyze` loop |
| 6 | Agentic harness engineering | `agentic engineering harness evaluation agent benchmark` | harness design; AAII v4.1 alignment |

## Method (per `henri-research`)

1. **Vault first.** Query the local vault via `scripts/local_vault_search_server.py`
   before any external call.
2. **NotebookLM consult.** Bank `ca4bb787-de9d-4ee0-89c9-bf71259cc86d` is the
   always-on consult subagent. Auth proof is the log line
   `Cookies: N extracted` plus `--check`, never exit code 0 alone.
3. **arXiv retrieval** with SHA-256 + section range + bounded excerpt per source.
4. A leaf claim stays `unverified` until source hash, citation, and the
   deterministic source check pass. Label every claim
   `OBSERVED`/`DERIVED`/`INFERRED`/`HYPOTHESIS`/`FALSIFIED`/`BLOCKED`.
5. Never pass a full paper or raw retrieval trace upward; pass ID, SHA, section
   range, and a compact excerpt.
6. Findings route `HENRI_Inbox` (Drive) → local → `HENRI_Research_Vault`
   projection. GitHub capture only via the bounded
   `scripts/telemetry/henri_capture_to_repo.py` path. Never mirror everything.

## The reporting stack this queue must harden

Already measured working end to end (`OBSERVED` 2026-09-11):

```text
deterministic reducer  ->  figure (viz-venv matplotlib, fixed DPI/palette)
                       ->  vision_analyze coarse read
                       ->  numeric confirmation against the JSON   [authoritative]
```

The vision read of `scorecard_gates.png` matched the JSON exactly (GPQA margin
0.048 FAIL, MMLU 0.0098 FAIL) — but a prior probe hallucinated curve counts, so
the numeric check is mandatory and the vision read is only a filter.

Queued figures, in priority order:

1. MMLU per-subject bars (57 subjects; `--subjects`) — currently only in text.
2. HumanEval 8-variant comparison (the condition sweep is the real finding).
3. Cache-hit trend over time from the audit JSON (is 25.5% improving?).

Query #5 exists to source better practice for this loop, not to justify the
vision read as evidence. Vision never overrides a number.

## Explicitly out of scope

- No roster, model, or config change mid-session (cache rule). Cold restart first.
- No new MCP install without a preflight probe.
- No claim of Google authorship for a paper until the retrieved record confirms it.

## Next action

Run the vault pass (step 1) for queries #1 and #3 only, return two compact
evidence summaries, and record the outreach in the audit ledger via
`henri_audit.py record` (subcommands are `record|verify` only).
