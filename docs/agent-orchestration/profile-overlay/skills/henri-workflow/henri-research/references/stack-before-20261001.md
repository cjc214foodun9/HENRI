# Historical instruction snapshot

This is the previous entry file. It is not current policy. Use current SKILL.md for routing. Revalidate every dated runtime claim.

---
name: henri-research
description: HENRI research pipeline — vault, NotebookLM, Firecrawl, arXiv, and delegated multi-source retrieval with compact audit artifacts.
category: henri-workflow
---

# HENRI RESEARCH PIPELINE

Use this skill when an external claim, paper, HENRI feature, or design dependency needs evidence. Research output must be compact, cited, and ingestible into the HENRI vault.

## HOLONIC STATIC PREAMBLE — shared contract header

Byte-identical across `henri-research`, `henri-architecture`, `henri-agent-integration`. Edit all three copies together or not at all; drift fails `validate_holonic_contracts.py`.

- Triad: `/henri-agent-integration` (Root Holon & Infrastructure Conductor) ←→ `/henri-research` (Exploration & Spec Holon) ←→ `/henri-architecture` (Implementation Engine Holon).
- Frozen contracts: A `SpecContract` (research → architecture); B `HarnessContract` (architecture → integration); C `ExecutionFeedbackContract` (integration → research/architecture). Schemas: `references/holonic-contracts.md`.
- MoA engine (LIVE `hermes moa list` 2026-09-11): references 1 `openrouter/z-ai/glm-5.3-flash` [high], 2 `openrouter/meta/muse-spark-1.3` [high], 3 `openrouter/minimax/minimax-m3` [high] = Deep Context Judge; aggregator `deepseek/deepseek-flash` [max]. A roster line is a dated snapshot: qwen/qwen3.8-flash (slot 2) and the `:free` id are superseded. Re-read `hermes moa list` before quoting a roster.
- State passing: intent, local constraints, and structured payloads only. Never pass full traces, raw tracebacks, or chain-of-thought upward.
- Evidence labels: `OBSERVED`, `DERIVED`, `INFERRED`, `HYPOTHESIS`, `FALSIFIED`, `BLOCKED`.
- KV-cache rule (STRICT, enforced 2026-08-27): this header, the system prompt, the loaded skill set, and the toolset are the session cache prefix. MUST stay byte-identical; append ephemeral execution traces at the END of the payload, never in the middle. Any mid-session change to system prompt, skills, tools, MoA roster, or model invalidates the prefix cache — apply such changes only between sessions, then restart.

## Holon position: Exploration & Spec Holon

You are `/henri-research`, the theoretical and algorithmic exploration node of the HENRI holonic triad. You are a complete local workflow (question → evidence → spec) and part of the parent graph rooted at `/henri-agent-integration`. Your children are bounded leaf workers (delegated retrieval, arXiv/NotebookLM/vault queries, hashing). The parent receives only your compact `SpecContract` and evidence summary — never your exploration trace.

INPUTS

- User query or objective (root intent, not the full session trace).
- `ExecutionFeedbackContract` with `status = UNIT_FAIL | GRADIENT_FAIL | PERF_FAIL | OOM_CRASH`: re-evaluate the foundational equations and generate updated invariants. Do NOT patch code; that is the architecture holon's role.
- Local constraints: bounded budget, `spec_id` of the affected carrier.

OUTPUTS

- `SpecContract` (Contract A) — the only upward payload to `/henri-architecture`. Schema + example: `references/holonic-contracts.md`.
- Vault note and governance event as evidence layers, never inter-holon state.

BOUNDARY RULES

- Never output speculative code implementations. Output mathematical equations, loss topologies, tensor shapes, boundary conditions, and falsifiable invariants.
- Every tensor operation must state floating-point precision constraints (e.g. `bf16` scaling, accumulation dtype, numerical clamp limits).
- Run an internal reflection loop (evidence → hypothesis → falsification check) before emitting the `SpecContract`. Emit only the finished spec.
- Schema freeze: Contract A is immutable. Propose schema changes as a bounded design with approval; never mutate the payload format to optimize locally.

## Source order

1. **Local vault** — fastest HENRI-specific source for prior theory, telemetry, and decisions.
2. **NotebookLM corpus (MCP)** — the HENRI research corpus is an always-on consult subagent, not a fallback. Query it BEFORE web search for HENRI theory, mechanism philosophy, and telemetry interpretation (see the corpus-consult section below). Primary bank `ca4bb787-de9d-4ee0-89c9-bf71259cc86d`.
3. **Official web sources** — use Hermes documentation, GitHub, Firecrawl-backed extraction, and vendor documentation for current APIs and versions. Internet access is required when freshness matters; do not suppress it by policy.
4. **arXiv / Semantic Scholar** — use for primary literature and ungrounded parameter decisions.
5. **Apify** — conditional only. Use an installed, credential-backed Apify MCP/plugin or API. Verify the integration before citing it; do not invent an Apify tool or actor.
6. **OpenResearch `orx` retrieval (Phase-0 grounding)** — the local autoresearch node. Deterministic, no-login literature grounding that returns provenance-pinned candidates plus `EvidenceReceipt` records. Use it BEFORE web search for an ML/CS/math/physics literature question: it is faster, hash-pinned, and its receipts feed graph promotion. See `references/autoresearch-grounding.md`.

```bash
cd "C:/Users/chan/Desktop/HENRI 7B SWARM/HENRI V2"
python -m agentic_graph.autoresearch_cli --query "<research question>" --limit 8 \
    --out ./.ar --receipts ./.ar/receipts.jsonl          # add --json for the full object
python -m agentic_graph.autoresearch_cli --paper <arxiv-id>   # one paper, report or full text
python -m agentic_graph.autoresearch_cli --probe             # binary + version + sha256
```

Grounding rule: a `query`/`citation` receipt proves **what was fetched**, never what is
true. `routing.assert_evidence_supports_scope(..., "capability")` refuses a capability
promotion on retrieval-only evidence. An empty candidate set is an empty set, not proof
that no literature exists. `orx up`/`exp run`/`agent spawn` need a login and a harness
(`codex`/`opencode` are ABSENT on this host) — those tiers stay `BLOCKED`; use Vast.ai
for GPU execution.

## NotebookLM corpus consult — always-on research subagent

The NotebookLM MCP is the project's curated theory/philosophy bank. It is authenticated (`auth_status=configured`, server `gemini-notebook-mcp` v4.0.3 / CLI 0.10.1, verified 2026-09-04) and its banks hold the corpus of HENRI's proprietary math, philosophy, and mechanism rationale. Integrate it into every workflow phase:

- RESEARCH: before external search, ask the corpus for prior treatment of the mechanism, its philosophy, and its stated assumptions.
- DESIGN: ask how the corpus frames the mechanism (physical interpretation, known limits, labeled falsifications).
- MEASURE: after any test execution, ask the corpus to interpret the telemetry (Sagnac delta, Kuramoto r, EFE decomposition, veto counts, learning engagement) against the theory bank.
- INFER: ask whether a proposed attribution contradicts any corpus statement; record conflicts.

Corpus banks (source counts verified 2026-09-04):

| Notebook | ID | Sources |
|---|---|---|
| HENRI philosophy (primary) | `ca4bb787-de9d-4ee0-89c9-bf71259cc86d` | 240 |

MCP tools in-session:

- `notebook_query(notebook_id, query)` — grounded Q&A with citations; one focused question per call.
- `cross_notebook_query(query, notebook_names=...)` — aggregate answer across banks for questions spanning philosophy + architecture + math.
- `batch(action=query)` — same question across multiple notebooks.
- `source_add(source_type, ...)` — push a verified paper/PDF/URL into a bank after ingestion; keeps the corpus current.
- `chat_get` / `chat_export` — retrieve a consult thread for the vault record.
- `server_info` — auth probe. If `auth_status=stale`, re-run `nlm login` (pip CLI in the Hermes venv, own cookie store) before queries.
- Reauth is NOT complete when cookies validate — verify the FULL chain (OBSERVED 2026-09-04): (1) `nlm login` + `nlm login --check`; (2) the configured MCP server process starts — a `request_ctx` ImportError means fastmcp 3.x is running against the Hermes `mcp==2.0.0` pin (mcp 2.x removed `request_ctx`); the server now runs from the isolated venv `C:\Users\chan\AppData\Local\hermes\mcp-envs\notebooklm` (notebooklm-mcp-cli 0.10.1 + fastmcp 4.0.3 + mcp 2.0.0), wired with `hermes config set mcp_servers.notebooklm.command` (direct config.yaml writes are refused); (3) a stdio handshake passes — the python mcp SDK stdio dialect is NEWLINE-delimited JSON, never LSP Content-Length framing; (4) `server_info` reports `auth_status=configured`. MCP tools attach at session start: after auth/server/config changes start a NEW session, never hot-swap mid-session. Full topology, version matrix, and failure modes: `references/notebooklm-mcp-stack-verification.md`; re-runnable probe: `scripts/notebooklm_mcp_probe.py`.

Query discipline:

- One focused question per query; ask for the source-grounded answer and its stated limits.
- Ask "which sources in the notebook support this?" to get citations.
- Label corpus answers `INFERRED` (grounded synthesis from bank sources) or `HYPOTHESIS` (bank speculation). Never label corpus output `OBSERVED` for telemetry the corpus did not generate.
- Live code and measured CUDA telemetry OVERRIDE corpus claims when they conflict; record the conflict in the vault note.

## Retrieval protocol

- One focused question per search.
- Prefer `web_extract` for known URLs and `web_search` for discovery.
- For difficult or dynamic pages, use the configured Firecrawl web backend or browser only when extraction is insufficient.
- Primary-source fallback when managed web extraction is unavailable: query
  arXiv directly with curl. Search:
  `curl -sL 'https://export.arxiv.org/api/query?search_query=all:%22term%22+AND+all:%22term2%22&max_results=6'`
  then grep `<id>`/`<title>`. Look up by ID: `?id_list=2508.12305,1503.06237`.
  Summaries are XML-escaped (`&lt;` etc.); decode or `sed` them. Hash the
  returned bytes and record the boundary as `OBSERVED_PRIMARY_BYTES`. This
  This works with zero Firecrawl credits (verified 2026-08-02 for the OAM
  skyrmion paper arXiv:2508.12305 and HaPPY arXiv:1503.06237). On this Windows
  host, native curl needs a NATIVE output path (`-o C:/Users/chan/arxiv.xml`);
  MSYS `~/...` and `/c/...` forms silently fail to write the file (OBSERVED
  2026-08-19: rc=0 but no file; native-path retry returned HTTP:200 + bytes).
- For multiple independent questions, use `delegate_task` in parallel and return URLs, paper IDs, and short evidence blocks. Do not use MoA for mechanical retrieval.
- Respect arXiv rate limits. For bulk work, delegate query groups, use HTML/search fallbacks, deduplicate base IDs, and merge once.

## Grounding pitfalls (official-source bytes)

- When probing MULTIPLE candidate URLs, write each response to its OWN file;
  a later probe overwrites the earlier one (a 404 page silently replaced the
  pinned 200 homepage bytes on 2026-08-24; evidence had to be re-fetched).
- Pin exact string + SHA-256 + HTTP code + retrieval date. Version labels may
  be family prefixes — extract the precise version from pinned bytes
  (e.g. "v4.1" → official "Intelligence Index v4.1.1").

## Video/DOI primary-source fallbacks

When web search is blocked (no Firecrawl key) and the source is a YouTube talk or a paper DOI, use `references/primary-source-fallbacks-youtube-crossref.md`: innertube search → oembed title resolution → transcript fetch (`uv run --with youtube-transcript-api`), CrossRef DOI lookup, and the spec-table-vs-live-telemetry audit rule.

## Related skills

- `henri-soul` — session operating system.
- `henri-co-scientist-rigor` — execution-grounded claim clipping vs E_log, scaffold-transition audit, conditional TrueSkill/UCB ranking (source: Google co-scientist, alphaXiv 2608.26701).
- `henri-architecture` — load-bearing HENRI constraints (F1 Lie displacement carrier contracts, adjoint su(3) sign convention, FALSIFIED Hadamard coupling, live audit-ledger lookup: `references/f1-lie-displacement-carrier-validation.md`).
- `henri-moa-routing` — bounded MoA escalation.
- `arxiv` — focused arXiv access.
- `notebooklm` — curated math bank access.
- `literature-search-workarounds` — resilient multi-source fallback.

## Asymmetric Leaf-Worker Research Routing (CONDITIONAL)

The canonical default MoA profile (verified live via `hermes moa list` + wire probe 2026-08-29) is:

```text
aggregator:  deepseek / deepseek-v4-flash        [reasoning=max]
references:  1. openrouter / z-ai/glm-5.3-flash   [reasoning=high, 800t]
             2. openrouter / qwen/qwen3.8-flash  [reasoning=high, 800t]
             3. openrouter / minimax/minimax-m3:free  [reasoning=high, NO cap] = Deep Context Judge
fallback:    openai-api / gpt-5.6-luna            [reasoning=high]
preset name: default (model.provider=moa, model.default=default)
```

Balanced cost posture: aggregator at `max` is the standing default; cost control comes from compact retrieval artifacts (bounded excerpts, hashes, source IDs), not effort downgrade. The fallback runs at `high` and resolves only on primary failure.

Use this topology when a verified DeepSeek provider and model ID are available:

```text
aggregator (DeepSeek V4 Flash) macro-planner/auditor
  -> bounded DeepSeek leaf workers
  -> deterministic receipts
  -> aggregator synthesis and CoE promotion
```

DeepSeek is a stateless leaf worker, not a state owner. Use it only for bounded paper extraction, claim-to-source mapping, and compact candidate summaries. Do not use it for final CoE promotion, human approval, repository mutation, remote execution, or external-outcome reporting.

Rules:

- Hash and inventory documents with local deterministic code before any model call.
- Give each leaf one atomic task and one bounded source slice. No multi-turn leaf conversation.
- Request schema-constrained output with at most 500 tokens. Validate and truncate deterministically; a prompt limit is not runtime enforcement.
- Dispatch at most 3 independent leaves. Retry only the failed leaf once with a changed repair fingerprint.
- Use parallel worker calls only when the installed runtime supports them. Do not claim Batch API savings unless a live provider integration and returned batch ID are verified.
- Pass only source ID, SHA-256, section range, compact excerpt, and task spec. Never pass a full paper or raw retrieval trace.
- A leaf claim remains `unverified` until its source hash, citation, and deterministic source check pass.

The proposed values for DeepSeek speed, price, prompt-cache discount, and long-horizon behavior are `HYPOTHESIS` until provider documentation and live telemetry support them. The exact model identifiers `deepseek:deepseek-flash` (aggregator), `openrouter:z-ai/glm-5.3-flash` (reference 1), `openrouter:meta/muse-spark-1.3` (reference 2), and `openrouter:minimax/minimax-m3` (reference 3, Deep Context Judge) are `VERIFIED` (live `hermes moa list`, 2026-09-11). Aggregator cache and spend remain `UNVERIFIED` — the trace aggregator slot carries no `usage`/`cost_usd` field.

## Isolated-Cache Dual-Max Research Enclaves (CONDITIONAL)

Use this procedure only after a provider preflight verifies the exact model IDs, credentials, response schema, and reasoning-effort controls. It is not the current default MOA route.

```text
DeepSeek enclave: static prefix + one atomic task
  -> local response firewall
  -> compact evidence receipt only
OpenAI enclave: static CoE prefix + rehydrated state delta
  -> aggregator synthesis and promotion
```

Rules:

- Never forward raw DeepSeek output, provider metadata, hidden reasoning, or full traces to the aggregator.
- If an adapter returns fields such as `reasoning_content` or `final_content`, treat the reasoning field as non-forwardable and parse only the allowlisted structured result. Do not assume either field exists.
- If the response is not valid JSON matching the versioned receipt schema, emit `SCHEMA_INVALID` and stop promotion. Do not ask the aggregator to repair raw prose.
- The firewall may hash and archive the raw response locally under the approved evidence policy, but it must forward only source IDs, hashes, bounded claims, status, and limitations.
- Each provider has a separate static prefix. A local prefix hash proves byte stability only; it does not prove a cache hit, a cache discount, or cache sharing across providers.
- A leaf uses one request and is discarded after receipt creation. Obsidian and the local event store receive state deltas, not chat histories.
- `Max` is a per-route setting. Do not infer that the global `reasoning_effort` setting applies to MOA references. Record the actual provider, model, and effort returned by the live route. The standing aggregator runs at `max`; the fallback (`openai-api / gpt-5.6-luna`) runs at `high`.

The claims that Dual-Max gives a 100% cache miss at the boundary, a 95% cache rate within each provider, or zero intelligence degradation are `HYPOTHESIS` until provider telemetry and a controlled comparison support them.

## Autoresearch grounding (Phase-0 node — deterministic, no model call)

`orx 0.2.10` (local binary, NO LOGIN) supplies literature grounding through
`agentic_graph/autoresearch_cli.py` under HENRI V2. It turns a research question
into provenance-pinned candidates plus `EvidenceReceipt` records BEFORE any model
call; the agentic graph has no `classified → dispatched` path that skips it.

```bash
cd "C:/Users/chan/Desktop/HENRI 7B SWARM/HENRI V2"
python -m agentic_graph.autoresearch_cli --query "<question>" --limit 8 --out ./.ar --receipts ./.ar/receipts.jsonl
python -m agentic_graph.autoresearch_cli --paper <arxiv-id>    # one paper
python -m agentic_graph.autoresearch_cli --probe               # version + exe sha256
```

Bounds are hard: ≤3 strategies, ≤3 primitive calls, ≤90 s wall, `--limit` ≤25.
`orx up` / `exp run` / `agent spawn` need login + a harness → BLOCKED; GPU
execution stays on Vast.ai.

Evidence rules (do not weaken): a receipt proves WHAT WAS FETCHED, never what is
true; an empty candidate set is an empty set, not proof of absence;
retrieval-only evidence cannot promote a capability claim. Full verified
substrate, falsified claims, and reproduction:
`references/autoresearch-grounding.md`. Receipts become Evidence records in the
ontology layer: `henri-ontology`.

## Compacted references (2026-09-22)

- `references/evidence-discipline.md` — Evidence discipline. Audit patterns, theory boundaries, and recorded falsifications. Sub-sections: Sub-sections: Formal-theorem-to-architecture boundary; Reverse-engineered architecture review (roadmap premise…; Supplied blueprint with its own harness…; Supplied theory-packet audit (paper + audit doc + blueprint…; Frozen-backbone structural egress; Heldout sealing and structural-egress promotion corrections; Bounded intervention smoke protocol (flag-gated live-path…; Grammar-expansion safety; Mechanism efficacy and order-invariance audits; Candidate-specific ranking saturation; +9 more.
- `references/stage0-dynamical-substrate.md` — Stage-0 dynamical-substrate integration. Dynamical-substrate integration protocol and its boundaries.
- `references/vault-ingestion.md` — Vault ingestion. Drive-ingest canonical paths, PDF recovery, companion resolution. Sub-sections: Sub-sections: Drive-ingest canonical-path and receipt verification; Extensionless PDF recovery; Companion-file resolution (learned 2026-08-12).
- `references/agentic-graph-ingestion.md` — Agentic graph and time-series ingestion. Graph/time-series ingestion rules and store boundaries.
- `references/token-controls.md` — Token controls. Token-control rules and qFHRR/gauge audit notes. Sub-sections: Sub-sections: Delta-rule qFHRR associative memory (premise audit); Gauge-invariant and relational egress audits.
