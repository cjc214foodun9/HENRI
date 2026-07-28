# HENRI Agentic Graph — TimescaleDB & Production Specification (v2, audited)

**Status: audited against Hermes Agent v0.19.0 (Quicksilver) release notes, 2026-07-23.**
This document replaces the previous draft, which cited a fabricated config schema
(`hermes_fleet_profiles.yaml`), a LangGraph `PostgresSaver` that Hermes does not use,
and a mock executable engine with hardcoded approvals and fabricated telemetry.
All features below are verified real or explicitly marked as custom builds.

---

## I. The Agentic Graph Already Exists — Node Mapping

The requested graph (ingest → plan → MoA verify → human gate → GPU execute → report)
is LIVE as native Hermes primitives. No LangGraph wrapper is required or desirable —
a wrapper would be an arbitrary abstraction over machinery that already handles
durability, delivery, and routing.

```
NODE                          IMPLEMENTATION (all live, verified 2026-07-23)
─────────────────────────────────────────────────────────────────────────────
1. Research Ingress           cron b0249aa158b1 "henri-research-ingest" (10m)
                              G:\My Drive\HENRI_Inbox → pypdf → Obsidian vault
                              → all-MiniLM-L6-v2 embeddings (zero-token, local)
2. Aggregator Planner         Agent session, gemini-3.6-flash + henri-soul skill
                              RESEARCH → SYNTHESIZE → plan, WAITS for approval
3. MoA Verification           /moa preset: kimi-k3 + sakana/fugu-ultra advisors
                              (reasoning_effort: low, 500-token cap, user_turn
                              fanout) → gemini-3.6-flash aggregator (medium)
4. Human-in-the-Loop Gate     Telegram gateway (PENDING: bot token + chat id)
                              + v0.19 smart approvals (LLM reviewer, default ON)
                              + v0.19 user-defined deny rules (see §IV)
5. GPU Execution              cron 8027351ab01e "henri-ci" (3m): push → pull on
                              5090 → CUDA suite → experiment launch → monitor
6. Telemetry Report           henri-ci completion delivery (cd82 summary,
                              JSONL paths) → §V TimescaleDB loader (NEW, below)
─────────────────────────────────────────────────────────────────────────────
EDGES: cron delivery, git push, Telegram messages
STATE: state.db (sessions, session_model_usage, delivery_obligations)
```

## II. Quicksilver v0.19.0 — Real Feature Map

| Phantom spec claim | Real v0.19 feature | Action for HENRI |
|---|---|---|
| "<0.9s cold start" | ~80% TTFT cut, 4.3s→0.9s, all surfaces incl. cron | Automatic on update |
| "Real-Time Thinking Stream" | Reasoning streams live by default | Automatic; `display.show_reasoning` default ON |
| "Smart Dual-AI Approval Gate" | Smart approvals default: independent LLM reviewer per flagged command, verdict-scoped (no pattern reuse) | Already default ON in v0.19 |
| "Non-bypassable deny rules" | User-defined deny rules — block commands even under yolo; `/deny <reason>` feeds refusal reason back to the agent | Configure HENRI invariant rules (§IV) after update |
| "Crash-Resilient DB Execution" | Delivery-obligation ledger in state.db: finished responses redelivered after gateway crash; durable background delegation with ownership-checked result ledger | Automatic on update |
| "Single-Gateway Multi-Profile Fleet" | Profile-based message routing: one gateway/bot routes guilds/channels/threads to isolated profiles (config, skills, memory, secrets) | Optional: split `henri-research` / `henri-executor` profiles (§VI) |
| (not in spec) | Live subagent transcripts (`tail -f` per delegate_task child) | Use for RESEARCH-stage parallel sprints |
| (not in spec) | `hermes sessions export` — Markdown/HTML/HF-trace, filters, `--redact` secret scrub | Basis of the audit chain (§III) |
| (not in spec) | Per-slot reasoning effort in MoA presets | **Already deployed** (advisors low / aggregator medium) |
| (not in spec) | Bitwarden/1Password secret sources | Optional hardening for `.env` |

## III. Immutable Audit Chain — the Honest Version

The v0.19 substrate already records: every message (state.db `messages`), per-call
token/cost accounting (`session_model_usage`), durable deliveries
(`delivery_obligations`), subagent results (delegation ledger), and full API
request dumps (`sessions/request_dump_*.json`). MoA turns can persist full traces
(`moa.save_traces`).

**Audit export (zero new code):**
```bash
hermes sessions export ./audit/ --redact   # HF-trace or Markdown, secret-scrubbed
```

**Optional cryptographic seal (small custom script, if desired):**
append-only SHA-256 hash-chained JSONL written by the two real cron scripts
(henri_ingest.py, henri_ci.sh) at each governance event
(PAPER_INGESTED → PLAN_DELIVERED → HUMAN_DECISION → PATCH_PUSHED → CI_VERDICT).
This is the only part of the previous draft's "HenriAuditLedger" worth keeping —
and it must be fed by real events from real scripts, never by simulated nodes.
(The draft engine's hardcoded `human_vote = True` and fabricated
`env_scores: {ar25: 1.0}` are exactly the mock-loop failure mode this project bans.)

## IV. HENRI Invariant Deny Rules (enable post-update)

v0.19 user-defined deny rules block matching commands even under yolo. Encode the
load-bearing physical invariants from henri-architecture:

```
RULE_PRESERVE_UNIT_NORM     block: rm/sed/patch removing F.normalize call sites
RULE_MIN_REJECT_THRESHOLD   block: edits setting constraint_reject_thresh < 0.30
                            (below intrinsic phase-linewidth of the manifold)
RULE_NO_ADDITIVE_BOUNDARY   block: re-adding additive boundary row to the
                            boundary channel (falsified, run 12)
RULE_NO_NS_RETRACTION       block: Newton-Schulz Stiefel retraction (diverges
                            past sqrt(3) singular values; Cholesky only)
```

Exact config keys to be read from `hermes config` post-update; this section is a
design target, not a verified schema. `/deny <reason>` in Telegram supplies the
human-feedback leg.

## V. TimescaleDB Integration — Real, and Already Half-Built

Zone C (TimescaleDB + pgvector) is HENRI's production long-term memory on the 5090.
The agentic-graph integration is one loader, not a new database:

1. **CI already pulls** `telemetry_logs/production_run_*.jsonl` on completion.
2. **Add one step to henri_ci.sh**: after pull, INSERT per-step records
   (run_id, env_id, step, efe, rms_residual, sagnac_delta, admissible_count,
   fallback_executed, pearl_repaired, score) into a `zone_c_agent_telemetry`
   hypertable on the 5090's existing TimescaleDB.
3. **Continuous aggregate** for hourly per-env rollups (avg EFE, fallback rate,
   PEARL repair rate) — TimescaleDB native, zero agent tokens.
4. INFER stage (henri-soul §7) then queries SQL instead of re-parsing JSONL —
   cheaper, faster, cross-run.

Schema sketch (adapt to the live Zone C schema before applying — audit first,
per project rule): hypertable on `time`, index on `(run_id, env_id)`,
`vector(4096)` column reserved for quantized engram waves only if the 65536-dim
projection contract is settled (PCA/quantization is a design decision, not a given).

## VI. Docker — Minimal Honest Topology

REJECTED from previous draft (duplicates of live native machinery):
- `chromadb` container — vault search is an embedded PersistentClient + FastAPI
  process on the Windows host (port 8000, running, 1,138 chunks indexed)
- `gdrive_sync` container — Google Drive for Desktop mounts G:\; the ingest cron
  (10m) is the watcher
- `langgraph_agent` container — Hermes IS the agent; gateway runs as a Windows
  service with auto-restart

ACCEPTED (optional, dev convenience only):
- A **single-service** TimescaleDB container on the Windows host for local
  schema prototyping against Zone C DDL before touching the 5090 production DB.
  **Deployed:** `docker/zonec-dev/docker-compose.yml` — container `henri-zonec-dev`,
  port **5433**, db `henri_zonec_dev`, user `zonec_dev_user`, with
  `init_scripts/01_env_marker.sql` seeding the `_zonec_environment = 'dev'`
  marker row.

**Dev/prod separation is enforced structurally, not by memory** — see
`HENRI V2/zone_c_env.py` (resolver + connection assertion, 9/9 CPU-verified)
and the contract table in the henri-architecture skill:

| | PRODUCTION Zone C | DEV Zone C |
|---|---|---|
| Reachable at | SSH tunnel `localhost:10100` | `localhost:5433` |
| DB / user | `henri` / `postgres` | `henri_zonec_dev` / `zonec_dev_user` |
| Marker table | none | `_zonec_environment = 'dev'` |
| DSN source | `ZONE_C_PROD_DSN` + explicit `ZONE_C_ENV=prod` | default; localhost + `_dev` suffix enforced |

Production Zone C stays on the 5090 beside the GPU runs — co-locating the DB
with the telemetry producers avoids shipping per-step rows across SSH.

## V-A. TrustGraph holonic context boundary

TrustGraph is an optional context-graph and bounded specialist-execution
service. It is not the HENRI governance ledger and it is not a replacement for
Zone C. The local event store remains the source of truth for governed events.

The controlled path is:

```text
Drive source
  → source hash
  → Hermes audit event
  → Obsidian/local event projection
  → TrustGraph library document and processing IDs
  → GraphRAG provenance IDs
  → compact holon result
  → agentic time-series event
```

The installed Hermes environment currently exposes `trustgraph.api`,
`trustgraph-base 2.7.14`, `trustgraph-cli 2.7.14`, and the `tg-*` command
entry points. This proves installed client components only. The default local
TrustGraph ports refused connections during the current audit, so service
integration remains `BLOCKED` until `tg-verify-system-status --skip-ui` and a
two-document controlled probe pass.

The TrustGraph supervisor pattern may decompose a bounded research question,
fan out child holons, and synthesise compact findings. Child traces stay inside
the child boundary. A child cannot approve a patch, mutate the repository,
launch CUDA execution, or emit an external outcome.

Do not copy TrustGraph graph payloads, Obsidian notes, or Hermes session state
into Zone C. Do not copy wave checkpoints or full latent engrams into the local
event store or agentic time-series database. Use stable IDs, hashes, and
artifact paths for cross-store links.

## VII. Profile Fleet (optional, post-update)

If task segregation is wanted after the update: `hermes profile create henri-research`
(isolated skills: research/arxiv/notebooklm only) and keep `default` as executor;
route Telegram topics to profiles via v0.19 profile routing. Defer until the
Telegram leg is live — a fleet with one connected channel is speculative
infrastructure.

## VIII. Deployment Sequence

1. Close Hermes desktop/terminals → `hermes update` (v0.18.2 → v0.19.0; blocked
   while sessions hold venv locks)
2. Verify: `hermes --version` = v0.19.x; gateway auto-restarts
3. Telegram: BotFather token + TELEGRAM_HOME_CHANNEL in `.env` →
   `hermes gateway restart` → flip crons b0249aa158b1 / 8027351ab01e / 6ef3bb1c1659
   to `deliver='telegram'`
4. Configure §IV deny rules (keys from `hermes config` post-update)
5. Add §V telemetry loader step to henri_ci.sh (audit Zone C live schema first)
6. Optional: §VI dev container, §VII profile fleet
