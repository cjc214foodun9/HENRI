# HENRI TrustGraph Agentic Workflow Manual

**Audience:** HENRI development operators using Hermes, Photon, Google Drive, TrustGraph, the local governance graph, and remote CUDA verification.

**Repository:** `C:\Users\chan\Desktop\HENRI 7B SWARM`

**Primary interaction surfaces:**

- **Photon** — mobile control surface for requests, status, review, and human decisions.
- **Google Drive for Desktop** — research and controlled-source ingress.
- **Hermes** — deterministic collectors, agent reasoning, approvals, cron, and audit records.
- **TrustGraph** — optional holonic context and specialist-execution layer.
- **Local HENRI event store** — governance source of truth for typed workflow events.
- **HENRI CI or Vast CUDA** — required execution and runtime verification path.

> **Important status boundary:** TrustGraph service health is currently observed from a real `tg-verify-system-status` run. Google Drive for Desktop ingress is configured and the mounted inbox exists. Google Workspace API authentication is not currently configured. Photon is enabled as a Hermes platform plugin, but a fresh user-visible Photon round trip is not verified by this manual. Treat those claims as separate.

---

## 1. What this workflow does

The workflow converts a research source or engineering request into a governed HENRI development cycle:

```text
Photon request or Google Drive source
  → deterministic discovery and hashing
  → local event and Hermes audit seal
  → Obsidian note and optional semantic index
  → TrustGraph context retrieval or bounded holon execution
  → compact research and audit finding
  → human approval through Photon
  → bounded repository change
  → commit and push
  → HENRI CI or Vast CUDA verification
  → compact telemetry reduction
  → separate external-outcome decision
```

The workflow is not an unrestricted autonomous coding loop. TrustGraph may retrieve context and coordinate bounded specialist work. It must not:

- approve its own patch;
- mutate the repository without the required human decision;
- launch an expensive HENRI run without an approved scope;
- write a simulated benchmark result;
- turn internal coherence into task success;
- place research notes, graph data, or Hermes state in Zone C.

### 1.1 Evidence classes

Use these labels in Photon messages, graph events, and reports:

| Label | Meaning |
|---|---|
| `OBSERVED` | Returned by a real tool, service, file, or external result. |
| `DERIVED` | Calculated from observed data with a stated rule. |
| `INFERRED` | Reasoned from observations but not directly measured. |
| `HYPOTHESIS` | Proposed mechanism that still needs a test. |
| `FALSIFIED` | Contradicted by a valid test or measurement. |
| `BLOCKED` | Required evidence or execution is unavailable. |

Use the complete chain for every material claim:

```text
claim → assumption → evidence → mechanism → action → verification → uncertainty
```

### 1.2 Evidence boundaries

| Layer | Use it for | Do not use it as proof of |
|---|---|---|
| Google Drive | Source ingress and operator-controlled files | Source correctness or task progress by itself |
| Obsidian Markdown | Human-readable research projection | Governance integrity by itself |
| `_agentic/events` | Append-only typed governance events | HENRI intelligence or external success |
| `graph_projection.json` | Deterministic node and edge view | Causal validity when edges are absent |
| Chroma vault index | Cheap semantic retrieval | Provenance or causal correctness by itself |
| TrustGraph | Relationship-heavy context and bounded specialist execution | Human approval or HENRI task success |
| Hermes audit chain | Event integrity and provenance | Model quality or external outcome |
| Kanban | Goals, ownership, approvals, blockers, and state | Code execution by itself |
| CI/Vast CUDA | Real HENRI runtime execution | Task success unless an external outcome is measured |
| Zone C | HENRI latent artifacts and approved CUDA telemetry | Research notes, Photon state, or graph memory |

---

## 2. Current deployment map

### 2.1 Local paths and services

| Component | Current path, port, or identifier | Function |
|---|---|---|
| Repository | `C:\Users\chan\Desktop\HENRI 7B SWARM` | Production source tree and workflow scripts |
| Google Drive inbox | `G:\My Drive\HENRI_Inbox` | Drop PDF, Markdown, or text research sources |
| Obsidian vault root | `G:\My Drive\HENRI_Research_Vault\HENRI_Research_Vault` | Research notes and local event projection |
| Local event records | `<vault>/_agentic/events/` | Governance source of truth |
| Graph projection | `<vault>/_agentic/graph_projection.json` | Deterministic graph projection |
| Local vault server | `http://127.0.0.1:8000` | Chroma-backed semantic retrieval and event endpoints |
| TrustGraph API gateway | `http://127.0.0.1:8088/` | TrustGraph CLI and REST endpoint |
| TrustGraph MCP host | `http://127.0.0.1:8001/mcp` | Hermes MCP server endpoint from current config |
| TrustGraph Workbench | `http://127.0.0.1:8888/` | Browser UI |
| Hermes audit ledger | `C:\Users\chan\AppData\Local\hermes\audit\henri_audit_chain.jsonl` | Hash-linked governance ledger |
| Hermes scripts | `C:\Users\chan\AppData\Local\hermes\scripts\` | Cron scripts and governance bridge |
| Hermes handoffs | `C:\Users\chan\AppData\Local\hermes\handoffs\` | Session continuation artifacts |
| MoA traces | `C:\Users\chan\AppData\Local\hermes\moa_traces\` | Separate reasoning evidence layer |

The exact vault path is controlled by `OBSIDIAN_VAULT_PATH` when set. Do not use a stale handoff path without checking the live filesystem.

### 2.2 Active Hermes cron jobs

The current scheduler reports these jobs as active and their latest inspected runs as completed:

| Job ID | Name | Schedule | Function |
|---|---|---:|---|
| `b0249aa158b1` | `henri-research-ingest` | Every 10 minutes | Reads the Drive inbox and writes research notes/events |
| `8027351ab01e` | `henri-ci` | Every 3 minutes | Polls the HENRI CI path and reports compact results |
| `6ef3bb1c1659` | `context-watchdog` | Every 15 minutes | Detects context pressure and requests handoff |
| `b7f2d33a4772` | `henri-agentic-context-progress` | Every 15 minutes | Collects read-only audit, graph, task, Git, CI, and telemetry state |

Cron `active` is not execution proof. Inspect the latest durable run and require `completed`.

### 2.3 TrustGraph status interpretation

The current live checks observed:

- Docker Compose TrustGraph containers are running.
- API gateway responds on port `8088`.
- Workbench responds on port `8888`.
- `tg-verify-system-status --skip-ui --global-timeout 30 --check-timeout 5` passed 6 of 6 checks.
- The verifier observed 38 processors, 3 flow blueprints, 1 flow, 24 prompts, and 1 library document.
- The local vault server on port `8000` was not responding during the same audit. This blocks local semantic retrieval, but does not invalidate existing local event records.

Do not infer TrustGraph provenance, HENRI integration, or external task progress from container status alone.

### 2.4 Token-minimizing holonic operating model

The intended operating model is a mobile-controlled, headless HENRI research
engine. The phone is the control surface. Deterministic collectors, TrustGraph
processors, and the remote CUDA target perform the work. The model receives
compact envelopes instead of raw logs or full transcripts.

The following distinction is mandatory:

| Statement | Current status |
|---|---|
| The scheduler can create 10-minute and 15-minute work slots | `OBSERVED` |
| 10–15 minutes corresponds to 96–144 scheduled slots per 24 hours | `DERIVED` |
| Each slot is an autonomous HENRI experiment cycle | `HYPOTHESIS`, not current evidence |
| TrustGraph agent orchestration code and a live processor are deployed | `OBSERVED` for the deployed processor and service health |
| A TrustGraph holon currently launches and evaluates HENRI experiments end to end | `BLOCKED` until a real controlled invocation returns linked IDs and a CUDA result |
| A persistent 24/7 HENRI experiment daemon is active | `BLOCKED`; the current HENRI runner is finite and the CI path launches queued scripts |
| The system provides 85% token reduction | `HYPOTHESIS` until a baseline comparison measures tokens per completed task |

Do not report 96–144 scheduled slots as 96–144 completed experiments. A slot
can be silent, blocked, a health check, a retry, or a telemetry reduction.

#### Four-tier execution cascade

Use the cheapest tier that can answer the question:

| Tier | Executor | Trigger | LLM use | Output |
|---|---|---|---:|---|
| 0 | Hermes no-agent collector | Every poll or event change | 0 tokens | Hashes, status, metrics, compact event |
| 1 | One TrustGraph leaf holon | A bounded research, audit, or telemetry question | One bounded call | Finding with source IDs and status |
| 2 | TrustGraph parent/supervisor | Independent findings must be combined or conflict | One decomposition plus one synthesis, bounded | Decision proposal and provenance IDs |
| 3 | Hermes MoA | Load-bearing math, persistent bug, or explicit approval | Three-model route | Advisory critique only |

Do not use the supervisor pattern for every task. Supervisor fan-out is useful
when the dimensions are genuinely independent. It increases calls when a task
has one answer. Use `react` for one bounded lookup, `plan-then-execute` for a
short ordered workflow, and `supervisor` only for independent specialist
boundaries.

The deployed TrustGraph CLI exposes these patterns:

```bash
tg-invoke-agent --help
```

The interface supports `react`, `plan-then-execute`, and `supervisor`, plus
`--no-streaming`, `--explainable`, and `--show-usage`. For a controlled query,
use the verified default workspace, collection, and flow only after checking
them with `tg-list-workspaces`, `tg-list-collections`, and `tg-show-flows`:

```bash
tg-invoke-agent \
  --workspace default \
  --collection default \
  --flow-id default \
  --pattern react \
  --no-streaming \
  --show-usage \
  --question 'Return a compact audit of the registered source and cite source identifiers.'
```

This command is a real model operation. Use it only with a bounded question.
Do not use it as a health probe. A health probe is:

```bash
tg-verify-system-status --skip-ui --global-timeout 30 --check-timeout 5
```

#### Context firewall contract

Every holon should receive a bounded contract, not raw parent context:

```yaml
holon_id: stable identifier
parent_holon_id: stable identifier or null
task_type: research | audit | telemetry | execution | governance
question: one bounded question
input_refs: source, event, commit, or run identifiers
source_hashes: list of hashes
allowed_paths: exact read boundary
allowed_tools: minimum tool set
max_iterations: bounded integer
max_output_chars: bounded integer
acceptance: list of checks
rejection: list of checks
```

The child returns:

```yaml
holon_id: stable identifier
status: observed | derived | inferred | falsified | blocked
finding: compact result
evidence_refs: source, event, provenance, or run identifiers
artifact_refs: paths to full output
token_usage: observed when available
failure_scope: local | parent | global
next_action: one bounded action
```

The parent does not receive raw tool traces, full PDFs, full telemetry, or
unbounded reasoning. It receives the compact contract result and can retrieve
the full artifact by reference when a human audit requires it.

#### Event-driven 24/7 target loop

The safe target loop is event-driven, not a blind model call every 10 minutes:

```text
no-agent collector
  → detect new source, failed run, metric boundary, or approved task
  → create one sealed event
  → route only the required leaf holon
  → reduce result to a compact contract
  → fan-in only when independent results exist
  → pause at approval or circuit-breaker boundary
  → run one bounded remote experiment
  → reduce telemetry deterministically
  → create a separate outcome event
  → send a Photon scorecard only on a decision-relevant change
```

Healthy ticks must be silent. A model call must not run because a timer fired
when no state changed. Use the timer to poll state; use state change to trigger
inference.

Recommended trigger policy:

| Event | Action | Human approval |
|---|---|---:|
| New Drive source | Hash, ingest, project, and run one research leaf | No, until code is proposed |
| Missing provenance | Stop the research branch and report `BLOCKED` | Yes, to change route |
| New bounded design | Create approval request | Required before code |
| Approved patch | Apply only the approved scope | Already required |
| Remote run completed | Reduce telemetry and compare criteria | Required to accept a result |
| Rank collapse, NaN, memory failure, or repeated SSH failure | Trip circuit breaker and preserve artifacts | Required to resume |
| No state change | No model call and no Photon message | No |

#### Autonomous safety controls

The target 24/7 loop must have these controls before unattended code mutation:

1. **One active lease per experiment lane.** A retry must not start a second
   copy of the same run.
2. **Idempotent run identity.** Use a task ID, approved scope hash, commit,
   and run ID to detect duplicates.
3. **Bounded episode and step budgets.** A finite HENRI run is safer than an
   unbounded process. Increase the budget only through an approved experiment.
4. **Failure budgets.** Stop after a registered number of repeated failures.
5. **Circuit breakers.** Stop on NaN/Inf, rank collapse, memory failure,
   excessive fallback, missing telemetry, or missing scorecard.
6. **Commit and artifact preservation.** Preserve the real commit, run ID,
   return code, logs, and telemetry path before retrying.
7. **Human resume gate.** A circuit breaker must not silently resume a changed
   code path.
8. **Separate rollback procedure.** A rollback command must be implemented and
   tested before it is advertised in Photon. The current audit chain does not
   prove that every commit is cryptographically signed or that a mobile
   `/rollback <hash>` command exists.

The current CI script uses `setsid` and `nohup` for a queued remote experiment,
not `tmux`. It checks the remote process with `pgrep`, throttles progress
reports, and prints compact telemetry fields. This is useful infrastructure,
but it is not evidence of a continuously self-scheduling HENRI research loop.

#### Token-efficiency measurement

Do not claim efficiency from architecture shape. Record these values for a
fixed task set and compare the current local path with the holonic path:

```text
task_id
source_hash
route_pattern
leaf_count
parent_count
moa_count
input_tokens
output_tokens
context_chars
retrieval_latency_ms
provenance_coverage
human_correction_count
remote_run_id
external_outcome
```

Accept the holonic route only when it reduces total tokens or improves evidence
quality at the same task scope without increasing unsupported claims. A graph
query that returns fewer tokens but loses provenance is not an efficiency gain.

#### Mobile executive operating mode

The intended Photon experience is a decision queue, not a terminal mirror:

```text
🟢 HOLON COMPLETE
Task: <task-id>
Route: leaf audit → parent synthesis
Finding: <one sentence>
Evidence: <source/event/provenance IDs>
Cost: <observed token usage or BLOCKED>
Decision: APPROVE / REJECT / INVESTIGATE
Artifact: <path or run ID>
```

Send a Photon message only when one of these occurs:

- a new source is ingested;
- a design needs approval;
- a circuit breaker trips;
- a remote run starts or completes;
- acceptance or rejection criteria change;
- a human decision is required.

Do not send routine healthy polls.

#### Current gaps before claiming the target model

The following implementation work remains separate from this manual:

1. Connect `henri_ingest.py` to a verified TrustGraph upload and processing
   consumer while retaining the local audit event as the authority.
2. Add a deterministic event router that triggers TrustGraph only on relevant
   state changes.
3. Add a compact holon envelope and result validator with token-usage capture.
4. Connect approved TrustGraph findings to `henri_governance.py` without
   allowing the child holon to approve itself.
5. Add a remote experiment lease, duplicate-run guard, and circuit breaker.
6. Add deterministic telemetry reduction and threshold events for rank,
   divergence, memory, fallback, and scorecard failures.
7. Add a real end-to-end controlled test: Drive source → audit event →
   TrustGraph document/processing IDs → explainable result → approval → remote
   CUDA run → telemetry → outcome.

Do not implement these as one large autonomous wrapper. Implement one bounded
consumer at a time and verify each returned effect.

---

## 3. First-time setup and prerequisites

### 3.1 Required local software

Install or verify:

- Hermes Agent v0.19 or later.
- Python 3.11 used by the Hermes environment.
- Docker Desktop with Docker Compose.
- Git.
- Google Drive for Desktop with the HENRI folders available.
- A configured Photon platform connection.
- TrustGraph CLI commands in the Hermes environment.

Use the MSYS/Git Bash syntax shown below. Do not use PowerShell syntax in the Hermes terminal.

### 3.2 Verify the repository and Hermes

```bash
cd "C:/Users/chan/Desktop/HENRI 7B SWARM"
hermes --version
hermes status
hermes cron list
hermes tools list
```

Confirm:

- Hermes reports the expected Python version.
- the repository path is correct;
- the TrustGraph MCP server is listed in the Hermes configuration;
- the four HENRI jobs are present;
- Photon is enabled through the installed platform plugin;
- no command displays a secret value.

### 3.3 Verify Google Drive for Desktop ingress

```bash
python - <<'PY'
from pathlib import Path
inbox = Path(r"G:/My Drive/HENRI_Inbox")
vault = Path(r"G:/My Drive/HENRI_Research_Vault/HENRI_Research_Vault")
print({
    "inbox_exists": inbox.exists(),
    "vault_exists": vault.exists(),
    "inbox": str(inbox),
    "vault": str(vault),
})
PY
```

The local ingest script accepts these file types:

- `.pdf`
- `.md`
- `.txt`

A Google-native `.gdoc`, `.gsheet`, or `.gslides` file is not normal document content when viewed through a Drive-for-Desktop shortcut. Export it through the Google API first, or use the matching local export. Do not read the shortcut file as if it were the document body.

### 3.4 Optional Google Workspace API setup

The mounted Drive path is sufficient for the current local ingest path. The Google Workspace API is needed for Google-native file export and API-based Drive operations.

Check authentication:

```bash
python "C:/Users/chan/AppData/Local/hermes/skills/productivity/google-workspace/scripts/setup.py" --check
```

A result of `AUTHENTICATED` permits API operations. A result of `NOT_AUTHENTICATED` means:

- mounted PDF/Markdown/text ingestion can still work if Drive for Desktop is online;
- `.gdoc`, `.gsheet`, and `.gslides` export is blocked;
- do not claim Google API integration is ready.

Do not place OAuth tokens, client secrets, or API keys in the repository.

### 3.5 Verify TrustGraph

```bash
tg-verify-system-status --skip-ui --global-timeout 30 --check-timeout 5
```

Then verify the user interfaces:

```bash
curl -sS -I http://127.0.0.1:8088/ | head
curl -sS -I http://127.0.0.1:8888/ | head
```

A `404` from the API root can still mean that the gateway is reachable. The dedicated verifier is the service-health check. A `200` from the Workbench root confirms that the UI is serving, not that a graph query has succeeded.

Open the Workbench in a browser:

```text
http://127.0.0.1:8888/
```

### 3.6 Verify local governance records

```bash
cd "C:/Users/chan/Desktop/HENRI 7B SWARM"
export OBSIDIAN_VAULT_PATH='G:/My Drive/HENRI_Research_Vault/HENRI_Research_Vault'
python scripts/agentic_graph_cli.py verify
python scripts/agentic_graph_cli.py project
python "C:/Users/chan/AppData/Local/hermes/scripts/henri_audit.py" verify
```

These checks prove different properties:

- `agentic_graph_cli.py verify` checks local event envelopes and payload hashes.
- `agentic_graph_cli.py project` regenerates the deterministic projection.
- `henri_audit.py verify` checks the Hermes hash-linked audit chain.

Run all three when a governance result matters.

---

## 4. Daily workflow: research to verified HENRI change

### Step 1 — Define one bounded goal

Before placing a source in the inbox, define a narrow goal. Good examples:

```text
Assess whether the source supports a candidate-specific constraint penalty in efe_planner.py.

Audit the live caller and consumer for the proposed EDMD rank change.

Compare the proposed wave update with the current Stiefel retraction contract.
```

Bad example:

```text
Improve the whole HENRI architecture using this paper.
```

Record or communicate:

- task ID;
- one claim;
- intended file or subsystem boundary;
- acceptance criteria;
- rejection criteria;
- required evidence class.

### Step 2 — Place the source in Google Drive

From any device:

1. Open Google Drive.
2. Open `My Drive/HENRI_Inbox`.
3. Add one uniquely named PDF, Markdown, or text file.
4. Use a name that identifies the source and experiment, for example:

```text
2026-07-27_constraint-penalty-controlled-source.pdf
```

Do not use a source filename that is likely to be reused. The ingest state skips an unchanged combination of filename, size, and modification time.

For a controlled test, add a note in the document title or first line:

```text
CONTROLLED TEST — NOT A RESEARCH CLAIM — TrustGraph/Drive ingest probe
```

### Step 3 — Wait for or inspect research ingestion

The no-agent job runs every 10 minutes. Inspect the latest durable run:

```bash
hermes cron runs b0249aa158b1 --limit 5
```

Require the latest run to show `completed`. A scheduler trigger acknowledgement is not sufficient.

The expected artifacts are:

1. Markdown note under:
   `G:\My Drive\HENRI_Research_Vault\HENRI_Research_Vault\ArXiv_Corpus\Inbox\`
2. `PAPER_INGESTED` event under `<vault>/_agentic/events/`.
3. A 64-character `audit_hash` in the event.
4. A matching `payload_hash`.
5. An updated `graph_projection.json`.
6. A reindex request if the local vault server is running.

A healthy ingest message is compact. It identifies the source, note, character/page count, and vault-index status. It should not include the full PDF or raw logs.

### Step 4 — Start local semantic retrieval when needed

The local vault server is a derived Chroma projection. Start it with the Hermes Python 3.11 interpreter:

```bash
cd "C:/Users/chan/Desktop/HENRI 7B SWARM"
export OBSIDIAN_VAULT_PATH='G:/My Drive/HENRI_Research_Vault/HENRI_Research_Vault'
"C:/Users/chan/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe" \
  scripts/local_vault_search_server.py \
  --vault "$OBSIDIAN_VAULT_PATH" \
  --db "$OBSIDIAN_VAULT_PATH/_agentic/vector_index" \
  --port 8000
```

Run the server in a tracked background process when needed. Check health:

```bash
curl -sS http://127.0.0.1:8000/health
```

Query the vault with bounded context:

```bash
curl -G -sS http://127.0.0.1:8000/query \
  --data-urlencode 'q=candidate-specific constraint penalty' \
  --data-urlencode 'top_k=5'
```

Use time and module filters when available. Do not send the complete vault or raw server output to Photon.

### Step 5 — Use TrustGraph for relationship-heavy questions

Use local Chroma first for cheap semantic retrieval. Use TrustGraph when the question needs explicit relationships, provenance, or bounded specialist work.

The TrustGraph path is:

```text
source hash
  → local audit event
  → TrustGraph library document
  → processing flow
  → GraphRAG query
  → explainability/provenance result
  → compact local event
```

Do not process the full Drive inbox through GraphRAG on first activation. Use two controlled documents.

### Step 6 — Produce a claim audit and design

The agent must return a bounded design before implementation:

```text
Claim:
Assumption:
Evidence:
Mechanism:
Action:
Verification:
Uncertainty:
```

The design must state:

- live code path;
- mathematical or physical assumption;
- tensor and device boundaries, when relevant;
- resource limit;
- expected benefit;
- failure mode;
- cheapest kill experiment;
- acceptance criteria;
- rejection criteria.

The agent must wait for a human decision after this step for load-bearing changes.

### Step 7 — Review in Photon

Use Photon as the control surface. Request a compact review summary, not a raw transcript.

Recommended request:

```text
Review task <task-id>. Use the Drive source and governed graph evidence.
Return: claim, evidence class, live files, proposed mechanism, changed scope,
acceptance criteria, rejection criteria, and the exact decision required.
Do not modify code or launch a run.
```

The response should use:

```text
STATUS: DESIGN_REVIEW
EVIDENCE: OBSERVED / DERIVED / INFERRED / BLOCKED
GOAL: <one sentence>
DONE: <latest verified event>
CURRENT: <task and state>
RISK: <one material risk>
DECISION: APPROVE_DESIGN or REJECT_DESIGN or BLOCK
NEXT: <one bounded action>
ARTIFACT: <path, event ID, or run ID>
```

Photon messages must remain compact. Store full artifacts in the repository, vault, audit ledger, or remote run directory.

### Step 8 — Record human approval

A model response is not human approval. Record the decision through the deterministic governance bridge:

```bash
python "C:/Users/chan/AppData/Local/hermes/scripts/henri_governance.py" \
  approve \
  --task-id t_example123 \
  --scope 'HENRI V2/efe_planner.py:100-180' \
  --assumption 'the penalty is candidate-specific and dimension-normalized' \
  --accept 'remote CUDA suite completes' \
  --accept 'candidate ranking changes in the registered comparison' \
  --reject 'all candidates receive the same additive score' \
  --channel photon \
  --reason 'Approve bounded constraint-penalty experiment'
```

The bridge verifies the Hermes audit chain before it writes the decision. Use `reject` or `block` when appropriate:

```bash
python "C:/Users/chan/AppData/Local/hermes/scripts/henri_governance.py" \
  reject \
  --task-id t_example123 \
  --scope 'HENRI V2/efe_planner.py:100-180' \
  --channel photon \
  --reason 'Provenance is incomplete; revise the design'
```

Do not widen the approved scope during implementation. Create a new approval for a new scope.

### Step 9 — Implement one bounded change

Before editing:

1. Inspect the live caller and consumer.
2. Confirm that every cited file and symbol exists.
3. Trace every configuration value to its consumer.
4. Keep the default path unchanged unless the experiment requires a default change.
5. Put A/B behavior behind a named flag.
6. Archive deprecated HENRI code under `HENRI V2/_archive/` instead of deleting it.
7. Keep user data, vault state, raw telemetry, and secrets outside the production tree.

Do not use graph-shaped demonstrator files as production execution evidence. A hardcoded plan, approval, telemetry dictionary, or simulated Vast result is a mock loop.

### Step 10 — Commit and run remote verification

The default path is:

```text
commit → push → henri-ci → CUDA verification → compact telemetry
```

Do not use local CPU tests as HENRI runtime verification. Local protocol checks may validate the event-store mechanics, but HENRI runtime and mathematical claims require the Vast CUDA target or canonical HENRI CI.

Inspect the CI cron:

```bash
hermes cron runs 8027351ab01e --limit 5
```

Require a real return code, commit or run identifier, CUDA execution result, and artifact path.

### Step 11 — Measure and create separate outcome records

Collect compact telemetry such as:

- external task score or documented outcome;
- prediction loss and trajectory shape;
- Sagnac delta and coherence;
- Kuramoto order parameter;
- EFE decomposition;
- candidate rejection/admission;
- fallback execution;
- rank and cross-window subspace overlap;
- GPU memory and step latency;
- learning engagement and parameter change.

Keep these meanings separate:

| Stream | Example | External task progress? |
|---|---|---:|
| `coherence` | Sagnac delta, Kuramoto order | No, by itself |
| `frame_change` | Environment observation changed | No, by itself |
| `execution` | CI return code, commit, run ID | No, by itself |
| `telemetry` | Reduced GPU/runtime metrics | No, by itself |
| `outcome` | Score, WIN, level completion | Yes, after causal checks |

Create explicit causal edges only when the relationship is supported:

```text
RESEARCH_COLLECTED --SUPPORTS--> DESIGN_PROPOSED
DESIGN_PROPOSED --REQUIRES_APPROVAL--> HUMAN_DECISION
HUMAN_DECISION --IMPLEMENTS--> PATCH_APPLIED
PATCH_APPLIED --VERIFIED_BY--> REMOTE_RUN_COMPLETED
REMOTE_RUN_COMPLETED --MEASURES--> TELEMETRY_REDUCED
TELEMETRY_REDUCED --SUPPORTS or CONTRADICTS--> OUTCOME_ACCEPTED/REJECTED
```

### Step 12 — Report the result in Photon

Use a compact scorecard:

```text
STATUS: REMOTE_VERIFY_COMPLETE
EVIDENCE: OBSERVED
GOAL: <task goal>
DONE: CUDA run <run-id> returned <status>
CURRENT: <acceptance/rejection decision pending or complete>
RISK: <one material uncertainty>
DECISION: ACCEPT_RESULT or REJECT_RESULT or INVESTIGATE
NEXT: <one falsification or maintenance action>
ARTIFACT: <CI URL, run ID, telemetry path, audit head>
```

Never state that the task improved unless a real external outcome supports that statement.

---

## 5. TrustGraph operating procedure

### 5.1 Understand TrustGraph objects

Use these boundaries:

- **Workspace** — top-level TrustGraph isolation boundary.
- **Collection** — approved research domain or bounded document group.
- **Flow** — processing or retrieval pipeline.
- **Library document** — uploaded source with a stable identifier.
- **Processing ID** — identifier for one processing request.
- **GraphRAG query** — relationship-aware retrieval and synthesis request.
- **Explainability result** — provenance and grounding output associated with a query.

Never guess a workspace, collection, flow, document ID, or processing ID in an evidence record. Verify each identifier from the live service response.

### 5.2 Controlled two-document probe

Use a dedicated collection and two uniquely named controlled fixtures. The following commands show the interface. Replace placeholders only with IDs returned by the service or approved configuration.

```bash
HASH='<content-hash>'
DOC_ID="urn:henri:probe:${HASH}"
PROCESS_ID="urn:henri:processing:${HASH}"

# Upload one controlled text document.
tg-add-library-document \
  --name 'HENRI controlled probe' \
  --description 'Controlled TrustGraph integration test; not research evidence' \
  --identifier "$DOC_ID" \
  --kind text/plain \
  --tags 'henri,controlled-test' \
  probe.md

# Start processing with a verified flow and collection.
tg-start-library-processing \
  --flow-id '<verified-flow-id>' \
  --document-id "$DOC_ID" \
  --processing-id "$PROCESS_ID" \
  --collection '<verified-collection-id>'

# Query with explainability enabled.
tg-invoke-graph-rag \
  --flow-id '<verified-flow-id>' \
  --collection '<verified-collection-id>' \
  --question 'What relationships are present in the controlled probe?' \
  --explainable
```

Acceptance requires:

- the document upload returns a real document identifier;
- processing returns a real processing identifier;
- processing reaches a verified completed state;
- GraphRAG returns source or provenance identifiers;
- the source hash links to the TrustGraph document identifier;
- the local audit event and graph projection remain valid;
- the second controlled document works after a process restart.

Reject or mark `BLOCKED` when:

- only package import succeeds;
- only the UI loads;
- only the upload succeeds;
- the query returns an answer without provenance;
- a guessed collection or flow is used;
- the full inbox is processed before the controlled probe passes.

### 5.3 Use TrustGraph efficiently

Use TrustGraph for:

- multi-document relationship questions;
- provenance-sensitive research synthesis;
- bounded supervisor, ReAct, or plan-then-execute specialist work;
- graph traversal where flat vector retrieval loses relations.

Use deterministic tools instead for:

- hashing;
- file discovery;
- audit verification;
- cron polling;
- telemetry reduction;
- graph projection;
- status checks.

Do not send raw TrustGraph traces to Photon. Save the full response as an artifact and pass only the source IDs, result status, compact finding, and artifact path.

### 5.4 TrustGraph and Hermes MCP

The current Hermes configuration contains a TrustGraph MCP server at:

```text
http://127.0.0.1:8001/mcp
```

A configured MCP endpoint is not proof that the MCP server completed a useful operation. Verify the tool call's returned status, workspace, flow, document ID, and provenance. If the MCP path fails, use the verified CLI or REST path for diagnosis. Do not create a wrapper that only renames the same operation and call it a new integration.

---

## 6. Photon operating procedure

### 6.1 Photon is a control surface

Photon carries:

- the current goal;
- compact status;
- the human decision;
- the next bounded action;
- the artifact or run identifier.

Photon does not replace:

- the audit ledger;
- the repository;
- the Kanban task;
- the canonical handoff;
- the CI artifact;
- the external outcome record.

### 6.2 Recommended commands and message forms

Use plain, explicit messages:

```text
status <task-id>
```

```text
research <source-title> for <bounded question>
```

```text
plan <source-title>
```

```text
APPROVE_DESIGN <task-id> scope=<exact scope>
```

```text
APPROVE_IMPLEMENTATION <task-id> scope=<exact scope>
```

```text
APPROVE_REMOTE_RUN <task-id> run=<registered run scope>
```

```text
REJECT_RESULT <task-id> reason=<specific reason>
```

If the active Photon adapter uses different command parsing, preserve the same decision vocabulary and include the task ID and scope.

### 6.3 Photon response contract

Keep a material response below the configured mobile limit. The current Hermes platform hint requests a compact response under approximately 1,800 characters.

```text
STATUS: <stage>
EVIDENCE: <class>
GOAL: <one sentence>
DONE: <latest verified event>
CURRENT: <durable task or blocker>
RISK: <one material risk>
DECISION: <exact response required>
NEXT: <one bounded action>
ARTIFACT: <path, run ID, or audit head>
```

Do not send:

- raw logs;
- full PDF text;
- full session history;
- large Git diffs;
- unbounded TrustGraph traces;
- credentials;
- fabricated telemetry.

### 6.4 Human approval rules

Require a sealed human decision before:

- a load-bearing mathematical change;
- an architecture or schema change;
- a production or expensive remote run;
- a destructive operation;
- accepting a benchmark as task progress.

A Photon message that says “looks good” is not sufficient if it does not identify the task, scope, assumptions, acceptance criteria, and rejection criteria. Use the governance bridge so the decision enters the audit ledger.

### 6.5 Context handoff

When the context watchdog reports pressure:

1. stop before a half-applied mutation;
2. collect repository, audit, Kanban, CI, and process state;
3. write the canonical handoff under `C:\Users\chan\AppData\Local\hermes\handoffs\`;
4. preserve the current task ID, audit head, approval scope, and next action;
5. start a new session;
6. read the handoff;
7. recheck live state before acting.

Use:

```text
handoff
```

Then, in the new session:

```text
resume
```

A successful compression or handoff proves context management only. It does not prove the HENRI change or its task outcome.

### 6.6 Photon troubleshooting discipline

Separate these failure states:

1. Photon configured.
2. Hermes gateway process running.
3. Photon adapter connected.
4. Sidecar healthy.
5. Outbound send accepted.
6. User-visible delivery received.
7. Session context continued correctly.
8. Agent execution completed.

Evidence at one level does not prove the next level. Do not send a real outbound probe only to test delivery without approval. Prefer a fresh inbound Photon round trip. If no round trip is available, report `BLOCKED`.

---

## 7. Local graph command reference

Set the vault path in each new shell:

```bash
export OBSIDIAN_VAULT_PATH='G:/My Drive/HENRI_Research_Vault/HENRI_Research_Vault'
```

### Verify events and projection

```bash
python scripts/agentic_graph_cli.py verify
python scripts/agentic_graph_cli.py project
python "C:/Users/chan/AppData/Local/hermes/scripts/henri_audit.py" verify
```

### Query research events

```bash
python scripts/agentic_graph_cli.py query \
  --stream research \
  --event-type PAPER_INGESTED \
  --limit 20
```

### Query a bounded time window

```bash
python scripts/agentic_graph_cli.py query \
  --stream telemetry \
  --after 2026-07-01T00:00:00Z \
  --before 2026-08-01T00:00:00Z \
  --limit 50
```

### Create a governed event

```bash
python scripts/agentic_graph_cli.py event CLAIM_AUDITED \
  --stream claim_audit \
  --actor human \
  --status observed \
  --payload '{"claim":"candidate-specific penalty changes ranking","source":"docs/example-audit.md"}' \
  --source-uri 'repo:docs/example-audit.md'
```

Save the returned event ID and audit hash in the task artifact.

### Create a causal edge

```bash
python scripts/agentic_graph_cli.py edge \
  '<source-event-id>' \
  '<target-event-id>' \
  SUPPORTS \
  --actor human \
  --status derived
```

Valid edge types include:

```text
SUPPORTS
CONTRADICTS
DERIVED_FROM
TRIGGERS
REQUIRES_APPROVAL
VERIFIED_BY
FALSIFIED_BY
IMPLEMENTS
MEASURES
CONSUMES
SEPARATE_FROM
```

A graph with nodes and zero edges is an event inventory, not a causal development graph.

---

## 8. Health checks and acceptance checklist

### 8.1 Compact health sweep

```bash
cd "C:/Users/chan/Desktop/HENRI 7B SWARM"
export OBSIDIAN_VAULT_PATH='G:/My Drive/HENRI_Research_Vault/HENRI_Research_Vault'

hermes --version
hermes cron list
python scripts/agentic_graph_cli.py verify
python scripts/agentic_graph_cli.py project
python "C:/Users/chan/AppData/Local/hermes/scripts/henri_audit.py" verify
tg-verify-system-status --skip-ui --global-timeout 30 --check-timeout 5
curl -sS http://127.0.0.1:8000/health
hermes cron runs b0249aa158b1 --limit 1
hermes cron runs 8027351ab01e --limit 1
hermes cron runs 6ef3bb1c1659 --limit 1
hermes cron runs b7f2d33a4772 --limit 1
```

Interpret each result separately. A failed vault health check does not invalidate the audit chain. A passing TrustGraph verifier does not prove a source-grounded HENRI claim. A passing CI cron record does not prove an external outcome.

### 8.2 Research-ingest acceptance

- [ ] Source exists in `G:\My Drive\HENRI_Inbox`.
- [ ] File type is supported or was exported from a Google-native format.
- [ ] Ingest cron latest run is `completed`.
- [ ] Markdown note exists in the vault inbox.
- [ ] `PAPER_INGESTED` event exists.
- [ ] Event has a 64-character audit hash.
- [ ] Payload hash recomputes.
- [ ] Hermes audit chain verifies.
- [ ] Graph projection parses.
- [ ] Semantic reindex completed or is explicitly `BLOCKED` because the server is down.

### 8.3 TrustGraph acceptance

- [ ] Package import and package version are observed separately.
- [ ] API verifier passes.
- [ ] Workspace is verified.
- [ ] Collection is verified.
- [ ] Flow is verified.
- [ ] Controlled document upload returns an ID.
- [ ] Processing returns an ID and reaches completion.
- [ ] GraphRAG returns source/provenance identifiers.
- [ ] Source hash links to the document ID.
- [ ] Second controlled document works after restart.
- [ ] Local audit and event verification still pass.

### 8.4 Implementation acceptance

- [ ] Claim audit exists.
- [ ] Live caller and consumer were inspected.
- [ ] Design states assumptions, failure mode, and kill experiment.
- [ ] Human approval is sealed.
- [ ] Changed scope matches approved scope.
- [ ] Default behavior is preserved or the change is explicitly registered.
- [ ] No mock payload, simulated result, or hardcoded approval exists.
- [ ] Commit and push are real.
- [ ] HENRI CUDA verification returns a real status and run ID.
- [ ] Compact telemetry artifact exists.
- [ ] External outcome is recorded separately.
- [ ] Acceptance or rejection criteria are evaluated.

---

## 9. Common troubleshooting

### 9.1 “I dropped a file into Drive and nothing happened.”

Check in order:

1. The file is in `G:\My Drive\HENRI_Inbox`, not Drive root.
2. Google Drive for Desktop is online.
3. Open the inbox once in Explorer so a streamed file is materialized.
4. The extension is `.pdf`, `.md`, or `.txt`.
5. The filename, size, and modification time changed. The ingest state skips an unchanged signature.
6. Inspect the durable cron run:

```bash
hermes cron runs b0249aa158b1 --limit 5
```

If the inbox path is missing, the script reports a Drive-for-Desktop warning. Do not call the workflow healthy until the path is restored.

### 9.2 “The PDF is ingested, but semantic search returns no result.”

The local vault server is probably stopped or stale.

```bash
curl -sS http://127.0.0.1:8000/health
```

If it refuses the connection, start `scripts/local_vault_search_server.py` with the Hermes Python 3.11 interpreter. If it responds but lacks the new note, trigger a reindex or restart the server. Existing local events can remain valid while semantic retrieval is blocked.

### 9.3 “The Google API check says `NOT_AUTHENTICATED`.”

The Google Workspace OAuth token is missing. Mounted Drive files can still work through Drive for Desktop. Google-native export and API operations are blocked until OAuth setup completes. Do not copy credentials into the repo or use a guessed token path.

### 9.4 “TrustGraph CLI imports, but the service does not work.”

Package installation and service integration are separate claims.

Run:

```bash
tg-verify-system-status --skip-ui --global-timeout 30 --check-timeout 5
docker compose ps --all
```

Then inspect the API gateway and service logs. Do not report TrustGraph integration from `import trustgraph` alone.

### 9.5 “The TrustGraph verifier passes, but GraphRAG has no provenance.”

Service health is not provenance evidence. Check:

- correct workspace;
- correct collection;
- correct flow;
- document processing completion;
- `--explainable` query output;
- source hash to document-ID link.

If provenance is missing, mark the result `BLOCKED` and do not use it for implementation approval.

### 9.6 “Docker Compose reports an unset variable.”

Inspect the variable name without printing its value:

```bash
docker compose config --environment 2>&1 | grep -E 'OLLAMA_HOST|IAM_BOOTSTRAP_TOKEN|GF_SECURITY_ADMIN_PASSWORD' || true
```

The current deployment uses `IAM_BOOTSTRAP_TOKEN` in the control service. Keep secrets in the environment or a secret file. Do not commit `.env` contents. An unset `OLLAMA_HOST` is a configuration review item if the selected flow requires Ollama.

### 9.7 “The Workbench opens, but the system is not healthy.”

The UI is only one layer. Run:

```bash
tg-verify-system-status --skip-ui --global-timeout 30 --check-timeout 5
```

A browser page proves UI reachability, not processor health, flow health, library access, or provenance.

### 9.8 “The local event store fails with `AuditUnavailable`.”

The event store refuses to append an event without a Hermes audit seal. Check:

```bash
python "C:/Users/chan/AppData/Local/hermes/scripts/henri_audit.py" verify
```

Do not bypass the seal or write a manual event. Restore the Hermes script path or repair the audit service first. If the ledger is corrupt, preserve the intact prefix and treat later events as unprovenanced until restored.

### 9.9 “The graph projection has nodes but no edges.”

This is an event inventory, not a causal development graph. Create explicit `EDGE_CREATED` events only when the causal relation is supported. Do not infer causality from event order alone.

### 9.10 “Photon says the job completed, but no result arrived.”

Separate:

- agent execution;
- cron completion;
- gateway delivery;
- Photon sidecar health;
- user-visible receipt.

Inspect the durable cron result and the remote artifact first. Do not rerun an expensive HENRI experiment only because the Photon message was not delivered. Use a fresh inbound Photon round trip for delivery verification. If it is unavailable, report `BLOCKED`.

### 9.11 “Photon context became too large.”

Use the context watchdog and handoff workflow. Do not paste the complete handoff into Photon. Send only the current status, blocker, audit head, task ID, and next action.

### 9.12 “A cron job is active but the last run failed.”

`active` describes scheduling state, not last execution health.

```bash
hermes cron runs <job-id> --limit 10
```

Inspect the latest completed or failed run. Fix the actual error, then trigger one bounded retry and inspect the durable result.

### 9.13 “The HENRI runtime test passed locally.”

That is not HENRI verification under the project policy. Run the approved HENRI CI or Vast CUDA path. Local event-store tests prove governance mechanics only.

### 9.14 “The CI job says OK but no scorecard arrived.”

Check the remote target directly through the approved recovery path. Verify the remote commit, CUDA return code, run ID, and telemetry artifact. Treat Photon delivery as a separate failure domain. Do not modify the scientific result because the report transport failed.

### 9.15 “A new configuration value has no effect.”

Trace the value from source to consumer:

1. configuration file or environment;
2. Hermes process load;
3. cron or plugin process environment;
4. runtime consumer;
5. returned behavior.

Restart Hermes after configuration or plugin changes. A value present in a config file is not evidence that the active process read it.

### 9.16 “The TrustGraph package version differs from the Docker image.”

This is possible because the Hermes CLI package and container image are separate layers. Record both versions. Verify the live service through the API and the verifier. Do not call the layers identical without evidence.

---

## 10. Frequently asked questions

### Q1. Is TrustGraph the source of truth?

No. The local HENRI event store and Hermes audit chain provide governance provenance. TrustGraph is an optional context and bounded specialist-execution layer.

### Q2. Is the Obsidian vault the source of truth?

The append-only event records under `<vault>/_agentic/events/` are the governance graph source of truth. Markdown notes, Chroma, and `graph_projection.json` are projections or retrieval layers.

### Q3. Why use Google Drive instead of uploading directly in Photon?

Google Drive provides a durable operator-controlled ingress path that is accessible from desktop and mobile devices. Photon should carry the request and decision, not large source files or full transcripts.

### Q4. Can I place a Google Doc shortcut in the inbox?

Not as document content. A `.gdoc` file is normally a shortcut/metadata file in a Drive-for-Desktop mount. Export it through the Google Docs/Drive API or save a supported PDF, Markdown, or text representation.

### Q5. Can TrustGraph process the entire inbox automatically?

Not on first activation. Use two controlled documents, verify IDs and provenance, then expand only after the acceptance criteria pass.

### Q6. What should I ask TrustGraph to do?

Ask relationship-heavy, provenance-sensitive questions. Use the local vault server for cheap semantic retrieval and deterministic tools for status, hashing, polling, projection, and telemetry reduction.

### Q7. Can TrustGraph approve a patch?

No. TrustGraph can propose or summarize. A human approval event must be recorded through the governance bridge before a load-bearing implementation or remote run.

### Q8. Can a passing audit chain prove that HENRI improved?

No. It proves event integrity and provenance. HENRI improvement requires a real external outcome or registered benchmark result with causal and statistical checks.

### Q9. Can internal coherence count as task progress?

No, not by itself. Sagnac, Kuramoto, EFE, or other internal metrics can diagnose the model. They do not replace a score, WIN, level completion, or documented external result.

### Q10. Why does the graph need explicit edges?

Event order does not prove causality. Explicit typed edges connect research, design, approval, patch, remote verification, telemetry, and outcome while preserving the reason for each connection.

### Q11. When should I use MoA?

Use solo execution for routine work. Use MoA only for a narrow mathematical derivation, a load-bearing multi-file audit, a bug that survived two genuine fixes, or explicit user direction. Keep the session limit at three MoA calls.

### Q12. Does a TrustGraph UI response prove a GraphRAG result?

No. Require the API result, verified workspace/collection/flow, document and processing IDs, and explainability/provenance identifiers.

### Q13. Is the local vault server required for event integrity?

No. The vault server is a derived semantic-retrieval service. A stopped server blocks retrieval, not already sealed event records. Start it when semantic search is needed.

### Q14. Can I store HENRI wave checkpoints in the event store?

No. Store compact references, hashes, run IDs, and artifact paths. Keep wave checkpoints and latent artifacts in the approved Zone C boundary.

### Q15. What does `BLOCKED` mean?

It means the required evidence or execution is unavailable. It is not a failed scientific result. Preserve the local evidence and state the missing prerequisite.

### Q16. What is the cheapest valid kill experiment for TrustGraph?

Use two uniquely named controlled documents. Compare the local retrieval path and TrustGraph path at a fixed question set and context budget. Reject TrustGraph if it adds cost without higher provenance coverage, lower correction count, or better relationship retrieval.

### Q17. What should I do after a Hermes update?

Recheck Hermes version, resume paused cron jobs, restart helper processes such as the vault server, run the health sweep, and verify the TrustGraph stack. Do not trust an old completion report after an update.

### Q18. What should I do if a design changes after approval?

Stop and request a new approval with the new scope, assumptions, and acceptance/rejection criteria. Do not silently widen an approved event.

### Q19. Where do full logs go?

Keep them in the relevant local, vault, CI, or remote artifact path. Photon receives only a compact reduction and an artifact identifier.

### Q20. What is the final completion standard?

A task is complete only when the requested artifact exists, the real execution path ran, the returned status and output were inspected, evidence classes are labelled, failures and uncertainty are stated, and the next falsification or maintenance action is clear.

---

## 11. Operator quick reference

### New research source

```text
1. Place PDF/MD/TXT in G:\My Drive\HENRI_Inbox.
2. Wait for b0249aa158b1 or inspect its durable run.
3. Verify note, PAPER_INGESTED event, hashes, projection, and audit chain.
4. Start vault server if semantic retrieval is needed.
5. Use TrustGraph only for verified relationship/provenance work.
6. Send a compact plan to Photon.
7. Wait for sealed human approval.
```

### Approved implementation

```text
1. Re-read approval scope.
2. Inspect live caller and consumer.
3. Apply one bounded patch.
4. Record PATCH_APPLIED.
5. Commit and push.
6. Inspect b0249aa158b1 or 8027351ab01e as applicable.
7. Verify on HENRI CI/Vast CUDA.
8. Reduce telemetry.
9. Record outcome separately.
10. Send the Photon scorecard.
```

### If anything is uncertain

```text
Do not guess the ID, path, flow, collection, result, or approval.
Mark the step BLOCKED.
Preserve the last verified artifact.
Report the exact missing evidence and one bounded next action.
```

---

## 12. Source files for maintenance

This manual is derived from the live repository and active Hermes workflow files:

- `docs/HENRI_AGENTIC_GRAPH_ENGINE_GUIDE.md`
- `docs/HENRI_AGENTIC_OPS_GUIDE.md`
- `scripts/agentic_event_store.py`
- `scripts/agentic_graph_cli.py`
- `scripts/local_vault_search_server.py`
- `C:\Users\chan\AppData\Local\hermes\scripts\henri_ingest.py`
- `C:\Users\chan\AppData\Local\hermes\scripts\henri_governance.py`
- `C:\Users\chan\AppData\Local\hermes\scripts\henri_agentic_context_collector.py`
- `docker-compose.yaml`
- `trustgraph/config.json`
- `INSTALLATION.md`
- Hermes skills `henri-holonic-graph`, `henri-mobile-governance`, `henri-agent-integration`, `google-workspace`, and `henri-soul`

**Maintenance rule:** re-run the live health checks before updating status claims. A package import, stale handoff, old cron report, or static configuration file is not current service evidence.

---

## Evidence note for this manual

- `OBSERVED`: TrustGraph Compose stack was running; the TrustGraph verifier returned 6/6 checks; API gateway and Workbench endpoints were reachable; HENRI cron jobs were listed active with completed latest inspected runs; the Drive inbox and vault event paths existed; Hermes Photon plugin configuration and TrustGraph MCP configuration were present.
- `OBSERVED`: Google Workspace `setup.py --check` returned `NOT_AUTHENTICATED`; therefore Google-native API export is documented as optional and currently blocked.
- `OBSERVED`: The local vault server on `127.0.0.1:8000` refused the health probe during the audit; semantic retrieval is therefore conditional on starting the server.
- `INFERRED`: Photon is intended to be the human control surface from active configuration and the mobile-governance contract. A fresh user-visible Photon round trip is still required before claiming end-to-end delivery.
- `BLOCKED`: No claim in this manual treats TrustGraph output, audit integrity, internal coherence, or CI completion as external HENRI task success without a separate outcome record.

**Last live audit basis:** 2026-07-27.
