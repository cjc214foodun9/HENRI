# ScientistTwo → HENRI bounded engineering workflow

## Source and claim boundary

Primary source: Nam et al., *ScientistTwo: Pioneering the Human Knowledge Frontier with Autonomous AI*, arXiv:2609.19644v1, https://arxiv.org/abs/2609.19644v1 . Attached PDF: 71 physical pages, SHA-256 `98fb7802ec28beea159895a3de219307e019867b16296dfe4a79e68daaef9a17`.

The paper describes procedures and author-reported outcomes. This is an instruction-level adaptation, not a replication or a measured HENRI gain. Model families, review scores, success rates, and provider costs do not transfer by citation.

| Paper mechanism | Physical PDF pages | HENRI adoption |
|---|---:|---|
| bounded candidate → critic → refine | 5, 7, 30 | one named gate; explicit accept/reject/refine/blocked; fixed repair budget |
| baseline reproduction then subset/full-set ladder | 6–7 | matched baseline and real-data scaffold on approved remote target |
| evolution from successful and failed traces | 7–8 | retain rejection history; use STRACE before repeated repairs |
| component ablation and result comparison | 8, 30–32 | fixed evaluator/readout, discriminating control, causal attribution |
| review objection → supplementary experiment | 8–9 | targeted check, not a prose edit for reviewer score |
| keep previous best after failed refinement | 9 | parent stays accepted; dependent artifacts invalidated on code change |
| four integrity checks | 14–15, 31–32 | reproduce score, preserve spec, resolve citations, trace method to code |

Do not copy the paper's novelty-ranking or critic score as objective truth. Novelty search used two reference papers (p30); that does not prove exhaustive novelty. Its ScholarPeer reviewer is used both for refinement and evaluation; the paper acknowledges this dependency (p13). A held-out reviewer is still an AI evaluator. Independent HENRI acceptance must use external outcomes and execution artifacts, not reviewer ratings.

## State, owner, gate, artifact

This table is a workflow instruction, not a new JSON schema or scheduler.

| Phase | Owner | Gate before advance | Compact artifact |
|---|---|---|---|
| RESEARCH | research + ontology | resolved source and observed baseline limit | source/hash/page refs; at most two initial hypotheses |
| AUDIT | architecture | definition → caller → consumer; shapes/device/causal/gradient checks | affected-path map and missing checks |
| DESIGN | architecture + research | mechanism, alternative cause, budget, kill, matched controls | frozen SpecContract A + preregistration |
| APPROVAL | integration + human | required scope explicitly approved | exact human decision and ledger hash |
| IMPLEMENT | architecture | one bounded change; defaults protected; test written first | changed-path manifest, HarnessContract B, commit |
| REMOTE VERIFY | integration | valid preflight and exact-SHA CUDA/CI execution | command, RC, run ID, stdout/stderr and receipt refs |
| MEASURE | telemetry + rigor | evaluator/split/memory/budget match; mechanism engagement | paired external score, ablation, cost/latency, uncertainty |
| INFER | integration | four integrity checks and pre-registered rule | accept/reject/blocked; feedback C; sealed decision |

Infrastructure failure is BLOCKED, not mechanism rejection. Rejection retains parent and records the failed family. Do not tune to sealed test answers. Use a development set for revisions; reserve final evaluation for the decision. A held-out reviewer is not a substitute for held-out task evaluation.

## Default budgets and stopping

These are HENRI policy defaults, not paper-derived optimums or runtime-enforced caps:

- Two initial hypotheses, one bounded implementation at a time.
- At most two genuine targeted repair attempts before STRACE attribution.
- One ablation refinement and one review-driven revision per approved cycle.
- At most three explicit MoA escalations/session and one/STRACE cycle. Already-selected MoA fanout is separate; inspect actual call counts.
- Literature grounding: existing CLI limits 3 strategies, 3 primitives, 90 seconds; at most 25 results.
- Leaf summaries target ≤500 tokens; root decision packet target ≤2000 tokens. Verify actual length where possible; the prompt is not a hard cap.
- GPU/time/spend cap, seed list, acceptance margin, confidence rule, and evaluator are registered per experiment. No unbounded refinement or unapproved full-scale launch.

Stop on a failed prerequisite, absent causal consumer, invalid control, insufficient evidence, or spent budget. Do not increase internal coherence objectives to explain zero external score.

## Ontology grounding path

1. Resolve the current task against existing typed records and vault decisions.
2. Read original Drive or vault source bytes. API path: file ID, revision/modified time, export MIME, hash. Mounted-file path: exact path/hash, cloud revision/sync unverified. Do not read `.gdoc` as document text.
3. MCP server_info checks auth. One focused notebook_query requests citations. source_get_content resolves the supporting IDs. Record version/page/excerpt/hash when available.
4. Distinguish an approved Aletheia specification from a claim that its code exists. A spec can bind the intended design; it cannot prove implementation.
5. Commit existing-schema Terms/Mappings/Constraints/Evidence only with probe_ref and valid locators. Keep contradictions and supersedes. No gold benchmark answers or latent tensors enter the semantic layer.
6. Pass only relevant resolved records and gaps. Reconsult on changed evidence, not mechanically at every tool step.

NotebookLM primary bank: `ca4bb787-de9d-4ee0-89c9-bf71259cc86d`. Ontology store: `C:/Users/chan/henri-telemetry/ontology/`. Resolve Obsidian via `OBSIDIAN_VAULT_PATH`. Do not conflate engineering ontology, execution dependency graph, and Zone C latent graph.

## Verified host entry points

Resolve HERMES_HOME from the active profile. This task targets the default profile only. Git Bash needs native `C:/...` paths for native Python/git programs.

```bash
python -m hermes_cli.main moa list
python -m hermes_cli.main bundles show henri-bundle
python "$HERMES_HOME/scripts/henri_audit.py" verify
python "$HERMES_HOME/scripts/henri_audit.py" record henri-arbiter HUMAN_DECISION '<json>'
python "$HERMES_HOME/skills/henri-workflow/henri-agent-integration/scripts/validate_holonic_contracts.py"
python "$HERMES_HOME/scripts/nlm_run.py" login --check
```

The NotebookLM console shim uses the separate NotebookLM profile store. After login, MCP refresh_auth reloads tokens; verify with a source read. MCP upgrade or plugin/config changes are a separate between-session operation.

Autoresearch, from the selected HENRI V2 checkout that actually contains it:

```bash
python -m agentic_graph.autoresearch_cli --probe
python -m agentic_graph.autoresearch_cli --query "<question>" --limit 8 --out "<outside-git-dir>" --receipts "<outside-git-dir>/receipts.jsonl"
```

Use installed STRACE scripts, not the untracked vendor runner:

```bash
python "<strace-skill-dir>/scripts/trace_reducer.py" --corpus "<real-corpus>" --out "<outside-git-run-dir>" --top-k 5 --exemplars 5
python "<strace-skill-dir>/scripts/causal_slice.py" --summaries "<run-dir>/trace_summaries.json" --edg "<verified-edg.json>" --trace "<key>" --manifestation "<node>" --out "<run-dir>/slice.json"
```

A dependency slice is a suspect set. Causal attribution needs an intervention. Unknown trace mappings keep severity unknown and must not be silently dropped.

## Governance and GitHub sync

Verify the audit chain before and after a serial append; read back exact hash/payload. Events include source ingestion, human scope decision, patch, push, remote verdict, and acceptance/rejection. Never overwrite earlier observations with later outcomes. Link exact artifact hashes and commit/run IDs.

The current ledger writer is hash-linked but unsigned, locally rewritable, and lacks a demonstrated concurrent-writer lock. The chain does not prove source truth or signer identity. These limits are not repaired by SOUL editing; concurrent append failure blocks downstream promotion.

Use an isolated worktree, explicit file allowlist, diff/seal checks, commit, and review-branch push. Compare local HEAD with git ls-remote for that exact branch. Read GitHub checks and artifacts for the exact SHA. Do not force-push, auto-merge, or blanket-stage. A review branch is synced only to its own remote ref. Production release requires local/GitHub/tested-remote SHA equality and approval.

Default HENRI path: commit → push → henri-ci → CUDA evidence → telemetry delivery. If CI does not support the branch, record BLOCKED. Do not merge unverified code just to start the main-only collector. Manual remote runs require an explicit queue/approval.

## Live limits observed in this audit (2026-10-01)

- Config resolves three references and a DeepSeek aggregator. The acting parent session uses a different selected model; no parent MoA execution is inferred from config.
- MoA references lack tools and HENRI system prompt. Local `_reference_messages` includes flattened tool previews while public docs describe a narrower view. Supply essentials in the task packet.
- Main lacks autoresearch modules present on carrier/e6-physical-verifier. Their consumers are examples/tests, not a verified production scheduler. GraphRuntime allows classified → dispatched; the claimed compulsory grounding path is not enforced.
- Graph task/result/state/receipt schemas are not A/B/C. Static source inspection finds `_delta` emits `state`, absent from the strict state-delta schema. Carrier promotion also adds fields absent from that schema. Treat schema compliance as BLOCKED pending real remote validation/repair; no HENRI tests ran locally for this instruction task.
- GraphRuntime does not append governance automatically. Use the existing ledger through the acting agent; do not claim an automatic transition-to-ledger link.
- henri-ci is local no-agent cron → henri_ci_runner.py → henri_ci.sh. It follows main and has inconsistent stored endpoint strings. Cron ok is not run success. Remote endpoint and targeted suite need fresh verification before execution.
- GitHub workflows on origin/main include docker-publish and egress-manifest. They are not a general agentic-graph CUDA gate.
- STRACE installed tools exist but are not an automatic telemetry consumer. Vendor STRACE and SHACL folders are not evidence of runtime integration.
- Drive Desktop mount is readable; separate Drive API OAuth is absent. NotebookLM reauth and cited-source read succeeded. Drive cloud revision and delivery remain unverified.
- The existing ontology objects.jsonl contains mixed historical record formats, including records without canonical record_id/kind/probe_ref fields. New records follow the existing documented schema; whole-store schema compliance remains BLOCKED. This task does not migrate or overwrite history.

These limits require separate code/infrastructure designs. Do not hide them with new policy text.

## Acceptance of this instruction change

Pass only if the real Hermes SOUL loader returns untruncated/unblocked content, the real bundle builder loads the exact current fixed bundle membership with none missing, A/B/C and plugin mirrors remain identical, every retained installed reference resolves, the audit chain verifies, and published overlay bytes match the installed files.

This establishes instruction loading and provenance, not better MoA output or HENRI scores. Cheapest benefit test: replay the same resolved research tasks through old/new stacks at fixed model/data/tool budgets; independently grade unsupported claims, missed gates, source-to-consumer errors, wall time, and actual token/cost fields. Use genuine archived tasks, no fabricated traces. Pre-register acceptance and keep the parent if gains do not reproduce.
