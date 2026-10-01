# HENRI execution dependency graph

Updated for the approved five-core bundle. This is a structural prior, not a live execution graph. Nodes/edges in henri-edg.json guide slicing; production edges remain hypotheses until matched to live callers and run records. Role nodes are skills, not autonomous services.

- root_holon: henri-agent-integration: owns approvals, Vast dispatch, MoA gating
- research_holon: henri-research: emits SpecContract A
- architecture_holon: henri-architecture: emits HarnessContract B
- strace_holon: henri-strace-optimizer: root-cause attribution + policy proposals
- moa_aggregator: acting model resolved from live MoA preset; only tool-bearing MoA node
- moa_ref_a: tool-less code/consumer advisor; exact model from live preset
- moa_ref_b: tool-less mechanism/system advisor; exact model from live preset
- moa_ref_c: tool-less independent falsification advisor; does not see current A/B outputs
- cron_ci: henri-ci runner every 3m
- cron_ingest: henri-research-ingest every 10m
- cron_watchdog: notebooklm auth watchdog daily
- codec_encoder: historical codec/representation node; verify actual representation and caller
- edmd: R-EDMD online dynamics learner
- planner: historical planning node; verify action-policy caller; do not assume SagnacMCTS selects actions
- egress_head: hopfield egress + AST decode; fail-closed veto
- constraints: constraint penalties; candidate-specific, dim-normalized
- carrier_c1: sealed SO8 rotor carrier (ed7b603); dispatch-driver-only
- carrier_k3: sealed KG5 TZCSM carrier; remote run needs approval
- carrier_g8: sealed gauge carrier; grade_scramble 0.0
- benchmark_harness: MBPP / ARC-AGI-3 / gauntlet runners
- telemetry_store: G:\My Drive\HENRI_Telemetry JSONL/JSON artifacts
- audit_chain: active-profile audit/henri_audit_chain.jsonl; hash-linked unsigned governance
- vault: Obsidian local vault projections
- trustgraph: optional holonic context layer (service must be verified)
- zone_c: latent artifacts and CUDA telemetry only
- ontology_holon: typed evidence/mapping source manifest; current five-core member
- notebooklm: cited corpus synthesis; resolved source reads required
- drive_source: original source bytes/revisions; mounted and API evidence distinct

MoA references execute in parallel without tools or cross-reference outputs. NotebookLM, Drive, vault, typed ontology, audit chain, and Zone C have distinct evidence meanings. Unknown trace mappings require fallback and coverage reporting.
