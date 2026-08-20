# Phase 8.39-V2 — Zone C lexical partition verdict

Status: **BLOCKED_MISSING_PREMISE (premise kill; not executed)**
Branch: `phase839/humaneval-wave-ast` @ `17fec17`
Date: 2026-08-20

## User-approved lever (option 3, sequential)
Doc Stage 2: partition Zone C into `zone_c_action_engrams` + `zone_c_lexical_engrams`; ingest structured-codec phrase embeddings (GPQA/MMLU text) for lexical retrieval.

## Premise kill (audited, not assumed)
The partition's value premise: lexical engrams retrieved at test time must rank MCQ options above the sealed 0.295–0.298 codec ceiling. That requires a legitimate seed corpus with an answer signal. Probed options:

| Seed source | Status | Notes |
|---|---|---|
| MMLU/GPQA test items | FORBIDDEN | benchmark leakage; zero-pretrain invariant |
| HF train splits (cais/mmlu, hendrycks/test) | 401/404 | probed 2026-08-20, no verified path |
| simple-evals mirror | 2 datasets only | gpqa_diamond.csv + mmlu.csv (test sets, forbidden) |
| General-knowledge text (Wikipedia etc.) | No answer signal | retrieval cannot rank A–D options |
| Zone C existing engrams | MISMATCH | action waves (`m0r0:ACTION2`…); Lens A.2 confirmed |

Without a canonical, non-leaky, answer-bearing lexical corpus, a lexical partition is an empty store with no causal consumer. Creating the DDL (prod policy requires approval; CHECKPOINT/VACUUM only standing) without a premise is exactly the mock-loop pattern the arbiter rejects.

## Disposition
- Phase 2 **NOT executed**; recorded `BLOCKED_MISSING_PREMISE`.
- Re-open gate: a verified non-leaky lexical corpus with labels becomes available (e.g. canonical MMLU train split via an authenticated HF token, or an approved general-knowledge corpus with a task-appropriate retrieval objective).
- Prod Zone C DDL remains untouched (no CHECKPOINT/VACUUM bypass).

## Evidence
- Zone C probe (OBSERVED 2026-08-20): tables `zone_c_engrams`, `phylogenetic_engrams_65536`, `zone_c_engrams_hourly`; columns (time, axiom_id, domain_tag, phase_vector, sagnac_stress) — no lexical namespace.
- Doc Lens A.2 (action-vs-lexical category mismatch) — CONFIRMED against live store.
