# Historical instruction snapshot

This is the previous entry file. It is not current policy. Use current SKILL.md for routing. Revalidate every dated runtime claim.

---
name: henri-strace-optimizer
description: "Use when HENRI failures repeat or R&D spins. STRACE loop."
category: henri-workflow
---

# henri-strace-optimizer — Trace Root-Cause Optimization Loop

Port of STRACE (arXiv:2607.07702, Microsoft Research; official repo
https://github.com/moomight/STRACE — repo has no LICENSE file, so this skill
paraphrases concepts and cites sources; it does not copy text).

Purpose: attribute repeated HENRI failures to the ROOT-CAUSE module and inject
generalized heuristics into that module's instructions only. STRACE treats
execution records as causal evidence, not linear text. Do not rewrite code.
Do not optimize every failed trace. Do not patch the manifestation node
without first running the attribution stage.

## When to use

- A failure repeats at the same module (two consecutive runs, same error class).
- R&D loops spin: multiple patches applied without task-score change.
- A benchmark gate fails and the cause may be upstream of the failing step.
- An experiment requires a pre-registered A/B before policy change.

Do NOT use: when no execution records exist, or when the goal is new-agent
construction from scratch.

## Stage 0 — Triage (always run first)

| Signal | Action |
|---|---|
| `output/dependency_prior.json` exists and topology unchanged | Skip Stage 1 |
| Only one trace or user names the failing module | Skip Stages 1-2, run Stage 3 |
| Attribution already exists for the trace | Skip Stage 3 |
| User asks only for a prompt/skill fix with attribution in hand | Skip Stages 1-3, run Stage 4 |
| Two consecutive runs failed at the same node | MANDATORY Stages 2-3 before the next patch |

State the plan in one line, then execute it. Do not enumerate options.

## Pipeline (file handoffs, no raw traces upward)

```text
Stage 1  EDG refresh      -> output/dependency_prior.json   (skip if unchanged)
Stage 2  Trace reduction  -> scripts/trace_reducer.py        (deterministic)
                            output/trace_summaries.json
                            output/severity.json
                            output/patterns.json
                            output/selection.json
Stage 3  Causal slice     -> scripts/causal_slice.py         (deterministic)
                            output/slices/<trace>.json
                            then LLM attribution per slice   -> output/attributions/<trace>.json
Stage 4  Policy synthesis -> output/proposals/<module>.md    (isolated context)
                            approval gate BEFORE any skill/contract edit
```

Default output root: `C:\Users\chan\henri-telemetry\strace_output\<run_id>\`.
Never write stage outputs into the Git repository.

## Stage 1 — Structural Modeling (EDG)

- Canonical prior: `references/henri-edg.json` (+ human view
  `references/henri-edg.md`). Mirror the bundle topology: holons, MoA roster,
  cron executors, carriers, stores.
- Refresh ONLY when topology changed (contracts, bundle members, cron jobs,
  carriers). Refresh = regenerate both files together, run
  `validate_holonic_contracts.py` afterward if contracts were touched.
- A stale-but-unchanged prior beats a fresh LLM inference: keep the file.

## Stage 2 — Failure Pattern Mining and Trace Filtering

1. Deterministic pass (NO LLM on raw traces):
   ```bash
   python scripts/trace_reducer.py --corpus <dir> --out <run_dir> [--taxonomy references/taxonomy-harness.json] [--top-k 5] [--exemplars 5]
   ```
2. Reducer emits schema fidelity, per-component severity
   P(task fail | component error), structural patterns
   (self-loop, oscillation, dead-end), and an exemplar set.
3. LLM reads ONLY `selection.json` + `severity.json` summaries. If the corpus
   is huge, the exemplar set is the optimizer input; never the full corpus.

Schema inference is heuristic. Report which keys matched. If fidelity is low
(no node/outcome fields), treat severity as UNKNOWN-class and say so.

## Stage 3 — Causal Localization

1. Deterministic backward slice:
   ```bash
   python scripts/causal_slice.py --summaries output/trace_summaries.json --edg references/henri-edg.json --trace <trace_key> --manifestation <node_id>
   ```
2. Slice retains positions whose node can reach the manifestation through
   EDG dependency closure. Pruned positions are temporal antecedents only.
3. If no position node maps to the EDG, the script keeps ALL positions and
   sets `fallback: true`. Do not delete evidence silently.
4. LLM then reads the slice positions (targeted reads, never the full trace)
   and writes `output/attributions/<trace>.json`:
   `{trace, manifestation_node, root_cause_node, causal_chain, evidence_positions, confidence}`.
5. Aggregated attributions re-map manifestation nodes to root-cause nodes
   (STRACE Figure 4 case: 12 of 25 traces re-mapped upstream). If a root-cause
   node differs from the manifest node, the policy target is the root cause.

## Stage 4 — Inductive Policy Optimization

- Input: root-cause node + its aggregated slices ONLY. Isolated context
  (STRACE Appendix B.1: Phase 4 runs in its own window; phases 1-3 share).
- Synthesize <=3 generalized heuristics per module. De-contextualize:
  convert trace instances into reusable rules. Prefer rules of the types
  validated in the paper Appendix D: failure-aware stopping conditions,
  action-selection boundaries, multi-round planning strategy.
- Output: `output/proposals/<module>.md` with exact target file/line for the
  future patch, expected effect, and a falsifiable A/B criterion.
- Approval gate: a proposal touching any live SKILL.md, contract, or skill
  reference requires human approval. Proposal application = one bounded
  patch, then remote verify, then measure. Failed A/B = revert + record
  `FALSIFIED`; a negative is a governance win, not a loop iteration.

## Orchestration modulation rules (workflow change)

1. Attribution-first escalation: repeated same-node failure triggers Stages
   2-3 BEFORE the next code patch. Blind retry is prohibited.
2. Targeted policy injection: only the root-cause module receives a new
   instruction. Never a global prompt rewrite.
3. Evidence classes travel with every artifact:
   OBSERVED / DERIVED / INFERRED / HYPOTHESIS / FALSIFIED / BLOCKED.
4. Storage separation: `strace_output/` is a derived analysis layer over
   telemetry. It is NOT a new memory store, NOT Zone C, NOT the audit chain.
   Link artifacts with run_id + source hash.
5. MoA usage: max 1 MoA call per STRACE cycle, only for Stage 3 attribution
   disagreement or Stage 4 synthesis after two failed solo attempts.
6. Token budget per stage (hard caps):

| Stage | Input to LLM | Cap |
|---|---|---|
| 1 | EDG refresh prompt | 2K tokens |
| 2 | selection+severity summaries | 4K tokens |
| 3 | slice positions per trace | 6K tokens per trace |
| 4 | root node + slices | 8K tokens |

Raw traces never enter LLM context; positions and hashes do.

## Kill criteria (pre-registered)

- Stage 2 severity shows no component with P(task fail | error) > 0.5 and
  count >= 3 across two runs -> optimizer has no signal; stop and audit the
  trace schema instead of synthesizing policy.
- A Stage 4 heuristic that fails its A/B on the held-out set is reverted.
  A second failed A/B for the same module kills the heuristic family.
- If attribution confidence stays low (< 0.5) for 5 traces, the EDG prior is
  stale: refresh Stage 1 before further Stage 4 work.

## Links

- Paper: https://arxiv.org/abs/2607.07702 ; HTML: https://arxiv.org/html/2607.07702v1
- Official repo: https://github.com/moomight/STRACE (skill + prompts + scripts)
- EDG prior: `references/henri-edg.md`, `references/henri-edg.json`
- Stage instructions: `references/stage1-env-modeling.md` ... `references/stage4-policy-synthesis.md`
- Deterministic tools: `scripts/trace_reducer.py`, `scripts/causal_slice.py`
- Example taxonomy: `references/taxonomy-harness.json`
- Ontology evolution binding: STRACE Stage-2 products (`patterns.json`,
  `severity.json`) are the `diagnose` input of the ontology evolution loop, and
  Stage-4 proposals are Tool-level patches subject to the paired gate —
  `henri-ontology/references/evolution-loop.md`.
- Literature evidence for a policy proposal: `agentic_graph/autoresearch_cli.py`
  (orx, no login) → receipts as ontology Evidence records.
