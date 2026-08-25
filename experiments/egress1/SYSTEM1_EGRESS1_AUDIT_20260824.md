# System-1 Egress-1 — Frozen Backbone → AST Egress → CEGIS (2026-08-24)

**Verdict: COST_EFFECTIVE** · Approval `2b30c69f…` · Prereg `75e4f911…` · Outcome `1bc01c10…`

## Carrier
`frozen Qwen3-VL-8B-Instruct (0c351dd0) semantic backbone → structural AST egress (13-rule DSL) → CEGIS first-pass admission → external pass@1` on a fresh single-use 520-task heldout.

## Execution summary (all OBSERVED)
| Item | Evidence |
|---|---|
| Approval | `2b30c69f…` (parent `50c2528e…`) |
| run18 baseline audit | commit `9b165ad` (canonical MBPP 17/500, checkpoint `75572389…`, split consumed — historical context ONLY, never comparator) |
| Consult #30 | INFERRED: reranking is decorative; conditioning must change structural support / first verified rank |
| Contract | FINAL sha `156b705c…` (corrections pre-seal: skeleton expansion infeasible — 4–9 candidate saturation; family-variant expansion infeasible — gen_task bodies fixed per family; FINAL reorder-only) |
| Split 1 | heldout54_egress1 `529e5ddc…` — QUARANTINED (arms executed; aggregation crash lost evidence) |
| Split 2 (replacement) | heldout55_egress1 `ec2e1cfd…` — 520 = 13×40, seed 82126, single_use, generation-only seal |
| Plumbing | 13/13 families canonical candidates pass verifier+outcome, tokenizer closed, pools nonempty |
| Engagement | identity arm OK; 360/520 pools reordered; sim spread 0.058; first-rank mean 3.23→3.00; rank-0 80→120 |
| Remote | Vast 45864: transformers 5.15.1, CUDA 2.12, Qwen shard-verified LOADED; hashes 12/12 local==remote |
| Run 2 (fresh) | 520/520 tasks, per_task.jsonl 520 rows, aggregation complete |

## Verdict math (paired, 520 tasks)
- Outcome: A 1.0 = B 1.0 (ceiling-blocked DSL); delta 0.0; McNemar p = 1.0 (b=c=0)
- AST validity (admitted): 1.0 / 1.0; family min support: 1.0 / 1.0
- Verifier calls: A 2200 (mean 4.231) vs B 2080 (mean 4.000) → −120 calls (−5.45%); paired diff CI90 [0.156, 0.308], **lb > 0**
- First verified rank: mean 3.231 → 3.000; rank-0 80 → 120; reordered pools 360/520
- Conditioning engaged + changed first verified rank → NOT a decorative reranker

## Verdict chain (as pre-registered)
Pass@1 improvement is ceiling-blocked (1.0/1.0 baseline) → CAPABILITY_PROMOTED vacuous. COST_EFFECTIVE requires exact outcome preservation + call-reduction CI lb > 0 → **MET**.

## Boundaries
No backbone fine-tuning (zero trainable backbone params; `freeze_for_baseline` + requires_grad audit PASS). No CartPole coupling. No AAII v4.1.1 claim. No VLA claim. Split is MBPP-style DERIVED (generated 13-family DSL), not official MBPP. any_pass@K structurally identical (reorder-only; same pool content — grammar cardinality bound).

## Artifacts
- Contract `egress1_contract.md` (`156b705c…`, prior `d59e1a0a…`, orig `7fcc9361…` — all recorded)
- Split `egress1_split2/heldout55_egress1.json` + `.seal.json`
- Results `egress1_results_remote.json` (`be4395b9…`) + `egress1_per_task_remote.jsonl` (`e9c06a60…`)
- Evaluator `egress1_eval.py` (`71c0e73d…`); backbone adapter `henri_backbone_adapter.py` + `embed_text` (`e8bf6c53…`)

## Next (require decision)
- (A) Hold at COST_EFFECTIVE (Egress-1 promoted as cost-effective; capability unchanged).
- (B) Egress-2: richer 13-family grammar (add argument-shape variants) to break the 4–9 saturation → then conditioning can change structural support, enabling CAPABILITY_PROMOTED test.
- (C) Move to AAII-relevant carrier (Terminal-Bench/SciCode family) with the same backbone-conditioning pattern.
- (D) Roll the pattern into the canonical MBPP path with a canonical fresh split (needs official-partition audit).
