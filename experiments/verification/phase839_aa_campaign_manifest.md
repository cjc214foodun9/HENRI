# Phase 8.39 — AA v4.1 adapter campaign manifest (registry)

Lifecycle: UNSTAGED → ADAPTER_READY → EVALUATED → BLOCKED / EXCLUDED
Rule: a catalog (BenchLM) is index metadata only, never evidence. Promote
external performance only at gate 5 (canonical source + digest + evaluator +
item outcomes + RunEvidence). Status 2026-08-20.

| benchmark | source | status | digest / notes |
|---|---|---|---|
| HumanEval | openai/human-eval (official) | EVALUATED — 2/50 PASS | wave-AST egress; verdict `arc_phase839_humaneval_verdict.md`; event `8edb7753` |
| GPQA Diamond | `openaipublic.../simple-evals/gpqa_diamond.csv` | EVALUATED — FALSIFIED 0.298 (gate 0.30) | 198 items, SHA `41d1213c…`; verdict `arc_phase839_gpqa_verdict.md`; event `866bf08d` |
| MMLU | `openaipublic.../simple-evals/mmlu.csv` | ADAPTER_READY (staged) | 14,042 rows, SHA `15b6785d…`; cols Question/A–D/Answer/Subject |
| HLE | HF `cais/hle` | BLOCKED — 401 | canonical source unresolved |
| MMMU-Pro | HF `MMMU/MMMU_Pro` | BLOCKED — 401 | canonical source unresolved |
| MMLU-Pro | simple-evals `mmlu_pro.csv` | BLOCKED — 404 | guess path; canonical path unresolved |
| IFEval Official | google-research guess | BLOCKED — 404 | canonical path unresolved |

## Verified source facts (OBSERVED)
- `gpqa_diamond.csv`: HTTP 200, 1,373,492 B, 198 rows, SHA `41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305` (local = remote).
- `mmlu.csv`: HTTP 200, 6,667,575 B, 14,042 rows, SHA `15b6785d49e0012602e089558a7a0dfb916baf97e9295aa25b48062f13c6afbb`.
- GPQA run: remote RTX 5090, D=65,536, n=198, 22.6 s, scorecard `gpqa_839_scorecard.json`.

## Kill evidence (GPQA, pre-registered)
- Gate: accuracy >= chance + 0.05 (0.30). Observed 0.2980 → FALSIFIED.
- Geometry control: distinct-option cosine 0.4529 vs random 0.0039 → codec not
  orthogonal; correct>wrong gap +0.0168 = weak lexical signal below gate.
- Inference: zero-demo raw-cosine MCQ ranking is not task-competent. Next:
  calibrated head or in-context demo pairs (grammar+ranking pattern from wave-AST).
