# Phase 8.39 — AA v4.1 adapter campaign manifest (registry)

Lifecycle: UNSTAGED → ADAPTER_READY → EVALUATED → BLOCKED / EXCLUDED
Rule: a catalog (BenchLM) is index metadata only, never evidence. Promote
external performance only at gate 5 (canonical source + digest + evaluator +
item outcomes + RunEvidence). Status 2026-08-20.

| benchmark | source | status | digest / notes |
|---|---|---|---|
| HumanEval | openai/human-eval (official) | EVALUATED — **2/50 PASS** (baseline; control re-verified `0066e9a`) | wave-AST egress; verdict `arc_phase839_humaneval_verdict.md`; event `8edb7753` |
| HumanEval grammar-extension rerun | same dataset | EVALUATED — **NO GAIN 2/50** (4661 candidates; ranking inert, not coverage) | verdict `arc_phase839_humaneval_grammar_rerun.md`; commit `171c21c` |
| HumanEval reward-rank | same dataset | EVALUATED — **FALSIFIED 1/50** (control 2/50; reordered 25 items) | verdict `arc_phase839_reward_rank_verdict.md`; event `a7b93863`; commit `3654b60` |
| HumanEval decoder-rank | same dataset | EVALUATED — **FALSIFIED 0/50** (control 2/50; oracle ranks 49/71 + 68/71) | verdict `arc_phase839_decoder_rank_verdict.md`; event `124a47b6`; commit `2b048d2` |
| HumanEval spec-rank (V1 doc Stage 1) | same dataset | EVALUATED — **FALSIFIED 1/50** (control 2/50; docstring targets 50/50; ranking-lever class closed) | verdict `arc_phase839_spec_rank_verdict.md`; event `43d04e6a`; commit `3078c90` |
| HumanEval trained-head (V3, MBPP-execution probe) | same dataset | EVALUATED — **FALSIFIED Gate A** (/35 rank 15>12; val_acc 0.602; Gate B not run) | verdict `arc_phase839_trained_head_verdict.md`; event `8cfa4b81`; commit `2abf70f` |
| Zone C lexical partition (V2 doc Stage 2) | — | **BLOCKED_MISSING_PREMISE** (no non-leaky labeled corpus) | verdict `arc_phase839_zonec_lexical_verdict.md`; commit `d9624e2` |
| Class 2.0 AST codec + MBPP codebook (Levers 2.1/2.2) | HumanEval oracle (proxy d=2048) | EVALUATED — **FALSIFIED at proxy Gate A** (ranks 39/71 + 29/71 > 5; carrier cos 0.59; Gate B skipped) | verdict `arc_phase839_class2_ast_codec_verdict.md`; event `20e4f7c3`; commit `4d2ab55` |
| Class 3.0 discriminative kernel (carrier-subtract + ast-idf) | HumanEval oracle (proxy d=2048) | EVALUATED — **FALSIFIED at proxy Gate A** (both: cos 0.2526 > 0.10; ranks 10/12 > 5; idf-only ranks 3/5 recorded-not-promoted; Gate B skipped) | verdict `arc_phase839_class3_discriminative_verdict.md`; event `1a2a69af`; commit `5f7c3fe` |
| GPQA Diamond | `openaipublic.../simple-evals/gpqa_diamond.csv` | EVALUATED — FALSIFIED 0.298 (gate 0.30) | 198 items, SHA `41d1213c…`; verdict `arc_phase839_gpqa_verdict.md`; event `866bf08d` |
| MMLU | `openaipublic.../simple-evals/mmlu.csv` | EVALUATED — **FALSIFIED 0.2598** (gate 0.30; chance+0.98%) | 14,042 rows, SHA `15b6785d…`; verdict `arc_phase839_mmlu_verdict.md`; event `93298071`; commit `4e3f84b` |
| Codec repair (5 position modes) | MMLU 200-slice @ D=65,536 | EVALUATED — **FALSIFIED** (no mode ≥ 0.31; best `full` 0.295) | verdict `arc_phase839_codec_repair_verdict.md`; commit `47dc9e5`; event `93298071` |
| HLE | HF `cais/hle` | BLOCKED — 401 | canonical source unresolved |
| MMMU-Pro | HF `MMMU/MMMU_Pro` | BLOCKED — 401 | canonical source unresolved |
| MMLU-Pro | simple-evals `mmlu_pro.csv` | BLOCKED — 404 | guess path; canonical path unresolved |
| IFEval Official | google-research guess | BLOCKED — 404 | canonical path unresolved |
| simple-evals mirror (other) | math_500/aime/medqa/gpqa_main/simpleqa/brag/… | BLOCKED — 404 (all probed 2026-08-20) | mirror hosts exactly 2 verified datasets: gpqa_diamond.csv + mmlu.csv |

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
