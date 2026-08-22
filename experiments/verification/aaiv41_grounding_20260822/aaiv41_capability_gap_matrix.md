# AAII v4.1 — Live Capability-Gap Matrix (2026-08-22)

Evidence classes: OBSERVED (measured/verified live), BLOCKED (required evidence unavailable), FALSIFIED (contradicted by valid test), INFERRED (reasoned, not directly measured).

## Benchmark-by-benchmark

| Benchmark | Canonical source | Evaluator | Modality | Current HENRI path | Proven score | Principal blocker |
|---|---|---|---|---|---|---|
| GDPval-AA v2 | BLOCKED (non-public) | AA judge Elo | agentic tool use + files | none | none | no tool-agent loop; no semantic backbone |
| 𝜏³-Banking | tau2-bench (OBSERVED) / tau3 path BLOCKED | backend DB state pass@1 | agent tool use, ~195K-token KB | none | none | no tool-agent loop; no long-context readout |
| Terminal-Bench v2.1 | BLOCKED_VERIFY (repo path) | test suite pass@1 | terminal exec | none | none | no terminal agent; no backbone |
| SciCode | scicode-bench/SciCode (OBSERVED) | unit tests pass@1 | Python code gen | grammar enumeration + wave ranking | HumanEval proxy 2/50 (FALSIFIED class) | synthesis/ranking capacity (MBPP 17/500 ceiling) |
| AA-LCR | BLOCKED (non-public) | LLM equality checker | long-context open answer | none | none | no long-context readout; no backbone |
| AA-Omniscience | BLOCKED (non-public) | accuracy + hallucination rate | open answer | raw wave cosine MCQ (MMLU proxy) | MMLU 25.98% ≈ chance (FALSIFIED) | no semantic knowledge; no calibrated readout |
| HLE | cais/hle (OBSERVED) | LLM equality checker | open answer | none | none | knowledge + reasoning capacity |
| GPQA Diamond | simple-evals gpqa_diamond.csv (prior OBSERVED sha `41d1213c…`; raw path now 404 → BLOCKED_VERIFY) | regex pass@1 | MCQ | raw wave cosine | 29.8% < 0.30 gate (FALSIFIED) | weak lexical geometry (correct>wrong gap +0.0168) |
| CritPt | BLOCKED (official grading server) | official grader | code/symbolic | none | none | reasoning; external grader |

## Cross-cutting capability audit (OBSERVED 2026-08-22)

1. **No pretrained semantic backbone anywhere in the codebase.** Grep of `HENRI V2` for `transformers`/`torchvision`/`vit`/`clip`/`siglip`/`llama`/`qwen`/`deepseek` imports: **zero hits** (excluding tests/_archive/exploratory). The only encoders are HENRI's random UWE/phase encoders (no learned semantics).
2. **Environment asymmetry:** local Windows has `transformers`/`PIL`/`accelerate` (CPU torch); **Vast 5090 lacks `transformers` and `accelerate`** (has hf_hub, PIL, torch 2.12.0+cu130, CUDA true) → backbone inference on the CUDA target requires provisioning (pip install, ~20 GB disk).
3. **No calibrated action/tool agent loop.** Live ARC action policy = `orch.plan_action → EFEPlanner.select_action` (EFE argmin/T4). SagnacMCTSPlanner only under `HENRI_ARC_TARGET_GROUNDING`/RT-MCTS (FALSIFIED_AB_ON_GATE_S). No tool-use harness exists for agentic evals.
4. **No trained VLM / vision semantics.** `HENRIVisionEncoder` = random phase UWE (deterministic but unlearned); MMMU-Pro / any image-bearing eval → BLOCKED.
5. **No long-context support** (AA-LCR ~100×3 repeats needs long-document reasoning; τ³-Banking ~195K-token KB).
6. **Egress: no LLM-judge-compatible open-answer generation.** HENRI egress (grammar/sandbox, Hopfield snap, wave-AST) has no calibrated free-text decoder; open-answer evals (HLE, AA-LCR, AA-Omniscience) → BLOCKED.
7. **Score-eligibility discipline is intact:** `score_eligible` stays False without LOADED calibrated checkpoint; synthetic markers → diagnostic-only. This must be preserved for any backbone path.

## Architectural inference (INFERRED, corpus-supported)

The gap classes (1), (3), (4), (5), (6) cannot be closed by EDMD/Zone C/Sagnac tuning — measured codec geometry and ranking levers are FALSIFIED as semantic-composition sources (run19/20, 8.39 class levers). The credible route is: **provenance-audited pretrained multimodal backbone (semantic System-1) + HENRI as memory/planning/retrieval/verification/online-adaptation layers (System-2/3)**. Corpus consult (bank `ca4bb787`) supports this with a strict functional division of labor preserving the zero-benchmark-pretraining invariant for benchmark-specific data (INFERRED).
