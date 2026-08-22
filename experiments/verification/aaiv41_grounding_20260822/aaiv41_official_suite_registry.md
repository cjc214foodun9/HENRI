# AAII (Artificial Analysis Intelligence Index) — Official Suite Registry (OBSERVED 2026-08-22)

**Scope:** current official Intelligence Index definition as served by artificialanalysis.ai on 2026-08-22.
**Evidence:** pinned primary bytes (below); deterministic extraction script `extract_aaiv41_facts.py`; extracted facts `aaiv41_extracted_facts.json`. HTML files kept local-only (repo hygiene); hashes recorded.

## Pinned sources (OBSERVED)

| URL | bytes | sha256 (prefix) |
|---|---|---|
| https://artificialanalysis.ai/methodology | 386,942 | `577a1de761092d92` |
| https://artificialanalysis.ai/ | 1,778,846 | `2e18dd8eb1775c15` |
| https://artificialanalysis.ai/blog/intelligence-index | 17,067 | `6c29aa0e8477256f` |
| https://artificialanalysis.ai/methodology/capability-indices | 391,969 | `fc6a7d30419e108e` |
| https://artificialanalysis.ai/methodology/intelligence-benchmarking | 687,781 | `db0c7fff753c0582` |

## Versioning note (OBSERVED / INFERRED)

No literal "v4.1" version string appears in the fetched bytes. The **current official index composition** (below) is what AA serves today. The local 14-benchmark manifest (`phase839_aa_campaign_manifest.md`, BenchLM-era) is **STALE**: it lists MMLU, HumanEval, IFEval, AIME, MATH-500, LiveCodeBench, Terminal-Bench, MMMU-Pro, τ²-Telecom, τ³-Banking, AA-LCR, AA-Omniscience, HLE, GPQA Diamond. The official page marks MATH-500, AIME 2025, LiveCodeBench, Terminal-Bench Hard, τ²-Bench Telecom as **Legacy Evaluations**. The current index uses Terminal-Bench **v2.1** and SciCode (not HumanEval). Archived local runners (`HENRI V2/_archive/invalid_evaluators/run_artificial_analysis_*.py`) remain **quarantined** — they are not official-suite evidence.

## Current Intelligence Index suite (OBSERVED, extracted from intelligence-benchmarking.html)

| Category (weight) | Evaluation | Questions | Repeats | Response type | Scoring | Index weight | Tools |
|---|---|---|---|---|---:|---|---|
| Agents (34%) | GDPval-AA v2 | 220 tasks | 1 | Agentic task completion w/ file outputs | Pairwise Elo by judge panel, anchored to human experts at 1000, frozen & scaled | 20% | ✓ |
| Agents (34%) | 𝜏³-Banking | 97 | 5 | Dual control agent-user simulation w/ KB retrieval | Backend DB state evaluation, pass@1 | 14% | ✓ |
| Coding (24%) | Terminal-Bench v2.1 | 89 | 3 | Terminal-based task execution | Test suite pass/fail, pass@1 | 16% | ✗ |
| Coding (24%) | SciCode | 288 subproblems (test set) | 3 | Python code (must pass all unit tests) | Code execution, pass@1, sub-problem scoring, scientist-annotated background prompting | 8% | ✗ |
| General (18%) | AA-LCR | 100 | 3 | Open answer | Equality Checker LLM, pass@1 | 6% | ✗ |
| General (18%) | AA-Omniscience | 6,000 | 1 | Open answer | Accuracy (8%) and Hallucination Rate (4%) as separate components | 12% | ✗ |
| Scientific Reasoning (24%) | HLE | 2,158 | 1 | Open answer | Equality Checker LLM, pass@1 | 12% | ✗ |
| Scientific Reasoning (24%) | GPQA Diamond | 198 | 5 | MCQ (4 options) | Regex extraction, pass@1 | 6% | ✗ |
| Scientific Reasoning (24%) | CritPt | 70 | 5 | Python functions, symbolic expressions, numerical answers | Official grading server, pass@1 | 6% | ✗ |

**Index structure facts (OBSERVED):**
- "Every component benchmark is run independently by Artificial Analysis before its scores are combined into the index."
- Indices are skill-based (equal-weighted averages) or industry-based (O*NET-style task weights). Industry indices (Finance, Strategy & Ops, Legal, Healthcare, Engineering) are separate from the Intelligence Index.
- Category weighting emphasizes agentic tasks: Agents 34%, Coding 24%, General 18%, Scientific Reasoning 24%.

## Additional evaluations (NOT in the Intelligence Index; OBSERVED)

Global-MMLU-Lite (~6,000, ~400/language, 16 languages, MCQ 4-option, regex pass@1), MMMU Pro (1,730, MCQ 10-option), IFBench (294, open answer, rule-driven), MLCR-AA (60, LLM judge panel), plus agent extras: AA-Briefcase, Harvey LAB-AA, APEX-Agents-AA, AutomationBench-AA, AA-AnalystAgent, ITBench-AA, EnterpriseOps-Gym-AA.

## Canonical dataset source probes (OBSERVED/BLOCKED, 2026-08-22)

| Evaluation | Canonical source | Probe | Status |
|---|---|---|---|
| SciCode | github.com/scicode-bench/SciCode | raw README 200 | OBSERVED |
| HLE | huggingface.co/datasets/cais/hle | README 200 | OBSERVED |
| 𝜏²-Bench / 𝜏³-Banking | github.com/sierra-research/tau2-bench (repos tau-bench, tau2-bench, mu-bench exist) | API 200 | OBSERVED (exact tau3 path BLOCKED) |
| Terminal-Bench v2.1 | sierra-research org (raw README 404) | 404 | BLOCKED_VERIFY (branch/path unknown) |
| GPQA Diamond | openai/simple-evals (default_branch main; raw gpqa_diamond.csv 404 now — earlier 2026-08-20 fetch recorded sha `41d1213c…`, 198 rows) | 404/prior-OBSERVED | BLOCKED_VERIFY path |
| GDPval-AA v2, AA-LCR, AA-Omniscience, CritPt | not public / unresolved | — | BLOCKED (canonical source unresolved) |

## Governance consequence (INFERRED → binding)

Local reproduction can approximate index components whose datasets are public (SciCode, GPQA Diamond, HLE, Terminal-Bench). GDPval-AA, AA-LCR, AA-Omniscience, CritPt are AA-run or non-public → local scores for those are **BLOCKED**; only AA-published scores are OBSERVED, and only if an official submission path exists (P5 gate, unresolved).
