# Artificial Analysis Intelligence Index v4.1.1 — Official-Byte Composition & HENRI Capability-Gap Matrix

**Date:** 2026-08-24 · **Reference 3 (gpt-5.6-sol) binding** · Evidence: OBSERVED_PRIMARY_BYTES

## Pinned official bytes (distinct files — overwrite pitfall avoided)

| Page | URL | HTTP | Bytes | SHA-256 (full) | UTC |
|---|---|---|---|---|---|
| Methodology | `https://artificialanalysis.ai/methodology` | 200 | 387,164 | `211ae8a812e65521d17f6bff9fbf405f974395efc4555103ff73995e04142b1c` | 2026-08-24T21:13:48Z |
| Intelligence benchmarking | `https://artificialanalysis.ai/methodology/intelligence-benchmarking` | 200 | 687,653 | `a4d1161fd2e097de277182610df8d04636812120dca7de09e51b5a2c7d9bf68f` | 2026-08-24T21:13:48Z |
| Capability indices | `https://artificialanalysis.ai/methodology/capability-indices` | 200 | 391,841 | `26f8023125519edc03da2071180c13873fc9905054c33247e5153be5ffcd9c1c` | 2026-08-24T21:13:48Z |
| Homepage (label anchor) | `https://artificialanalysis.ai/` | 200 | 1,790,335 | `52b130a12bbc31bec16ee183…` | earlier cycle |

Exact version string from pinned bytes: **"Artificial Analysis Intelligence Index v4.1.1"** (v4.0 strings also present for retired content).

## Composition table (extracted deterministically from pinned table HTML)

**Category weights:** Agents 34% · Coding 24% · Scientific Reasoning 24% · General 18% (weighted average; "weighting emphasizes agentic tasks").

| # | Evaluation | Category | Questions | Repeats | Response type | Scoring | Index weight | Tool |
|---|---|---|---|---|---|---|---|---|
| 1 | GDPval-AA v2 | Agents | 220 tasks | 1 | Agentic task completion, file outputs | Pairwise Elo, judge panel (3 frontier LLMs), Bradley-Terry MLE, anchored to human experts at 1000, frozen & scaled | 20% | ✓ |
| 2 | 𝜏³-Banking | Agents | 97 | 5 | Dual control agent-user simulation + knowledge retrieval | Backend database state evaluation, pass@1 | 14% | ✓ |
| 3 | Terminal-Bench v2.1 | Coding | 89 | 3 | Terminal-based task execution | Test suite pass/fail, pass@1 | 16% | ✗ |
| 4 | SciCode | Coding | 288 subproblems (test set) | 3 | Python code (all unit tests) | Code execution, pass@1, sub-problem scoring | 8% | ✗ |
| 5 | AA-LCR | General | 100 | 3 | Open answer | Equality Checker LLM, pass@1 | 6% | ✗ |
| 6 | AA-Omniscience | General | 6,000 | 1 | Open answer | Accuracy 8% + (1 − Hallucination Rate) 4% as separate components | 12% | ✗ |
| 7 | HLE (Humanity's Last Exam) | Scientific Reasoning | 2,158 | 1 | Open answer | Equality Checker LLM, pass@1 | 12% | ✗ |
| 8 | GPQA Diamond | Scientific Reasoning | 198 | 5 | Multiple choice (4 options) | Regex extraction, pass@1 | 6% | ✗ |
| 9 | CritPt | Scientific Reasoning | 70 | 5 | Python functions, symbolic, numeric | Official grading server, pass@1 | 6% | ✗ |

Index = weighted average across the four categories; 95% CI < ±1% claimed (based on >10 repeats for index datasets; individual evals may be wider).

## Excluded / legacy / additional (from pinned bytes)

- **IFBench** — removed from the Index in v4.1; still run standalone (294 questions, 5 repeats, pass@1).
- **Additional evaluations (reported separately, not in Index):** AA-Briefcase, Harvey LAB-AA, APEX-Agents-AA, AutomationBench-AA, AA-AnalystAgent, ITBench-AA, EnterpriseOps-Gym-AA, MLCR-AA, Global-MMLU-Lite, MMMU Pro.
- **Legacy (retired/superseded):** Terminal-Bench Hard, 𝜏²-Bench, Telecom, MATH-500, AIME 2025, MMLU-Pro, LiveCodeBench (list per pinned sidebar).

## Reproducibility audit (BLOCKED items)

| Item | Status |
|---|---|
| GDPval-AA v2 judge panel (3 frontier LLMs, blind pairwise) | `BLOCKED` — judge model identities and grading transcripts not public |
| CritPt official grading server | `BLOCKED` — task bytes/execution not fully public |
| ITBench-AA private scenarios | `BLOCKED` — 59 scenarios "public + private" |
| EnterpriseOps-Gym-AA oracle servers | `BLOCKED` — resettable enterprise gyms require harness access |
| Scoring formulas (Elo anchor 1000, pass@1, pass^5, sub-problem scoring) | `OBSERVED` from methodology bytes; per-benchmark dataset versions partially public |

No AA component is currently reproducible end-to-end from public bytes; official published model scores are the only `OBSERVED` composite values.

## HENRI capability-gap matrix (v4.1.1 components vs live path)

| Evaluation | Required capability | Live HENRI path today | Gap disposition |
|---|---|---|---|
| GDPval-AA v2 | Long-horizon agentic file production, tool loop, judge-comparable output quality | System-1: bounded 13-family DSL only; VLA gate 0/12 | `BLOCKED` — needs tool-agent loop + calibrated egress (Stage-3/4) |
| 𝜏³-Banking | Dual-control agent-user simulation, knowledge retrieval, backend DB state | None | `BLOCKED` |
| Terminal-Bench v2.1 | Terminal/tool execution, long-context state | None | `BLOCKED` |
| SciCode | Python codegen + unit tests | System-1 AST egress covers tiny DSL, not open Python | `BLOCKED` — needs open-language codegen (Stage-3+) |
| AA-LCR | Long-context reasoning | No long-context path | `BLOCKED` |
| AA-Omniscience | Broad open-answer accuracy + low hallucination | No open QA path | `BLOCKED` |
| HLE | Deep knowledge + exact equality checking | None | `BLOCKED` |
| GPQA Diamond | MCQ extraction | None | `BLOCKED` |
| CritPt | Official grading server integration | None | `BLOCKED` |

**Conclusion:** 0/9 AA components have a live capability path. Stage-0b (frozen CartPole representation) addresses NONE of them. No CartPole work may be described as AA-relevant progress unless a causal evaluation path exists (per Reference 3).

## Next falsification

Any claim of AA v4.1.1 progress must cite (a) the pinned composition bytes above and (b) the specific live capability path for the component. Stage map for VLA: 0a dynamics ✓ → 0b frozen representation ✓ → 0c learned dynamics → 1 perception → 2 temporal world state → 3 action/policy → 4 real environment loop → 5 memory-to-action continual adaptation. Global VLA gate: **0/12**.
