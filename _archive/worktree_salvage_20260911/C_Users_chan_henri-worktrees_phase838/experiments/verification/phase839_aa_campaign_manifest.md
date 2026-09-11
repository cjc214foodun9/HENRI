# Phase 8.39 — Artificial Analysis v4.1 Adapter Campaign Manifest (DRAFT)
Status: BLOCKED — PRIMARY_SOURCE_NOT_VERIFIED (all entries)
Date: 2026-08-20
Source index: BenchLM benchmarks.json (371 entries, sha256 pending)

## Composite weight map (PDF HENRI-CAPABILITY-AA-INDEX-2026-08)
- Agents 34%: GDPval-AA v2 (20%), τ³-Banking (14%)
- Coding 24%: Terminal-Bench v2.1 (16%), SciCode (8%)
- Scientific Reasoning 24%: HLE (12%), GPQA Diamond (6%), CritPt (6%)
- General/Long-Context 18%: AA-Omniscience (12%), AA-LCR (6%)

## Required per-benchmark adapter sequence
1. canonical primary source (URL + immutable digest)
2. dataset sha256 + license + official split + official metric
3. task-specific evaluator (id + version + sha256)
4. staged items under data/official_benchmarks/staged_eval_suites/
5. runner → ADAPTER_READY → EVALUATED (registry promotion gate)

## Registry entries (from BenchLM index, all adapter_status=BLOCKED)
| benchmark_id | display_name | family | metadata_source |
|---|---|---|---|
| gdpvalaa | GDPval-AA | artificial_analysis_v41 | https://benchlm.ai/benchmarks/gdpvalaa |
| aatau3banking | AA Tau3 Banking | artificial_analysis_v41 | https://benchlm.ai/benchmarks/aatau3banking |
| aaterminalbench21 | AA Terminal-Bench 2.1 | artificial_analysis_v41 | https://benchlm.ai/benchmarks/aaterminalbench21 |
| terminal-bench-hard | Terminal-Bench Hard | artificial_analysis_v41 | https://benchlm.ai/benchmarks/terminal-bench-hard |
| aascicode | AA-SciCode | artificial_analysis_v41 | https://benchlm.ai/benchmarks/aascicode |
| aahle | AA-HLE | artificial_analysis_v41 | https://benchlm.ai/benchmarks/aahle |
| aagpqadiamond | AA-GPQA Diamond | artificial_analysis_v41 | https://benchlm.ai/benchmarks/aagpqadiamond |
| critpt | CritPt | artificial_analysis_v41 | https://benchlm.ai/benchmarks/critpt |
| aaomniscienceindex | AA-Omniscience Index | artificial_analysis_v41 | https://benchlm.ai/benchmarks/aaomniscienceindex |
| lcr | AA-LCR | artificial_analysis_v41 | https://benchlm.ai/benchmarks/lcr |
| aaifbench | AA-IFBench | artificial_analysis_v41 | https://benchlm.ai/benchmarks/aaifbench |
| aammmupro | AA-MMMU-Pro | artificial_analysis_v41 | https://benchlm.ai/benchmarks/aammmupro |
| ifeval | IFEval Official | artificial_analysis_v41 | https://benchlm.ai/benchmarks/ifeval |
| tau2-bench | τ²-Telecom | artificial_analysis_v41 | https://benchlm.ai/benchmarks/tau2-bench |
