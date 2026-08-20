# Phase 3 Verdict — Batched Mean-Phase-Cosine Kernel (RTX 5090)

Spec: **HENRI-PHASE3-BATCHED-GATE-2026** (roadmap M4: step latency ≤ 2.0 ms; failure threshold > 5.0 ms)
Status: **COMPONENT PASS (E1/L1/M1) — ROADMAP M4 FALSIFIED** — pre-registered E1/L1/M1 gate passed on NVIDIA RTX 5090, D=65,536; absolute M4 latency target falsified (17.9 ms > 5.0 ms failure threshold); fused Triton/CUDA C++ kernel INCOMPLETE.

## 1. Profiler-first measurement (OBSERVED, remote RTX 5090)

`arc_phase839_phase3_latency_probe.py` — production chain reuse (WaveASTDecoder._instantiate, ASTDiscriminativeEncoder IDF codebook N=100, SecurePythonSandbox):

| Item | Candidates | kept | gen_ms | rank_ms | sandbox_ms | total_ms | rank_frac |
|---|---|---|---|---|---|---|---|
| HumanEval/0 | 172 | 172 | 0.1 | 865.8 | 32.4 | 898.2 | 0.964 |
| HumanEval/1 | 71 | 71 | 0.1 | 347.5 | 32.4 | 379.9 | 0.915 |
| HumanEval/2 | 71 | 71 | 0.1 | 346.3 | 31.6 | 378.0 | 0.916 |

SUMMARY: wave_phase 1,559.9 ms (0.942 frac), sandbox_phase 96.4 ms (0.058), avg 552.0 ms/item.
**Gate: MEASURE_BOUNDED_PORT** — the qFHRR rank loop is the measured hot operator (94.2% of item latency), NOT the sandbox.

Audit trail: the first two probe runs (candidates=0, kept=0) were rejected as VACUOUS (empty pool from inline regex vs production `parse_signature`); only the run with the production parser counts.

## 2. Implementation (default-OFF)

- `qfhrr_ast_discriminative_kernel.py`: `batched_mean_phase_cosine(candidates, codebook, codebook_chunk=8)` — chunked [C, chunk, D] intermediates, NO [C, N, D] tensor; reference-equivalent float32 mean phase-cosine.
- `humaneval_wave_ast_runner.py`: `--ast-idf-batched` (default OFF, requires `--ast-idf-only`), batched branch preserves the exact ranking semantics (score = mean over codebook of mean phase-cosine; same tie-break by candidate index; unparseable candidates score -1e9).
- Scorecard telemetry: `ast_idf_batched` field recorded.
- Commits: `d753196` (kernel + wiring + gate), verified at `448926c`.

## 3. Gate evidence (OBSERVED, remote RTX 5090)

`arc_phase839_phase3_batched_gate.py` — real 172-candidate pool (HumanEval/0) × 100-codebook, D=65,536:

- **E1 (equivalence):** max |batched − reference| = 2.98e-08 ≤ 1e-3 → PASS (local d=2048 smoke 1.49e-08, 40×12)
- **L1 (latency):** p50 batched 17.9 ms vs reference 667.8 ms → 37.4× speedup → PASS
- **M1 (memory):** peak 344 MB (C×8×D×4) < naive 4.2 GB ([C,N,D]); no dense [C,N,D] → PASS

## 4. Roadmap M4 — FALSIFIED (explicit)

The roadmap's absolute step-latency target (≤ 2.0 ms, failure threshold > 5.0 ms) is **FALSIFIED**: the measured batched path runs 17.9 ms at D=65,536 — above the failure threshold. The fused Triton/CUDA C++ kernel required to approach M4 is **INCOMPLETE** (current implementation is chunked torch-CUDA behind default-OFF `--ast-idf-batched`).

This is a component-throughput result, not an intelligence-score claim: Gate B (sealed, `2bdef68`) showed grammar expressiveness — NOT latency — is the current HumanEval accuracy bottleneck (2/50, oracle items only).

## 5. Evidence classes

- `OBSERVED`: gate metrics (remote CUDA, this run), profiler breakdown (remote CUDA, corrected probe)
- `DERIVED`: speedup ratio, memory bounds
- `FALSIFIED`: roadmap M4 absolute latency target (17.9 ms > 5.0 ms failure threshold; target 2.0 ms)
- `BLOCKED / INCOMPLETE`: fused Triton/CUDA C++ kernel (not yet implemented; current = chunked torch-CUDA)
