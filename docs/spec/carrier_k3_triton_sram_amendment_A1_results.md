# Carrier K3 — A1 In-SRAM Fused Triton Solve: Sealed Results

Document Identifier: `HENRI-SPEC-2026-09-V3-CARRIER-K3-A1-SEALED-RESULTS`
Instrument: `HENRI-SPEC-2026-09-V3-CARRIER-K3-AMENDMENT-A1` (commit `8f220ba`)
Branch: `feat/carrier-k3-empirical-koopman`. Local only; no push.
Date: 2026-09-03. Remote: vast-5090 (`/workspace/henri-k3-dispatch` @ `ddc950f`, torch 2.12.0+cu130, triton 3.7.0).

## 1. Verdict

**`K3_GATE_KG5_LATENCY_FAILED` (A1 arm) — the in-SRAM Triton Cholesky remedy is FALSIFIED at measurement on this geometry.**

| Gate | Bound | Result | Evidence |
|---|---|---|---|
| A1-EQ | rel err ≤ 1e-4 | **PASS** — max 2.92e-7, mean 1.18e-7, finite, shape (8192,8,8) | probe `a1run7.txt`, seed 20260903, n=256, M=8192, d=8, alpha=1e-4 |
| A1-ENG | fired-batch engagement | PASS at fixture level (random data is contractive → fired 0, correct); planted-expansive fixture is the contract-test proof | contract suite (CPU, local) |
| A1-KG5 | score-path CUDA-event mean ≤ 2.00 ms | **FAIL** — fused kernel median 18.85 ms (min 18.52, max 19.80, 15 reps) vs torch accum+solve pair 0.768 ms (24.5× slower) | probe `a1run7.txt` |
| A1-DFLT | flag-absent byte identity | not executed as a differential on the engine (see §3 deviation); module import + flag-absent RuntimeError covered by local tests | — |

## 2. What was measured (OBSERVED)

The meta-generated straight-line Triton kernel (v4, file-backed via temp module; v1–v3 compile failures recorded: BN guard, nested `def` → StopIteration, list comprehension → NotImplementedError, exec-source → "must be defined in a Python file", missing `pid`) is NUMERICALLY CORRECT — equivalence to the sealed torch cholesky path at 3e-7 — but 24× SLOWER: 18.85 ms vs 0.768 ms for einsum accum + batched `torch.cholesky_solve` (cuSOLVER/cuBLAS).

Reading: for this geometry (M=8192 independent 8×8 systems, n ≤ 256), a per-block in-register Triton solve cannot beat cuBLAS/cuSOLVER batched primitives. The A1 remedy — even fully tiled — has no headroom to the 2.0 ms bound; the bound is dominated elsewhere (torch spectral screen 0.65 ms + G7 base affordance ~1.0 ms, both untouched by any solve swap — disclosed in amendment §5). Deficit #1's "in-SRAM Triton Cholesky" premise is closed as FALSIFIED_AT_MEASUREMENT on RTX 5090.

## 3. Disclosed deviation

Amendment §6 called for the full sealed engine run (12 envs × 150 steps). It was NOT launched: the component measurement (18.85 ms fit, the dominant term of the score path) makes the engine `k3_kernel_latency_ms` mean ≥ ~20 ms — a foregone fail. Per fail-closed resource discipline the run was withheld and the component measurement recorded as the A1-KG5 gate evidence. Amendment §3.2 wiring into `BlockRidgeKoopmanFit.fit` was also withheld as moot: K3 is sealed-FALSIFIED (no re-dispatch without a new instrument + qualified bank), and wiring a measured 24× slower path adds dead weight. The kernel module remains committed, flag-gated (`HENRI_K3_TRITON_SOLVE=1`), as the artifact of record for any future retry with a different strategy.

## 4. Artifacts

- Kernel: `HENRI V2/experiments/verification/arc_k3_triton_sram_solve.py` (canonical-LF SHA recorded at commit).
- Remote probe: `/tmp/a1/a1_remote_probe.py`, `/tmp/a1/arc_k3_triton_sram_solve.py` (SHA `89f7fd09…` → latest `227a5e1e…` → final at commit); evidence `a1run7.txt`.
- Local tests: `HENRI V2/tests/contract/test_arc_k3_triton_sram.py` (CPU: import without triton, flag-absent RuntimeError, emitted-source AST validity).

## 5. Chain status

Fail-closed chain: **32 falsifications, 0 solved**. K3 verdict (`591b526`) unchanged and NOT reopened. CAP12 bank remains `BLOCKED_ENTROPY_GATE` (`c44a00c`). Next honest steps (each REQUIRES_APPROVAL): (a) deficit #1 alternate remedy targeting the measured dominant terms (spectral screen 0.65 ms and/or base affordance ~1.0 ms) — NOT a solve swap; (b) diversity-directed re-capture for a qualified 12-env basis; (c) any full K3 re-dispatch only after (b).
