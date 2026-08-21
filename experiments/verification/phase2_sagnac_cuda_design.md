# Phase 2 Fused CUDA Sagnac Candidate-Scoring Kernel

Status: COMPONENT PASS; M4 ABSOLUTE TARGET FALSIFIED; CURRENT-STREAM REPAIR VERIFIED
Packet: `Project_HENRI_Class_4.0_Universal_VLA_Master_Plan.md`
Packet SHA-256: `1b4effe266015f3698ca8e48d00aa4ad08d920b51c04a81952f5072d8d3d5555`
Base: `849c65d0175699d1acc271c62b0d08974714f40c`
Approval event: `CLASS4_MASTER_PLAN_INTAKE` `79bf2cd7-d0da-40cd-a524-5da3ff2817e2`

## Academic foundation

The current M4 path scores each candidate against an IDF codebook with

`score(c) = mean_n mean_d cos(2*pi*(c[d]-cb[n,d])/256)`.

The live representation is a `uint8` phase ring. The proposed kernel uses a
256-entry cosine lookup table and performs the reduction without materializing a
three-dimensional `[C,N,D]` tensor. This is an implementation hypothesis, not a
claim of task intelligence or universal VLA capability.

Known limits:

- The existing M4 measurement is the AST candidate-scoring path. It is not proof
  that the advisory `SagnacMCTSPlanner` controls ARC action selection.
- A latency reduction can improve throughput only. It does not change the
  candidate set or grammar expressiveness.
- CUDA compilation and a forward pass are execution evidence, not benchmark
  accuracy evidence.

## Technical decision

### Mechanism

Add `HENRI V2/cuda/sagnac_mcts_cuda_core.cu` and a Python loader
`HENRI V2/sagnac_mcts_cuda.py`. The loader is enabled only when
`HENRI_SAGNAC_CUDA=1`. The default path remains the existing PyTorch chunked
implementation.

The CUDA op is registered as `torch.ops.henri.sagnac_mcts(candidates, codebook)`.
It accepts contiguous CUDA `uint8` tensors with shapes `[C,D]` and `[N,D]`, and
returns `[C]` float32 scores. Each candidate block reduces all codebook rows and
phase dimensions using the LUT. No `[C,N,D]` tensor is allocated.

### Causal data path

`humaneval_wave_ast_runner.py`
→ `batched_mean_phase_cosine()`
→ `sagnac_mcts_cuda.batched_mean_phase_cosine_cuda()` when the named flag is ON
→ `torch.ops.henri.sagnac_mcts`
→ candidate ordering.

The planner class is not silently relabeled as the consumer. The current live
M4 consumer is the AST candidate scorer.

### Resource limits

- Input dimension: `D=65,536`.
- Candidate/codebook tensors: `uint8`, contiguous, CUDA.
- No dense `[C,N,D]` intermediate.
- Gate peak allocation: `<512 MiB` for the measured candidate/codebook sizes.
- Timing requires `torch.cuda.synchronize()` before and after each measured call.
- The kernel launches on PyTorch's current CUDA stream and validates that both
  inputs use the same CUDA device.
- One CUDA process owns the GPU during the remote gate.

### Expected benefit

`HYPOTHESIS`: direct LUT reduction will reduce launch and intermediate-tensor
cost relative to the eager chunked PyTorch path.

### Failure modes

- Build failure or missing CUDA toolchain: `BLOCKED_INFRASTRUCTURE`.
- Shape, dtype, device, or registration mismatch: typed runtime failure.
- Output error above tolerance: `FALSIFIED` and flag remains OFF.
- No strict p50 latency improvement: `FALSIFIED` for M4; do not promote.
- Kernel speedup without full production-caller execution: component result only.
- Kernel speedup with unchanged HumanEval outcomes: throughput result only;
  grammar expressiveness remains the task bottleneck.

### Cheapest kill experiment

Run deterministic `[C=172,N=100,D=65,536]` `uint8` inputs through the exact
production scoring function twice: flag OFF and flag ON. Require synchronized
max absolute error `<=1e-3`, p50 latency improvement, and peak allocation below
512 MiB. Stop before any benchmark campaign if one criterion fails.

## Pre-registered acceptance and rejection

| Gate | Acceptance | Rejection |
|---|---:|---:|
| Equivalence | max abs error `<=1e-3` | `>1e-3` |
| Latency | fused p50 `<` eager p50 | fused p50 `>=` eager p50 |
| Memory | peak `<512 MiB`; no `[C,N,D]` | limit exceeded or dense intermediate |
| Production symbol | exact `batched_mean_phase_cosine` route exercised | standalone kernel only |
| Task outcome | not asserted by this phase | any score claim is rejected |

A release-level M4 PASS requires the remote CUDA gate and the existing Phase 3
receipt to record the exact candidate SHA, command, implementation arm, output
comparison, synchronized timings, and GPU identity. The flag remains default-OFF
until that receipt is reviewed.
