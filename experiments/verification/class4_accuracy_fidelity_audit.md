# Class 4 Accuracy and Fidelity Audit

Status: AUDIT COMPLETE; IMPLEMENTATION REQUIRES APPROVAL
Base: `origin/main@849c65d0175699d1acc271c62b0d08974714f40c`
Worktree: `C:/Users/chan/henri-worktrees/accuracy-main`
Latency branch frozen at verified code SHA: `191459f91b233dd7816feb3a24f8b61dbd7fbd5c`

## User directive

Abandon latency reduction. Prioritize benchmark accuracy and fidelity. Do not
promote the fused CUDA scorer or change the latency branch further.

## Evidence chain

### OBSERVED: live HumanEval path

`humaneval_wave_ast_runner.run_benchmark()` loads official HumanEval items,
creates `WaveASTDecoder`, enumerates bounded source-string templates, and runs
the candidate sources against official test code in `SecurePythonSandbox`.
The path records `checkpoint_used=false`. It is not a neural token-generation
path.

The live call chain is:

```text
official HumanEval prompt + tests
→ qFHRREpistemicCodec
→ WaveASTDecoder._wave / _instantiate
→ bounded grammar candidates
→ optional ranking sidecars
→ SecurePythonSandbox
→ item result
```

### OBSERVED: legacy representation is non-compositional

At reduced dimension `D=2048`, the live `qFHRREpistemicCodec` produced:

| Pair | Phase cosine |
|---|---:|
| `a+b` vs `b+a` | `0.011587` |
| `27` vs `28` | `-0.006406` |
| `f(3,3,3)` vs `f(4,4,4)` | `-0.004043` |
| identical string | `1.000000` |

The implementation is `encode_text()` in
`HENRI V2/zone_c_epistemic_axiom_harness.py:58-63`: SHA-256 seed followed by
independent random `uint8` phase draws. It preserves identity but not token,
position, AST, or semantic continuity.

### OBSERVED: candidate expressiveness is the external bottleneck

`WaveASTDecoder._instantiate()` in `HENRI V2/wave_ast_decoder.py:181-236`
contains a fixed template grammar. It has no general program generator and
cannot represent arbitrary Python control flow, library use, or data-structure
composition. The sealed HumanEval result is `2/50`; the ranking-lever class is
closed. The current external failure is therefore not a CUDA throughput issue.

### OBSERVED: ARC action path is not calibrated

`arc_score_gate.py` records that the live ARC action producer uses
`HolographicActionDecoder` in `darwinian_phase_swarm.py`. The generic neural
egress checkpoint is not the action-producing path. `trained_action_head_active`
and score eligibility remain blocked until authorized trajectory provenance,
held-out semantic validation, legal-action ordering, and coordinate payload
validation exist.

### FALSIFIED / REJECTED candidate

`qfhrr_structured_codec.py` provides character-position continuity, but its prior
joint Gate B result was FALSIFIED at scale. It must not be wired into production
as a presumed fix. A new codec requires a fresh causal comparison and a task
outcome gate.

### BLOCKED research source

NotebookLM bank `ca4bb787-de9d-4ee0-89c9-bf71259cc86d` authentication is stale.
The bank was not used as evidence for this audit. Live code and measured probes
override its historical claims.

## Accuracy-first design: Phase 4.1 semantic program egress

### Mechanism

Build a typed, compositional program-synthesis boundary for the code benchmark
path. Generate Python AST candidates from a versioned primitive grammar and
compose them by input arity and inferred value class. Keep wave representations
as a diagnostic ranking signal only until a task-valid signal is demonstrated.

The first implementation slice is a **binary collection transformation** tracer
bullet:

```text
(list-like a, list-like b)
→ zip/map/filter/set/sort/aggregate AST templates
→ compile candidate source
→ official sandbox tests
→ record PASS/FAIL/EXECUTION_ERROR
```

This slice changes candidate expressiveness, not candidate ranking and not CUDA
latency. It must reuse the production `SecurePythonSandbox` and the exact
HumanEval runner evaluator.

### Hypothesis

`HYPOTHESIS`: expanding the grammar with typed binary collection composition
will increase the fraction of official code tasks for which a correct program
is expressible, and will produce new external passes without relying on a
random-ring similarity signal.

### Data path

```text
HumanEval prompt signature
→ typed grammar builder
→ Python AST/source candidates
→ fail-closed syntax check
→ SecurePythonSandbox with official tests
→ item-level external outcome
```

No evaluation test code may enter candidate generation. No reference-answer
lookup, string matching, pseudo-demo, or generated benchmark item is allowed.

### Resource limits

- Candidate cap remains bounded by `attempts`.
- No dense `[D,D]` tensors.
- No Zone C writes.
- No test-time parameter adaptation in the first control comparison.
- No latency claim.
- One CUDA run owns the GPU only when the remote benchmark gate is executed.

### Expected benefit

`TARGET_GOAL`: more semantically valid candidate programs reach the sandbox;
external pass count increases above the sealed `2/50` baseline.

### Failure modes

- Grammar candidates increase but no new sandbox passes: representation or
  prompt-to-program inference remains insufficient.
- Candidate count increases with no expressibility gain: grammar expansion is
  redundant and is rejected.
- Execution errors increase: typed grammar or sandbox assembly is defective.
- Ranking changes but candidate set does not: classify as a closed ranking lever.
- Any use of official test code or reference answers during generation:
  `BLOCKED_LEAKAGE`.

### Cheapest kill experiment

Before broad implementation, write a red contract test showing that the new
binary collection tracer generates a valid AST for a representative pairwise
transformation and that the existing grammar does not. Then run a bounded
held-out slice through the production sandbox. Kill the slice if it causes
execution errors, leakage, or no increase in distinct executable semantic forms.

### Acceptance and rejection criteria

| Gate | Accept | Reject |
|---|---:|---:|
| AST validity | all generated candidates parse | any malformed template path |
| Sandbox integrity | infrastructure errors remain zero | any launcher error counted as FAIL |
| Expressibility | new semantic forms exist in candidate set | only reorder/duplicate existing forms |
| External result | at least one new pass on the pre-registered slice | no new pass after bounded run |
| Leakage | no test/reference data in generator | any task-label or answer use |
| Ranking independence | candidate set changes before ranking | only score/order changes |

A positive result does not establish SOTA. It only permits the next bounded
grammar tracer bullet.

## Required approval

The design changes a load-bearing accuracy path. Do not implement it until the
user approves this bounded Phase 4.1 slice. ARC remains separately blocked by
missing authorized action trajectories and semantic action-head calibration.
