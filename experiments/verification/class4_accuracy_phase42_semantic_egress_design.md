# Class 4.2 Semantic Egress Design Boundary

Status: BLOCKED_PENDING_SEMANTIC_SUPERVISION
Base: `origin/main@849c65d0175699d1acc271c62b0d08974714f40c`
Accuracy branch: `accuracy/fidelity-remediation`
Phase 4.1 verdict: `FALSIFIED_NO_EXTERNAL_GAIN` (`673f6ffc`)

## Root-cause update

Phase 4.1 increased the candidate set from 4,661 to 4,716 on the same 50-item
HumanEval slice. Expressibility was already `50/50`, infrastructure errors were
zero, and the external result remained `2/50`. Candidate count is not the
current bottleneck.

The live default rank signal is built from
`qFHRREpistemicCodec.encode_text()`, which creates an independent random ring for
each string. It can identify the same string but cannot map a prompt or program
body to a compositional semantic relation. The result is a near-random ordering
of an otherwise valid grammar pool.

## Mathematical and implementation hypothesis

`HYPOTHESIS`: a task-grounded egress requires a representation with a causal
relation between the task specification and candidate program semantics. A
random ring, a generic code AST wave, or a post-hoc score rescale cannot supply
that relation by itself.

For an online zero-pretraining path, the relation must come from authorized
in-context pairs or externally verified execution trajectories. For HumanEval,
the live prompt contains a specification and tests, but the tests are evaluator
inputs and cannot be used as generation labels. The current runner has no
authorized demonstration pair for the hidden target.

## Required causal path

```text
authorized task specification or demo pairs
→ typed program IR / AST representation
→ compositional task relation
→ candidate program construction
→ official sandbox execution
→ external item outcome
```

The candidate generator must not read the evaluator test or reference answer.
The evaluator must remain the only source of PASS/FAIL outcome.

## Blockers

1. No authorized `(observation, GameAction, data)` trajectory bank exists for
   ARC semantic action-head calibration.
2. HumanEval has no live demonstration-pair interface in the current runner.
3. The existing checkpoint is a legacy linear unbinder and is not a proven
   program synthesizer.
4. The structured character-position codec has a prior joint task gate
   falsification and cannot be promoted by geometry alone.
5. Ranking-lever experiments are sealed closed. Reopening them without a new
   semantic representation would repeat the same invalid hypothesis.

## Cheapest valid experiment

Before production wiring, construct an authorized paired transformation fixture
with explicit provenance and no evaluator leakage. Test:

- exact task relation recovery on held-out pairs;
- candidate rank `<=2`;
- margin `>=0.05`;
- no dense `[D,D]` allocation;
- external sandbox replay on held-out programs.

If the relation does not change candidate ranking and produce a new external
pass, kill the mechanism. Internal cosine, entropy, or checkpoint load cannot
accept it.

## Decision

Do not implement another ranking flag or another grammar-template batch. The
next implementation requires one of:

- authorized task demonstration/trajectory data compatible with the zero-
  pretraining boundary; or
- an explicitly approved semantic program model and provenance package.

Until one exists, benchmark SOTA claims are `BLOCKED`, and ARC score eligibility
remains `ACTION_HEAD_NOT_CALIBRATED` / `BLOCKED_NO_ACTION_TRAJECTORIES`.
