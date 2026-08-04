# HENRI Digital-Twin Convergence Roadmap

This document tracks the bounded path from the current software model to a
falsifiable digital twin of the proposed phase-processing architecture. It is
a development plan, not evidence of physical equivalence or task capability.

## Evidence classes

- `OBSERVED`: returned by a real source, repository inspection, or execution.
- `DERIVED`: calculated from observed evidence by a stated rule.
- `INFERRED`: reasoned from observed evidence; not directly measured.
- `HYPOTHESIS`: proposed mechanism awaiting a test.
- `FALSIFIED`: contradicted by a valid test.
- `BLOCKED`: required evidence, dependency, or approval is unavailable.

## Current baseline

- `OBSERVED`: the production planner boundary uses real `[K, 8]` waves.
- `OBSERVED`: qFHRR persistence uses integer phase-ring operations in `Z_256`.
- `OBSERVED`: the projective Hopfield implementation is experiment-only.
- `OBSERVED`: the weakness selector is default-off and has no verified live
  continuation-set provider.
- `OBSERVED`: production-shaped decoder execution requires an exact checkpoint;
  missing `policy=required` artifacts block that path.
- `INFERRED`: representation ambiguity is a higher-value first target than
  vector Sagnac wiring or kernel work.

## Stage sequence

### Stage 0 — evidence baseline

Status: `OBSERVED / COMPLETE`.

The Gap Analysis PDF was located, hashed, ingested, and audited. The primary
repository dirty boundary remains preserved. The source does not establish
projective capacity, GUE convergence, LLG dynamics, 20 kHz end-to-end timing,
or universal cognition.

### Stage 1 — canonical phase-state contract

Status: `IMPLEMENTATION APPROVED / IN PROGRESS`.

Artifact: `phase_codec_adapter.py` and its contract suite.

This stage defines explicit `PhaseRingState` and `ComplexPhaseState` objects,
shape/layout/dtype/device/normalization/quantization/provenance metadata, exact
modular bind/unbind, circular error, and typed rejection of projective flattening
and unapproved Clifford projection. The adapter has no production caller.

Acceptance and kill criteria are registered in
`experiments/verification/phase_codec_adapter_preregistration.md`.

### Stage 2 — channel-wise Sagnac diagnostic

Status: `HYPOTHESIS / BLOCKED`.

Add a default-off diagnostic only after Stage 1. Compare channel-wise
transmission against the existing scalar Sagnac control. Do not alter ranking,
EFE, MCTS, learning, or Zone C until the diagnostic discriminates candidates
and has a causal consumer.

### Stage 3 — Zone A/B/C ownership audit

Status: `HYPOTHESIS / BLOCKED`.

Trace every persistent object from definition to write, read, lifetime, and
external effect. Zone B must remain stateless. Zone C must own persistence and
must fail closed on live connection failure. No schema move is allowed without
separate approval.

### Stage 4 — structured compositional qFHRR ingress

Status: `HYPOTHESIS / BLOCKED`.

The current text codec geometry is a random-ring control, not a compositional
encoder. A future default-off structured encoder may use token or character
ngrams plus position binding. It must beat the random baseline on controlled
positive/negative pairs and held-out tasks before W_task tuning resumes.

### Stage 5 — online latent prediction

Status: `PARTIAL / BLOCKED`.

Audit the live caller for causal ordering, effective rank, young-fit blending,
differentiation, and improvement over identity. Internal coherence is not an
external outcome.

### Stage 6 — structured egress

Status: `PARTIAL / BLOCKED`.

Prefer program-level AST/CEGIS candidate generation and transformation-relative
ranking over a larger linear token head. Required decoder checkpoints and
external evaluator evidence remain mandatory for score-bearing runs.

### Stage 7 — weakness integration

Status: `COMPONENT COMPLETE / INTEGRATION BLOCKED`.

Integration requires a finite pre-decision continuation provider. Weakness is
extension cardinality only. It must not become entropy, complexity, or a global
score rescaling.

### Stage 8 — matched spherical/projective experiment

Status: `HYPOTHESIS / BLOCKED`.

Use matched controls, confidence intervals, and separate capacity and latency
criteria. A positive result requires a non-ceiling/non-floor separation under
registered conditions.

### Stage 9 — digital performance substrate

Status: `TARGET GOAL / BLOCKED`.

Profile reference semantics before kernel work. The 50 microsecond proposal
cycle and 20,000 proposals/second values are targets, not current telemetry.

### Stage 10 — hardware calibration

Status: `FUTURE / BLOCKED`.

Require voltage-to-phase, modulation, thermal, detector, optical-transfer, and
timing calibration. Software and hardware outputs must be compared under a
registered tolerance.

### Stage 11 — external VLA evidence

Status: `FUTURE / BLOCKED`.

Require authentic input, pre-action decision, environment action, external
outcome, post-action update, held-out evaluation, and complete evidence
artifacts. Report internal coherence, frame change, and task outcome separately.

## Governance gates

- Stage 1 implementation approval: `5775cf82-bf60-4e60-8282-2d65c7926352`.
- Stage 1 remote CUDA execution: not approved.
- Production caller wiring: not approved.
- Zone C mutation: not approved.
- Decoder checkpoint loading: not approved.
- Benchmark score promotion: blocked until dataset, evaluator, checkpoint, and
  item-level evidence gates pass.
