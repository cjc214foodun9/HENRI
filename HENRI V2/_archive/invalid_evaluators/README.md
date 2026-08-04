# Quarantined evaluators

The Python files in this directory are retained for audit and historical reference. They are not release evaluators and cannot produce external-performance claims.

They may contain generated prompts, synthetic fixtures, incomplete benchmark adapters, generic graders, or unsupported execution paths. Do not import them from production code. A future evaluator must pass the benchmark registry, dataset-digest, evaluator-digest, item-level outcome, and checkpoint gates before promotion.

The `henri_benchmark_gauntlet.py` file was moved here because its live imports do not match the current `zone_c_epistemic_axiom_harness.py` API. The correct status is `BLOCKED`, not a compatibility shim.
