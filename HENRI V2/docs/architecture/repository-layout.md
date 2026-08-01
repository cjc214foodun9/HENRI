# HENRI V2 repository layout

## Production root

The Python files at the HENRI V2 root are importable production modules and the two
current production entrypoints: `production_arc_run.py` and
`run_live_inference_eval.py`.

Do not move root modules into a package in this organization change. Current runtime
and remote commands use flat imports. Package migration requires its own change with
import updates and remote verification.

## Non-production code

- `experiments/verification/`: mathematical and GPU verification scripts.
- `experiments/performance/`: performance measurements.
- `experiments/sweeps/`: research sweeps.
- `experiments/exploratory/`: exploratory benchmark and physical-model scripts.
- `scripts/staging/`: corpus staging only.
- `scripts/training/`: decoder and SGLD training.
- `scripts/telemetry/`: telemetry synchronization.
- `scripts/maintenance/`: maintenance utilities.
- `tests/unit/`: unit and architectural tests.
- `tests/integration/`: progressive integration tests.
- `tests/contract/`: evidence, checkpoint, and fail-closed tests.
- `migrations/`: Zone C SQL.

## Invalid evaluator quarantine

`_archive/invalid_evaluators/` contains retained benchmark scripts that must not be
used for external performance claims. They contain synthetic data, toy paths,
generic graders, incomplete splits, or missing evaluator evidence.

The quarantine is deliberate. Do not restore a script to the production root.

## Artifact policy

Raw telemetry, checkpoints, caches, and local environment state remain outside the
source organization and are ignored by Git. Preserve their paths and hashes in a
run-evidence artifact when a result is eligible for promotion.
